"""Cached LLM client with a no-key dummy provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

from tsr.prompting.costs import make_usage_record


def _cache_key(provider: str, model: str, prompt: str, temperature: float) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def dummy_response(prompt: str) -> str:
    """Produce deterministic grounded-looking text from event-grounded prompts."""

    event_lines = re.findall(r"\{?'?event_id'?:?\s*'?([A-Za-z0-9_-]+)'?.*?\{?'?type'?:?\s*'?([A-Za-z_]+)'?.*?\{?'?start'?:?\s*(\d+).*?\{?'?end'?:?\s*(\d+)", prompt)
    if not event_lines:
        event_lines = re.findall(r'"event_id":\s*"([^"]+)".*?"type":\s*"([^"]+)".*?"start":\s*(\d+).*?"end":\s*(\d+)', prompt, flags=re.S)
    if event_lines:
        sentences = []
        for event_id, event_type, start, end in event_lines[:3]:
            pretty = event_type.replace("_", " ")
            sentences.append(f"[{event_id}] The series shows {pretty} from t={start} to t={end}.")
        return " ".join(sentences)
    return "The series contains a verifiable temporal pattern, but no extracted events were supplied."


def _openai_usage(usage) -> Dict[str, int]:
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def call_provider(provider: str, model: str, prompt: str, temperature: float = 0.0) -> Tuple[str, Dict[str, int]]:
    if provider == "dummy":
        return dummy_response(prompt), {}
    if provider in {"openai", "deepseek"}:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = None
            key_name = "OPENAI_API_KEY"
        else:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
            key_name = "DEEPSEEK_API_KEY"
        if not api_key:
            raise RuntimeError(f"{key_name} is not set. Use --provider dummy for no-key smoke tests.")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"Install the openai package to use provider={provider}.") from exc
        client_kwargs = {"api_key": api_key, "timeout": 120.0, "max_retries": 2}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or "", _openai_usage(getattr(response, "usage", None))
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use --provider dummy for no-key smoke tests.")
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the anthropic package to use provider=anthropic.") from exc
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=800,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        usage = getattr(response, "usage", None)
        usage_record = {}
        if usage is not None:
            usage_record = {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "total_tokens": int((getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)),
            }
        return "\n".join(text_parts), usage_record
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set. Use --provider dummy for no-key smoke tests.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed: {exc.code} {body}") from exc
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts)
        usage = data.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
        total_tokens = int(usage.get("totalTokenCount", 0) or 0)
        candidate_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
        output_tokens = max(candidate_tokens, total_tokens - prompt_tokens) if total_tokens else candidate_tokens
        usage_record = {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or prompt_tokens + output_tokens,
        }
        return text, usage_record
    raise ValueError(f"Unsupported provider: {provider}")


def cached_llm_call(
    provider: str,
    model: str,
    prompt: str,
    cache_dir: str,
    temperature: float = 0.0,
    cache_sensitive_prompts: bool = False,
    prices: Optional[Dict[str, dict]] = None,
    return_metadata: bool = False,
    max_retries: int = 2,
):
    """Call an LLM with content-addressed caching.

    By default the cache stores only the response and metadata hash, not the prompt text.
    """

    key = _cache_key(provider, model, prompt, temperature)
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "usage" not in payload:
            payload["usage"] = make_usage_record(prompt, payload.get("response", ""), model, prices)
        return payload if return_metadata else payload["response"]
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response, provider_usage = call_provider(provider, model, prompt, temperature)
            break
        except Exception as exc:  # noqa: BLE001 - provider clients raise heterogeneous exceptions.
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    else:
        raise RuntimeError("LLM call failed without an exception") from last_error
    payload = {
        "response": response,
        "provider": provider,
        "model": model,
        "usage": make_usage_record(prompt, response, model, prices, provider_usage),
    }
    if cache_sensitive_prompts:
        payload["prompt"] = prompt
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload if return_metadata else response

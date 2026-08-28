# Review Artifact Manifest

## Purpose

This directory is intended to be pushed as an anonymous under-review GitHub repository. It contains the minimum practical materials needed to inspect and rerun the paper's automatic evaluation without exposing private author, institution, API, or annotation information.

## Reproducibility Coverage

- Main synthetic 400-example evaluation: included.
- Multi-provider 80-example pilot diagnostics: included.
- Real-NAB localization diagnostics: included as processed windows and saved outputs.
- Verifier baselines and event-source diagnostics: included.
- Human audit: aggregate tables and summaries only; raw annotator files are excluded.

## Data-Minimization Choices

- Full raw provider cache files are excluded because they may contain prompts and request metadata.
- Raw third-party NAB files are excluded; processed diagnostic windows are included.
- Paper source files are excluded to avoid author metadata and local path leakage.
- Private annotation sheets are excluded to protect annotator identities and notes.

## Before Uploading

Run:

```bash
rg -n -i "sk-[A-Za-z0-9]|api[_-]?key|secret|password|token|/Users|KNOWN_AUTHOR_OR_INSTITUTION_TERMS|@[^ ]+" .
find . -name ".DS_Store" -print
```

The expected matches for API-related terms should be environment variable names in code or documentation, not real credential values.

# Architecture

This repository now keeps the original Chandra OCR smoke-test scripts and adds a typed `src/chandra_eval` package for resume-to-job-description evaluation.

The offline flow is:

1. Validate or create a structured candidate profile.
2. Validate or create a structured job description.
3. Build a scoring-safe candidate profile that excludes contact details and protected or non-job-related personal attributes.
4. Score each criterion with deterministic Python arithmetic.
5. Evaluate mandatory eligibility gates.
6. Write JSON, CSV, and HTML reports with `human_review_required=true`.

The Gemma and Chandra clients are configurable wrappers for Tyrone pilots. Unit and integration tests do not call live services.

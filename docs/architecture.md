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

## Pipeline Contract

The stable live contract is:

```bash
python scripts/run_full_resume_ocr.py resume.pdf \
  --output outputs/live/ocr/candidate-001

python scripts/extract_jd.py job.docx \
  --job-id marketing-coordinator \
  --output outputs/live/jd.json

python scripts/structure_resume.py \
  outputs/live/ocr/candidate-001 \
  --candidate-id candidate-001 \
  --output outputs/live/candidate.json

python scripts/evaluate_candidate.py \
  --jd outputs/live/jd.json \
  --resume outputs/live/candidate.json \
  --output outputs/live/evaluation.json

python scripts/build_report.py outputs/live --json --csv --html
```

`--mock-llm` is available only for offline tests and demos. Live commands use Gemma by default.

`run_full_resume_ocr.py` writes:

- `raw-responses.json`
- `page-001.html`, `page-001-metadata.json`, and subsequent pages
- `document.json`
- `source-map.json`
- `metadata.json`

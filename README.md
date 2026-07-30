# Chandra OCR-2 Evaluation

Evaluation pipeline for running Chandra OCR-2 on resumes and preparing a future resume-to-job-description matching workflow.

## Current status

- Chandra OCR-2 runs through the existing `llama.cpp` server
- Q8 GGUF community model
- F16 multimodal projector
- Tested on NVIDIA Tesla T4 GPU
- Initial smoke test: 3 out of 3 first pages successful
- Diverse layout test: 24 out of 24 first pages successful
- Output includes structured HTML, semantic block labels, and normalized bounding boxes

## Current OCR pipeline

    Resume PDF
        -> render PDF page as an image
        -> scale image for the vision model
        -> send image and OCR prompt to llama.cpp
        -> receive structured HTML
        -> save raw response, output HTML, and timing metadata

## Environment setup

Python 3.10 or newer is required.

Using uv:

    uv sync --frozen
    source .venv/bin/activate

## OCR server

The evaluation scripts expect an OpenAI-compatible multimodal server.

Default endpoint:

    http://127.0.0.1:18083

The actual model server runs on Tyrone and is not included in this repository.

## Project structure

    scripts/
        OCR evaluation, offline pilot, and reporting scripts

    src/chandra_eval/
        Typed resume/JD schemas, safety filtering, deterministic scoring,
        report writers, and configurable Chandra/Gemma clients

    tests/
        Offline synthetic fixtures and regression tests

    docs/
        Architecture, scoring, privacy/fairness, and Tyrone runbook notes

    inputs/
        Local resumes and job descriptions
        Not committed to Git

    outputs/
        Generated OCR results and reports
        Not committed to Git

    models/
        Local model weights
        Not committed to Git

    logs/
        Runtime logs
        Not committed to Git

## Data policy

Do not commit:

- Candidate resumes
- Job descriptions containing private information
- Candidate photographs or contact details
- OCR results containing candidate information
- Model weights such as GGUF or safetensors files
- API keys, tokens, passwords, or private SSH keys

## Next phase

The planned next phase is a resume-to-job-description evaluation pipeline:

    Full resume OCR
        -> structured candidate profile
        -> structured JD requirements
        -> eligibility checks
        -> criterion-level scores
        -> deterministic weighted final score
        -> evidence-backed candidate report

The scoring pipeline should use synthetic data for local development and testing. Real candidate data should remain on Tyrone.

## Offline synthetic pilot

The local Mac workflow does not require live Chandra or Gemma services:

    uv sync --frozen
    uv run python -m unittest discover -s tests
    uv run python scripts/run_matching_pilot.py

The pilot writes JSON, CSV, and HTML reports and always preserves `human_review_required=true`.

Live structuring commands use Gemma by default. Use `--mock-llm` only for offline tests and synthetic demos.

## End-to-end matching command

Quick start:

    uv sync --frozen
    source .venv/bin/activate
    uv run chandra-eval match

When run without `--jd` and `--resume`, the command checks configured services, lists supported
JD files from `JD_INPUT_DIR`, then lists supported resumes from `RESUME_INPUT_DIR`. Menus use
numbered selection plus `S` for search, `B` for back where applicable, and `Q` for clean
cancellation.

Non-interactive usage:

    uv run chandra-eval match \
      --jd "inputs/jds/Job description - Marketing Coordinator.docx" \
      --resume "inputs/resumes/Marketing/example.pdf" \
      --candidate-id candidate-marketing-001 \
      --job-id marketing-coordinator

The fallback script form is:

    uv run python scripts/match_resume_to_jd.py --jd "inputs/jds/job.docx" --resume "inputs/resumes/resume.pdf"

Each run creates a unique directory under `MATCH_OUTPUT_ROOT`, for example
`outputs/matches/marketing-coordinator/candidate-abc123/20260715-170500/`. The directory contains
structured JD/resume JSON, evaluation JSON/CSV/HTML, `run-manifest.json`, `run-summary.json`, and
`run-summary.md`.

Artifact reuse is based on source content hashes plus relevant model/prompt configuration. `--force`
regenerates stages, and `--no-cache` disables cache reads and writes. Reports are decision support
only; a qualified human reviewer must review the evidence before any employment decision.

Raw resume PDFs are OCR'd as full documents by default. `--ocr-pages` is an explicit page cap, not
the default. Partial OCR fails before scoring unless `--allow-partial-evaluation` is passed; allowed
partial runs force eligibility `REVIEW`, reduce evidence completeness, and add prominent warnings.
OCR metadata distinguishes attempted, completed, truncated, and failed pages. A page ending with
`finish_reason=length` is retried with the bounded Chandra retry token budget and is never treated
as complete merely because the page was attempted.

Private raw Chandra and Gemma model responses are not persisted unless
`PIPELINE_DEBUG_RAW_RESPONSES=true`. Normalized OCR HTML, metadata, document JSON, and source maps
remain available for audit.

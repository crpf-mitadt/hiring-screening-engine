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

# Scoring Design

The first scoring slice is deliberately deterministic. Gemma may be used later for structuring and criterion-level assessment, but final arithmetic is Python-owned.

The report preserves:

- criterion-level match fractions
- weighted criterion scores
- score before gates
- final score out of 100
- failed mandatory requirements
- missing or ambiguous information
- evidence completeness
- a recommendation label for human review only

Eligibility can be `PASS`, `REVIEW`, or `FAIL`. A `FAIL` does not delete the candidate, suppress the score, or make a final hiring decision.

Status semantics:

- `met`: affirmative resume evidence supports the requirement or criterion.
- `partially_met`: some required parts are supported and other parts are missing or unclear.
- `not_found`: the resume provides no evidence either way.
- `unclear`: evidence exists but is ambiguous, incomplete, or affected by extraction uncertainty.
- `not_met`: deterministic arithmetic or explicit evidence shows the requirement is below the minimum.
- `not_applicable`: the criterion has no applicable evidence terms.

Missing evidence is not a hard failure by itself. Mandatory eligibility uses:

- `PASS`: all mandatory requirements are supported.
- `REVIEW`: at least one mandatory requirement is unclear, partially supported, or not found, with no definitive failure.
- `FAIL`: at least one mandatory requirement is clearly contradicted or definitively below a required threshold.

Experience duration is overlap-adjusted. Confirmed months are calculated only from independently
parseable ranges. Malformed dates such as a truncated year are preserved as uncertain ranges and do
not erase duration from valid entries.

Criterion matching uses the scoring-safe profile only. It searches summary, education, experience
titles and responsibilities, skills used, top-level skills, projects, publications, teaching,
research, leadership, and achievements. Exact/normalized matches are combined with a narrow alias
map for auditable business terms such as MS Office, CRM, lead generation, and campaign management.

Degree requirements are source-constrained: they may match only structured education entries and
their evidence. Experience paragraphs, job titles, generic ML/AI mentions, leadership text, company
names, and technical operations text cannot become degree evidence.

`accepted_evidence` describes later verification sources such as interviews, coding tests,
transcripts, resumes, portfolios, or publication lists. It is not a candidate qualification and is
not included in match terms, missing qualifications, gaps, or match-fraction denominators. These
items appear as `verification_needed`.

Final criterion `match_fraction` values are discrete: `0.00`, `0.25`, `0.50`, `0.75`, or `1.00`.
Weighted scores are `criterion.weight * match_fraction`; Python owns this arithmetic.

Evidence completeness is deterministic:

1. Compute the fraction of evaluated criteria with at least one supported or partially supported,
   criterion-relevant resume evidence item. Duplicate copies of the same evidence do not increase
   the numerator.
2. If that fraction is zero, completeness is zero.
3. Otherwise, compute `criteria_fraction * 0.70 + date_score * 0.15 + ocr_score * 0.15`.
4. `date_score` is `1.0` when experience dates are complete and `0.5` when date warnings exist.
5. `ocr_score` is `1.0` for full OCR coverage and `0.5` for explicitly allowed partial OCR.

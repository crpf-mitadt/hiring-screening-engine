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
Evidence completeness is the fraction of evaluated criteria with at least one supported or partially
supported resume evidence item.

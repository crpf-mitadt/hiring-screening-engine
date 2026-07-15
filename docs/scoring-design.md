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

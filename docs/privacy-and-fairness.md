# Privacy And Fairness

Scoring payloads must use internal candidate IDs such as `candidate-001`.

Do not use or infer protected and non-job-related attributes, including photograph, name, gender, age, date of birth, marital status, religion, caste, race, ethnicity, disability, health information, pregnancy status, sexual orientation, political beliefs, home address, phone number, or personal email.

The `ScoringSafeProfile` schema intentionally excludes `personal_information`. Reports always keep `human_review_required=true` and include a decision-support disclaimer.

import unittest

from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.schemas.common import DatePrecision
from chandra_eval.schemas.resume import CandidateProfile


class ExperienceDateTests(unittest.TestCase):
    def test_common_live_date_formats_survive_with_month_precision(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-dates",
                "source_document_id": "ocr",
                "experience": [
                    {
                        "title": "Vice President",
                        "start_date": "08/2023",
                        "end_date": "Present",
                        "evidence": [
                            {
                                "source_type": "resume_ocr",
                                "source_document_id": "ocr",
                                "page_number": 1,
                                "block_index": 1,
                                "exact_text": "Vice President: 08/2023 - Present",
                            }
                        ],
                    },
                    {
                        "title": "Global Director",
                        "start_date": "05/2021",
                        "end_date": "08/2023",
                    },
                    {
                        "title": "GBS India Site Lead",
                        "start_date": "07/2018",
                        "end_date": "04/2021",
                    },
                ],
            }
        )

        entries = candidate.experience
        self.assertEqual(entries[0].start_date, "2023-08")
        self.assertIsNone(entries[0].end_date)
        self.assertTrue(entries[0].currently_active)
        self.assertEqual(entries[0].start_date_precision, DatePrecision.MONTH)
        self.assertEqual(entries[1].start_date, "2021-05")
        self.assertEqual(entries[1].end_date, "2023-08")
        self.assertEqual(entries[2].start_date, "2018-07")
        self.assertEqual(entries[2].end_date, "2021-04")

        duration = calculate_experience_duration(entries)

        self.assertGreater(duration.confirmed_months, 0)
        self.assertEqual(duration.uncertain_ranges, [])
        self.assertEqual(duration.date_warnings, [])

    def test_malformed_range_remains_uncertain_without_erasing_valid_duration(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-dates",
                "source_document_id": "ocr",
                "experience": [
                    {"start_date": "05-2021", "end_date": "08-2023"},
                    {"start_date": "2025-01", "end_date": "202"},
                    {"start_date": "2022-01", "end_date": "2022-12"},
                ],
            }
        )

        duration = calculate_experience_duration(candidate.experience)

        self.assertGreaterEqual(duration.confirmed_months, 27)
        self.assertEqual(len(duration.uncertain_ranges), 1)
        self.assertIn("ambiguous end_date", duration.date_warnings[0])

    def test_three_component_slash_date_is_not_month_precision(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-dates",
                "source_document_id": "ocr",
                "experience": [
                    {
                        "start_date": "08/07/17",
                        "end_date": "2018-07",
                        "start_date_precision": "month",
                    }
                ],
            }
        )

        entry = candidate.experience[0]
        self.assertEqual(entry.start_date, "08/07/17")
        self.assertEqual(entry.start_date_precision, DatePrecision.UNKNOWN)
        self.assertIn("Ambiguous start_date", entry.warnings[0])

    def test_overlapping_ranges_are_not_double_counted(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-overlap",
                "source_document_id": "ocr",
                "experience": [
                    {"start_date": "2021-01", "end_date": "2021-12"},
                    {"start_date": "2021-06", "end_date": "2022-06"},
                ],
            }
        )

        duration = calculate_experience_duration(candidate.experience)

        self.assertEqual(duration.confirmed_months, 17)


if __name__ == "__main__":
    unittest.main()

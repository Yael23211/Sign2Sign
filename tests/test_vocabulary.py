from pathlib import Path
import unittest

from src.text_processing import Vocabulary


ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


class VocabularyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary(VOCAB_PATH)

    def test_total_concepts(self):
        self.assertEqual(len(self.vocabulary), 71)

    def test_atomic_count(self):
        self.assertEqual(
            len(self.vocabulary.get_by_type("atomic")),
            65,
        )

    def test_expression_count(self):
        self.assertEqual(
            len(self.vocabulary.get_by_type("expression")),
            6,
        )

    def test_expected_concepts_exist(self):
        for concept_id in (
            "HELLO",
            "WANT",
            "NOT",
            "KNOW_INFORMATION",
            "KNOW_PERSON",
            "EXIST",
            "HOW_ARE_YOU",
        ):
            self.assertTrue(
                self.vocabulary.has_concept(concept_id)
            )

    def test_negative_forms_are_not_canonical_ids(self):
        for concept_id in (
            "NO_QUERER",
            "NO_PODER",
            "NO_SABER",
            "NO_ENTENDER",
            "NO_GUSTAR",
            "NO_CONOCER",
        ):
            self.assertFalse(
                self.vocabulary.has_concept(concept_id)
            )

    def test_prompt_catalog_has_all_concepts(self):
        catalog = self.vocabulary.get_prompt_catalog()
        self.assertEqual(len(catalog), 71)


if __name__ == "__main__":
    unittest.main()

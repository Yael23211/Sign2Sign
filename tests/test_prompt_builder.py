from pathlib import Path
import unittest

from src.text import (
    BuiltText,
    PredictionCandidate,
    TextToken,
)

from src.text_processing import (
    CorrectionRequest,
    PromptBuilder,
    TranslationResult,
    Vocabulary,
)


ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


class PromptBuilderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary(
            VOCAB_PATH
        )

        cls.prompt_builder = PromptBuilder(
            cls.vocabulary
        )

    def setUp(self):
        self.built_text = BuiltText(
            source_language="LSM",
            text="AGUA",
            tokens=(
                TextToken(
                    kind="letter",
                    value="A",
                    top_k=(
                        PredictionCandidate(
                            "A",
                            0.95,
                        ),
                        PredictionCandidate(
                            "E",
                            0.03,
                        ),
                        PredictionCandidate(
                            "O",
                            0.02,
                        ),
                    ),
                ),
                TextToken(
                    kind="letter",
                    value="G",
                    top_k=(
                        PredictionCandidate(
                            "G",
                            0.90,
                        ),
                        PredictionCandidate(
                            "H",
                            0.07,
                        ),
                        PredictionCandidate(
                            "Q",
                            0.03,
                        ),
                    ),
                ),
                TextToken(
                    kind="letter",
                    value="U",
                    top_k=(
                        PredictionCandidate(
                            "U",
                            0.54,
                        ),
                        PredictionCandidate(
                            "R",
                            0.43,
                        ),
                        PredictionCandidate(
                            "V",
                            0.03,
                        ),
                    ),
                ),
                TextToken(
                    kind="letter",
                    value="A",
                    top_k=(
                        PredictionCandidate(
                            "A",
                            0.97,
                        ),
                        PredictionCandidate(
                            "E",
                            0.02,
                        ),
                        PredictionCandidate(
                            "O",
                            0.01,
                        ),
                    ),
                ),
            ),
        )

        self.request = CorrectionRequest.from_built_text(
            self.built_text,
            target_language="ASL",
        )

    def test_request_languages(self):
        self.assertEqual(
            self.request.source_language,
            "LSM",
        )

        self.assertEqual(
            self.request.target_language,
            "ASL",
        )

    def test_bridge_languages(self):
        self.assertEqual(
            self.request.source_bridge_language,
            "Spanish",
        )

        self.assertEqual(
            self.request.target_bridge_language,
            "English",
        )

    def test_same_language_is_rejected(self):
        with self.assertRaises(ValueError):
            CorrectionRequest.from_built_text(
                self.built_text,
                target_language="LSM",
            )

    def test_prompt_contains_recognized_text(self):
        prompt = self.prompt_builder.build(
            self.request
        )

        self.assertIn(
            "AGUA",
            prompt,
        )

    def test_prompt_contains_top_k(self):
        prompt = self.prompt_builder.build(
            self.request
        )

        self.assertIn(
            "U: 0.5400",
            prompt,
        )

        self.assertIn(
            "R: 0.4300",
            prompt,
        )

    def test_prompt_contains_vocabulary(self):
        prompt = self.prompt_builder.build(
            self.request
        )

        self.assertIn(
            '"concept_id": "WATER"',
            prompt,
        )

        self.assertIn(
            '"concept_id": "HOW_ARE_YOU"',
            prompt,
        )

    def test_all_vocabulary_ids_are_in_prompt(self):
        prompt = self.prompt_builder.build(
            self.request
        )

        for concept_id in self.vocabulary.get_allowed_ids():
            self.assertIn(
                concept_id,
                prompt,
            )

    def test_prompt_contains_output_schema(self):
        prompt = self.prompt_builder.build(
            self.request
        )

        self.assertIn(
            '"corrected_text"',
            prompt,
        )

        self.assertIn(
            '"translated_text"',
            prompt,
        )

        self.assertIn(
            '"concepts"',
            prompt,
        )

        self.assertIn(
            '"unresolved"',
            prompt,
        )

    def test_prompt_does_not_include_source_traceability(self):
        prompt = self.prompt_builder.build(
            self.request
        )

        self.assertNotIn(
            "original_lsm_es",
            prompt,
        )

        self.assertNotIn(
            "original_asl_en",
            prompt,
        )

    def test_translation_result_serialization(self):
        result = TranslationResult(
            source_language="LSM",
            target_language="ASL",
            recognized_text="AGUA",
            corrected_text="AGUA",
            translated_text="WATER",
            concepts=(
                "WATER",
            ),
            unresolved=(),
        )

        data = result.to_dict()

        self.assertEqual(
            data["concepts"],
            ["WATER"],
        )

        self.assertEqual(
            data["unresolved"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
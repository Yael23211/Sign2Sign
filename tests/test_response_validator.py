from pathlib import Path
import json
import unittest

from src.text import (
    BuiltText,
    PredictionCandidate,
    TextToken,
)

from src.text_processing import (
    CorrectionRequest,
    ResponseValidator,
    ResponseValidationError,
    Vocabulary,
)


ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


class ResponseValidatorTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.vocabulary = Vocabulary(
            VOCAB_PATH
        )

        cls.validator = ResponseValidator(
            cls.vocabulary
        )

    def setUp(self):

        built_text = BuiltText(
            source_language="LSM",
            text="AGUA",
            tokens=(
                TextToken(
                    kind="letter",
                    value="A",
                    top_k=(
                        PredictionCandidate(
                            "A",
                            0.98,
                        ),
                    ),
                ),
            ),
        )

        self.request = (
            CorrectionRequest.from_built_text(
                built_text,
                target_language="ASL",
            )
        )

        self.valid_response = {
            "corrected_text":
                "AGUA",

            "translated_text":
                "WATER",

            "concepts": [
                "WATER",
            ],

            "unresolved": [],
        }

    # ========================================================
    # VALID RESPONSES
    # ========================================================

    def test_valid_dictionary_response(self):

        result = self.validator.validate(
            self.valid_response,
            self.request,
        )

        self.assertEqual(
            result.corrected_text,
            "AGUA",
        )

        self.assertEqual(
            result.translated_text,
            "WATER",
        )

        self.assertEqual(
            result.concepts,
            (
                "WATER",
            ),
        )

    def test_valid_json_string_response(self):

        raw = json.dumps(
            self.valid_response
        )

        result = self.validator.validate(
            raw,
            self.request,
        )

        self.assertEqual(
            result.concepts,
            (
                "WATER",
            ),
        )

    # ========================================================
    # UNKNOWN CONCEPTS
    # ========================================================

    def test_unknown_concept_is_rejected(self):

        response = {
            "corrected_text":
                "AGUA BONITA",

            "translated_text":
                "BEAUTIFUL WATER",

            "concepts": [
                "WATER",
                "BEAUTIFUL",
            ],

            "unresolved": [],
        }

        with self.assertRaises(
            ResponseValidationError
        ):

            self.validator.validate(
                response,
                self.request,
            )

    # ========================================================
    # REQUIRED FIELDS
    # ========================================================

    def test_missing_field_is_rejected(self):

        response = {
            "corrected_text":
                "AGUA",

            "translated_text":
                "WATER",

            "concepts": [
                "WATER",
            ],
        }

        with self.assertRaises(
            ResponseValidationError
        ):

            self.validator.validate(
                response,
                self.request,
            )

    # ========================================================
    # INVALID JSON
    # ========================================================

    def test_invalid_json_is_rejected(self):

        raw = """
        {
          "corrected_text": "AGUA",
          "translated_text": "WATER"
        """

        with self.assertRaises(
            ResponseValidationError
        ):

            self.validator.validate(
                raw,
                self.request,
            )

    # ========================================================
    # EMPTY TEXT WITH RESOLVED CONCEPTS
    # ========================================================

    def test_empty_corrected_text_is_rejected_when_concepts_exist(
        self,
    ):

        response = {
            "corrected_text":
                "",

            "translated_text":
                "WATER",

            "concepts": [
                "WATER",
            ],

            "unresolved": [],
        }

        with self.assertRaises(
            ResponseValidationError
        ):

            self.validator.validate(
                response,
                self.request,
            )

    # ========================================================
    # CONCEPT TYPE
    # ========================================================

    def test_concepts_must_be_list(self):

        response = {
            "corrected_text":
                "AGUA",

            "translated_text":
                "WATER",

            "concepts":
                "WATER",

            "unresolved": [],
        }

        with self.assertRaises(
            ResponseValidationError
        ):

            self.validator.validate(
                response,
                self.request,
            )

    # ========================================================
    # PARTIALLY UNRESOLVED
    # ========================================================

    def test_unresolved_is_allowed(self):

        response = {
            "corrected_text":
                "AGUA XYZ",

            "translated_text":
                "WATER XYZ",

            "concepts": [
                "WATER",
            ],

            "unresolved": [
                "XYZ",
            ],
        }

        result = self.validator.validate(
            response,
            self.request,
        )

        self.assertEqual(
            result.concepts,
            (
                "WATER",
            ),
        )

        self.assertEqual(
            result.unresolved,
            (
                "XYZ",
            ),
        )

    # ========================================================
    # ONLY UNRESOLVED WITH TEXT
    # ========================================================

    def test_only_unresolved_is_allowed(self):

        response = {
            "corrected_text":
                "XYZ",

            "translated_text":
                "XYZ",

            "concepts": [],

            "unresolved": [
                "XYZ",
            ],
        }

        result = self.validator.validate(
            response,
            self.request,
        )

        self.assertEqual(
            result.concepts,
            (),
        )

        self.assertEqual(
            result.unresolved,
            (
                "XYZ",
            ),
        )

    # ========================================================
    # ONLY UNRESOLVED WITH EMPTY TEXT
    #
    # This is the new case discovered during the real
    # robustness test with ZXQW.
    # ========================================================

    def test_empty_text_is_allowed_when_everything_is_unresolved(
        self,
    ):

        response = {
            "corrected_text":
                "",

            "translated_text":
                "",

            "concepts": [],

            "unresolved": [
                "ZXQW",
            ],
        }

        result = self.validator.validate(
            response,
            self.request,
        )

        self.assertEqual(
            result.corrected_text,
            "",
        )

        self.assertEqual(
            result.translated_text,
            "",
        )

        self.assertEqual(
            result.concepts,
            (),
        )

        self.assertEqual(
            result.unresolved,
            (
                "ZXQW",
            ),
        )

    # ========================================================
    # NOTHING RESOLVED AND NOTHING REPORTED
    # ========================================================

    def test_empty_concepts_and_unresolved_are_rejected(
        self,
    ):

        response = {
            "corrected_text":
                "",

            "translated_text":
                "",

            "concepts": [],

            "unresolved": [],
        }

        with self.assertRaises(
            ResponseValidationError
        ):

            self.validator.validate(
                response,
                self.request,
            )

    # ========================================================
    # EXPRESSION
    # ========================================================

    def test_expression_concept_is_valid(self):

        response = {
            "corrected_text":
                "CÓMO ESTÁS",

            "translated_text":
                "HOW ARE YOU",

            "concepts": [
                "HOW_ARE_YOU",
            ],

            "unresolved": [],
        }

        result = self.validator.validate(
            response,
            self.request,
        )

        self.assertEqual(
            result.concepts,
            (
                "HOW_ARE_YOU",
            ),
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def test_concept_ids_are_normalized_to_uppercase(
        self,
    ):

        response = {
            "corrected_text":
                "AGUA",

            "translated_text":
                "WATER",

            "concepts": [
                "water",
            ],

            "unresolved": [],
        }

        result = self.validator.validate(
            response,
            self.request,
        )

        self.assertEqual(
            result.concepts,
            (
                "WATER",
            ),
        )


if __name__ == "__main__":
    unittest.main()
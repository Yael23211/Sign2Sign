import unittest

from src.text import EmptyTextError, TextBuilder


class TestTextBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = TextBuilder(
            source_language="LSM",
            top_k_size=3,
        )

    def test_initial_state(self):
        self.assertEqual(
            self.builder.source_language,
            "LSM",
        )

        self.assertEqual(
            self.builder.current_text,
            "",
        )

        self.assertTrue(
            self.builder.is_empty
        )

        self.assertEqual(
            self.builder.letter_count,
            0,
        )

        self.assertEqual(
            self.builder.word_count,
            0,
        )

    def test_add_single_prediction(self):
        token = self.builder.add_prediction(
            letter="H",
            confidence=0.98,
            top_k=[
                ("H", 0.98),
                ("N", 0.01),
                ("M", 0.01),
            ],
        )

        self.assertEqual(
            token.value,
            "H",
        )

        self.assertEqual(
            token.kind,
            "letter",
        )

        self.assertEqual(
            self.builder.current_text,
            "H",
        )

        self.assertEqual(
            self.builder.letter_count,
            1,
        )

        self.assertEqual(
            self.builder.word_count,
            1,
        )

    def test_build_word(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter=letter,
                confidence=0.90,
            )

        self.assertEqual(
            self.builder.current_text,
            "HOLA",
        )

        self.assertEqual(
            self.builder.letter_count,
            4,
        )

        self.assertEqual(
            self.builder.word_count,
            1,
        )

    def test_build_sentence(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter=letter,
                confidence=0.90,
            )

        inserted = self.builder.add_space()

        self.assertTrue(inserted)

        for letter in "MUNDO":
            self.builder.add_prediction(
                letter=letter,
                confidence=0.90,
            )

        self.assertEqual(
            self.builder.current_text,
            "HOLA MUNDO",
        )

        self.assertEqual(
            self.builder.letter_count,
            9,
        )

        self.assertEqual(
            self.builder.word_count,
            2,
        )

    def test_no_leading_space(self):
        inserted = self.builder.add_space()

        self.assertFalse(inserted)

        self.assertEqual(
            self.builder.current_text,
            "",
        )

    def test_no_repeated_spaces(self):
        self.builder.add_prediction(
            "A",
            0.95,
        )

        first = self.builder.add_space()
        second = self.builder.add_space()

        self.assertTrue(first)
        self.assertFalse(second)

        self.assertEqual(
            self.builder.current_text,
            "A ",
        )

    def test_backspace_letter(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        removed = self.builder.backspace()

        self.assertIsNotNone(removed)

        self.assertEqual(
            removed.value,
            "A",
        )

        self.assertEqual(
            self.builder.current_text,
            "HOL",
        )

    def test_backspace_space(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        self.builder.add_space()

        removed = self.builder.backspace()

        self.assertIsNotNone(removed)

        self.assertEqual(
            removed.kind,
            "space",
        )

        self.assertEqual(
            self.builder.current_text,
            "HOLA",
        )

    def test_backspace_empty_sequence(self):
        removed = self.builder.backspace()

        self.assertIsNone(removed)

    def test_clear(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        self.builder.add_space()

        for letter in "MUNDO":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        self.builder.clear()

        self.assertEqual(
            self.builder.current_text,
            "",
        )

        self.assertTrue(
            self.builder.is_empty
        )

        self.assertEqual(
            len(self.builder.tokens),
            0,
        )

    def test_finalize(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        self.builder.add_space()

        for letter in "MUNDO":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        result = self.builder.finalize()

        self.assertEqual(
            result.source_language,
            "LSM",
        )

        self.assertEqual(
            result.text,
            "HOLA MUNDO",
        )

        self.assertEqual(
            result.character_count,
            9,
        )

        self.assertEqual(
            result.word_count,
            2,
        )

    def test_finalize_removes_trailing_space(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        self.builder.add_space()

        result = self.builder.finalize()

        self.assertEqual(
            result.text,
            "HOLA",
        )

        self.assertEqual(
            result.word_count,
            1,
        )

    def test_finalize_does_not_clear_builder(self):
        for letter in "HOLA":
            self.builder.add_prediction(
                letter,
                0.90,
            )

        result = self.builder.finalize()

        self.assertEqual(
            result.text,
            "HOLA",
        )

        self.assertEqual(
            self.builder.current_text,
            "HOLA",
        )

    def test_finalize_empty_sequence(self):
        with self.assertRaises(
            EmptyTextError
        ):
            self.builder.finalize()

    def test_top_k_is_preserved(self):
        self.builder.add_prediction(
            letter="K",
            confidence=0.48,
            top_k=[
                ("K", 0.48),
                ("P", 0.44),
                ("X", 0.05),
            ],
        )

        result = self.builder.finalize()

        token = result.tokens[0]

        self.assertEqual(
            token.value,
            "K",
        )

        self.assertEqual(
            len(token.top_k),
            3,
        )

        self.assertEqual(
            token.top_k[0].label,
            "K",
        )

        self.assertEqual(
            token.top_k[1].label,
            "P",
        )

        self.assertEqual(
            token.top_k[2].label,
            "X",
        )

    def test_top_k_size_limit(self):
        builder = TextBuilder(
            source_language="ASL",
            top_k_size=3,
        )

        builder.add_prediction(
            letter="A",
            confidence=0.50,
            top_k=[
                ("A", 0.50),
                ("S", 0.20),
                ("T", 0.15),
                ("N", 0.10),
                ("M", 0.05),
            ],
        )

        token = builder.tokens[0]

        self.assertEqual(
            len(token.top_k),
            3,
        )

    def test_prediction_is_normalized_to_uppercase(self):
        self.builder.add_prediction(
            letter="h",
            confidence=0.90,
        )

        self.assertEqual(
            self.builder.current_text,
            "H",
        )

    def test_invalid_language(self):
        with self.assertRaises(ValueError):
            TextBuilder(
                source_language="XYZ"
            )

    def test_invalid_letter(self):
        with self.assertRaises(ValueError):
            self.builder.add_prediction(
                letter="1",
                confidence=0.90,
            )

    def test_multiple_letters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.add_prediction(
                letter="AB",
                confidence=0.90,
            )

    def test_invalid_probability_above_one(self):
        with self.assertRaises(ValueError):
            self.builder.add_prediction(
                letter="A",
                confidence=1.50,
            )

    def test_invalid_probability_below_zero(self):
        with self.assertRaises(ValueError):
            self.builder.add_prediction(
                letter="A",
                confidence=-0.10,
            )

    def test_serializable_output(self):
        self.builder.add_prediction(
            letter="K",
            confidence=0.48,
            top_k=[
                ("K", 0.48),
                ("P", 0.44),
                ("X", 0.05),
            ],
        )

        result = self.builder.finalize()

        data = result.to_dict()

        self.assertEqual(
            data["source_language"],
            "LSM",
        )

        self.assertEqual(
            data["text"],
            "K",
        )

        self.assertEqual(
            data["character_count"],
            1,
        )

        self.assertEqual(
            data["word_count"],
            1,
        )

        self.assertEqual(
            data["tokens"][0]["value"],
            "K",
        )

        self.assertEqual(
            data["tokens"][0]["top_k"][1]["label"],
            "P",
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
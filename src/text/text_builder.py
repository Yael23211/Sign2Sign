from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SUPPORTED_LANGUAGES = {"ASL", "LSM"}
SUPPORTED_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class EmptyTextError(ValueError):
    """Raised when an empty sequence is finalized."""


@dataclass(frozen=True)
class PredictionCandidate:
    """
    One candidate produced by the classifier.

    Example:
        PredictionCandidate(label="H", probability=0.984)
    """

    label: str
    probability: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class TextToken:
    """
    One element in the recognized sequence.

    kind:
        "letter" -> character recognized by the classifier.
        "space"  -> word separator inserted by the user.

    For letter tokens, top_k stores the alternatives produced
    by the classifier.
    """

    kind: str
    value: str
    top_k: tuple[PredictionCandidate, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "top_k": [
                candidate.to_dict()
                for candidate in self.top_k
            ],
        }


@dataclass(frozen=True)
class BuiltText:
    """
    Immutable result produced when the user finalizes the sequence.

    It contains:
    - source language;
    - visible text;
    - complete token sequence;
    - Top-k information for every recognized letter.
    """

    source_language: str
    text: str
    tokens: tuple[TextToken, ...]

    @property
    def character_count(self) -> int:
        return sum(
            1
            for token in self.tokens
            if token.kind == "letter"
        )

    @property
    def word_count(self) -> int:
        if not self.text:
            return 0

        return len(self.text.split())

    def to_dict(self) -> dict:
        """
        Serializable representation.

        This structure can later be used as the input generated
        by SC-03 for the following subsystem.
        """

        return {
            "source_language": self.source_language,
            "text": self.text,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "tokens": [
                token.to_dict()
                for token in self.tokens
            ],
        }


class TextBuilder:
    """
    Builds the textual sequence produced by the sign recognizer.

    This component does not perform:
    - image capture;
    - hand detection;
    - landmark extraction;
    - classification;
    - spelling correction;
    - translation.

    It only manages the sequence returned by the recognizer.

    Expected UI mapping:

        ENTER      -> add_prediction(...)
        SPACE      -> add_space()
        BACKSPACE  -> backspace()
        ESC        -> clear()
        T          -> finalize()
    """

    def __init__(
        self,
        source_language: str,
        top_k_size: int = 3,
    ) -> None:

        language = source_language.upper().strip()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported source language: {source_language}. "
                f"Expected one of: "
                f"{sorted(SUPPORTED_LANGUAGES)}"
            )

        if top_k_size < 1:
            raise ValueError(
                "top_k_size must be at least 1."
            )

        self._source_language = language
        self._top_k_size = top_k_size
        self._tokens: list[TextToken] = []

    @property
    def source_language(self) -> str:
        return self._source_language

    @property
    def tokens(self) -> tuple[TextToken, ...]:
        """
        Returns an immutable view of the current sequence.
        """

        return tuple(self._tokens)

    @property
    def current_text(self) -> str:
        """
        Text currently visible to the user.
        """

        return "".join(
            token.value
            for token in self._tokens
        )

    @property
    def is_empty(self) -> bool:
        return not any(
            token.kind == "letter"
            for token in self._tokens
        )

    @property
    def letter_count(self) -> int:
        return sum(
            1
            for token in self._tokens
            if token.kind == "letter"
        )

    @property
    def word_count(self) -> int:
        text = self.current_text.strip()

        if not text:
            return 0

        return len(text.split())

    def add_prediction(
        self,
        letter: str,
        confidence: float,
        top_k: Sequence[tuple[str, float]] | None = None,
    ) -> TextToken:
        """
        Adds one recognized letter to the sequence.

        Parameters
        ----------
        letter:
            Top-1 prediction returned by the classifier.

        confidence:
            Probability associated with the Top-1 prediction.

        top_k:
            Optional classifier alternatives.

            Example:
                [
                    ("H", 0.98),
                    ("N", 0.01),
                    ("M", 0.01),
                ]

        Returns
        -------
        TextToken
            Token added to the sequence.
        """

        normalized_letter = self._normalize_letter(letter)
        normalized_confidence = self._validate_probability(
            confidence
        )

        candidate_probabilities: dict[str, float] = {}

        if top_k is not None:
            for candidate_letter, probability in top_k:

                candidate_letter = self._normalize_letter(
                    candidate_letter
                )

                probability = self._validate_probability(
                    probability
                )

                previous = candidate_probabilities.get(
                    candidate_letter
                )

                if (
                    previous is None
                    or probability > previous
                ):
                    candidate_probabilities[
                        candidate_letter
                    ] = probability

        # The selected Top-1 prediction is always preserved.
        candidate_probabilities[
            normalized_letter
        ] = normalized_confidence

        other_candidates = [
            PredictionCandidate(
                label=candidate_letter,
                probability=probability,
            )
            for candidate_letter, probability
            in candidate_probabilities.items()
            if candidate_letter != normalized_letter
        ]

        other_candidates.sort(
            key=lambda candidate: candidate.probability,
            reverse=True,
        )

        candidates = [
            PredictionCandidate(
                label=normalized_letter,
                probability=normalized_confidence,
            )
        ]

        candidates.extend(
            other_candidates[
                : max(0, self._top_k_size - 1)
            ]
        )

        token = TextToken(
            kind="letter",
            value=normalized_letter,
            top_k=tuple(candidates),
        )

        self._tokens.append(token)

        return token

    def add_space(self) -> bool:
        """
        Adds a word separator.

        Leading spaces and consecutive spaces are ignored.

        Returns
        -------
        bool
            True if the space was inserted.
            False otherwise.
        """

        if not self._tokens:
            return False

        if self._tokens[-1].kind == "space":
            return False

        self._tokens.append(
            TextToken(
                kind="space",
                value=" ",
            )
        )

        return True

    def backspace(self) -> TextToken | None:
        """
        Removes the last element in the sequence.

        It can remove either:
        - a letter;
        - a space.

        Returns the removed token or None if the sequence
        was already empty.
        """

        if not self._tokens:
            return None

        return self._tokens.pop()

    def clear(self) -> None:
        """
        Clears the complete current sequence.
        """

        self._tokens.clear()

    def finalize(self) -> BuiltText:
        """
        Creates the final immutable text representation.

        Trailing spaces are removed from the returned result.

        Important:
        finalize() does NOT clear the builder.

        This is intentional: if the next subsystem fails,
        the recognized text remains available and can be sent
        again instead of forcing the user to repeat the signs.
        """

        trimmed_tokens = list(self._tokens)

        while (
            trimmed_tokens
            and trimmed_tokens[-1].kind == "space"
        ):
            trimmed_tokens.pop()

        if not any(
            token.kind == "letter"
            for token in trimmed_tokens
        ):
            raise EmptyTextError(
                "Cannot finalize an empty text sequence."
            )

        text = "".join(
            token.value
            for token in trimmed_tokens
        )

        return BuiltText(
            source_language=self._source_language,
            text=text,
            tokens=tuple(trimmed_tokens),
        )

    @staticmethod
    def _normalize_letter(letter: str) -> str:
        """
        Validates and normalizes one classifier label.
        """

        if not isinstance(letter, str):
            raise TypeError(
                "Letter must be a string."
            )

        normalized = letter.strip().upper()

        if len(normalized) != 1:
            raise ValueError(
                f"Expected exactly one letter, got: {letter!r}"
            )

        if normalized not in SUPPORTED_LETTERS:
            raise ValueError(
                f"Unsupported letter: {normalized!r}"
            )

        return normalized

    @staticmethod
    def _validate_probability(
        probability: float,
    ) -> float:
        """
        Validates a classifier probability.
        """

        try:
            value = float(probability)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Probability must be numeric."
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Probability must be between 0 and 1. "
                f"Received: {value}"
            )

        return value
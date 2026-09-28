from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.text import BuiltText


SUPPORTED_SIGN_LANGUAGES = {
    "ASL",
    "LSM",
}


BRIDGE_LANGUAGES = {
    "ASL": "English",
    "LSM": "Spanish",
}


@dataclass(frozen=True)
class CorrectionRequest:
    """
    Input contract for SC-04.

    Represents the structured text produced by SC-03 together
    with the requested target sign language.
    """

    source_language: str
    target_language: str
    recognized_text: str
    tokens: tuple[dict[str, Any], ...]

    def __post_init__(self):

        source = self.source_language.upper()
        target = self.target_language.upper()

        if source not in SUPPORTED_SIGN_LANGUAGES:
            raise ValueError(
                f"Unsupported source language: "
                f"{self.source_language}"
            )

        if target not in SUPPORTED_SIGN_LANGUAGES:
            raise ValueError(
                f"Unsupported target language: "
                f"{self.target_language}"
            )

        if source == target:
            raise ValueError(
                "Source and target languages "
                "must be different."
            )

        if not self.recognized_text.strip():
            raise ValueError(
                "recognized_text cannot be empty."
            )

        object.__setattr__(
            self,
            "source_language",
            source,
        )

        object.__setattr__(
            self,
            "target_language",
            target,
        )

    @classmethod
    def from_built_text(
        cls,
        built_text: BuiltText,
        target_language: str,
    ) -> "CorrectionRequest":
        """
        Builds the SC-04 request directly from the final
        BuiltText produced by SC-03.
        """

        data = built_text.to_dict()

        return cls(
            source_language=data[
                "source_language"
            ],
            target_language=target_language,
            recognized_text=data[
                "text"
            ],
            tokens=tuple(
                data["tokens"]
            ),
        )

    @property
    def source_bridge_language(
        self,
    ) -> str:
        return BRIDGE_LANGUAGES[
            self.source_language
        ]

    @property
    def target_bridge_language(
        self,
    ) -> str:
        return BRIDGE_LANGUAGES[
            self.target_language
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_language":
                self.source_language,
            "target_language":
                self.target_language,
            "source_bridge_language":
                self.source_bridge_language,
            "target_bridge_language":
                self.target_bridge_language,
            "recognized_text":
                self.recognized_text,
            "tokens":
                list(self.tokens),
        }


@dataclass(frozen=True)
class TranslationResult:
    """
    Valid output contract for SC-04.

    The concepts are language-independent canonical
    Sign2Sign concept IDs.
    """

    source_language: str
    target_language: str

    recognized_text: str
    corrected_text: str
    translated_text: str

    concepts: tuple[str, ...]
    unresolved: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_language":
                self.source_language,
            "target_language":
                self.target_language,
            "recognized_text":
                self.recognized_text,
            "corrected_text":
                self.corrected_text,
            "translated_text":
                self.translated_text,
            "concepts":
                list(self.concepts),
            "unresolved":
                list(self.unresolved),
        }
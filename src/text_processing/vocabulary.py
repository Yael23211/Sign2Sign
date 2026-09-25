from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VocabularyError(ValueError):
    """Raised when the canonical vocabulary is invalid."""


class Vocabulary:
    SUPPORTED_TYPES = {"atomic", "expression"}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = self._load()
        self._concepts = self._validate_and_index()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.path}"
            )

        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            raise VocabularyError(
                "Vocabulary root must be a JSON object."
            )

        return data

    def _validate_and_index(self) -> dict[str, dict[str, Any]]:
        concepts = self._data.get("concepts")

        if not isinstance(concepts, list):
            raise VocabularyError(
                "'concepts' must be a list."
            )

        index: dict[str, dict[str, Any]] = {}

        for position, concept in enumerate(concepts, start=1):
            if not isinstance(concept, dict):
                raise VocabularyError(
                    f"Concept #{position} must be an object."
                )

            concept_id = concept.get("concept_id")
            concept_type = concept.get("type")

            if not isinstance(concept_id, str) or not concept_id:
                raise VocabularyError(
                    f"Concept #{position} has an invalid concept_id."
                )

            if concept_id in index:
                raise VocabularyError(
                    f"Duplicated concept_id: {concept_id}"
                )

            if concept_type not in self.SUPPORTED_TYPES:
                raise VocabularyError(
                    f"{concept_id}: invalid type '{concept_type}'."
                )

            for language in ("spanish", "english"):
                lang_data = concept.get(language)

                if not isinstance(lang_data, dict):
                    raise VocabularyError(
                        f"{concept_id}: missing '{language}'."
                    )

                canonical = lang_data.get("canonical")
                aliases = lang_data.get("aliases")

                if not isinstance(canonical, str) or not canonical.strip():
                    raise VocabularyError(
                        f"{concept_id}: invalid {language}.canonical."
                    )

                if not isinstance(aliases, list) or not all(
                    isinstance(alias, str) for alias in aliases
                ):
                    raise VocabularyError(
                        f"{concept_id}: {language}.aliases must be a list of strings."
                    )

            source = concept.get("source")

            if not isinstance(source, dict):
                raise VocabularyError(
                    f"{concept_id}: missing source traceability."
                )

            index[concept_id] = concept

        declared = self._data.get("counts", {})
        expected_total = declared.get("total")

        if expected_total is not None and expected_total != len(index):
            raise VocabularyError(
                f"Declared total {expected_total} does not match "
                f"the {len(index)} loaded concepts."
            )

        return index

    @property
    def version(self) -> str:
        return str(self._data.get("version", ""))

    def __len__(self) -> int:
        return len(self._concepts)

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    def get_concept(self, concept_id: str) -> dict[str, Any]:
        try:
            return self._concepts[concept_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown concept_id: {concept_id}"
            ) from exc

    def get_allowed_ids(self) -> list[str]:
        return list(self._concepts.keys())

    def get_by_type(self, concept_type: str) -> list[dict[str, Any]]:
        if concept_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported concept type: {concept_type}"
            )

        return [
            concept
            for concept in self._concepts.values()
            if concept["type"] == concept_type
        ]

    def get_prompt_catalog(self) -> list[dict[str, str]]:
        """
        Compact representation intended for the future SC-04 prompt.
        It deliberately excludes video/resource information.
        """
        return [
            {
                "concept_id": concept["concept_id"],
                "type": concept["type"],
                "spanish": concept["spanish"]["canonical"],
                "english": concept["english"]["canonical"],
            }
            for concept in self._concepts.values()
        ]

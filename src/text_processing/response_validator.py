from __future__ import annotations

import json
from typing import Any

from src.text_processing.contracts import (
    CorrectionRequest,
    TranslationResult,
)

from src.text_processing.vocabulary import (
    Vocabulary,
)


class ResponseValidationError(ValueError):
    """
    Raised when the response produced by the LLM does not
    satisfy the SC-04 output contract.
    """


class ResponseValidator:
    """
    Validates the structured response returned by the LLM.

    The validator does not perform linguistic correction.
    Its responsibility is to guarantee that the response:

    - is valid JSON;
    - contains the required fields;
    - uses the expected data types;
    - contains only canonical Sign2Sign concept IDs;
    - can safely be converted into TranslationResult.
    """

    REQUIRED_FIELDS = {
        "corrected_text",
        "translated_text",
        "concepts",
        "unresolved",
    }

    def __init__(
        self,
        vocabulary: Vocabulary,
    ):
        self.vocabulary = vocabulary

    def validate(
        self,
        response: str | dict[str, Any],
        request: CorrectionRequest,
    ) -> TranslationResult:
        """
        Validates a raw LLM response and converts it into
        TranslationResult.
        """

        data = self._parse_response(
            response
        )

        self._validate_fields(
            data
        )

        corrected_text = (
            self._validate_text_field(
                data,
                "corrected_text",
            )
        )

        translated_text = (
            self._validate_text_field(
                data,
                "translated_text",
            )
        )

        concepts = (
            self._validate_concepts(
                data["concepts"]
            )
        )

        unresolved = (
            self._validate_unresolved(
                data["unresolved"]
            )
        )

        if not concepts and not unresolved:
            raise ResponseValidationError(
                "The response must contain at least "
                "one concept or one unresolved item."
            )

        return TranslationResult(
            source_language=(
                request.source_language
            ),
            target_language=(
                request.target_language
            ),
            recognized_text=(
                request.recognized_text
            ),
            corrected_text=(
                corrected_text
            ),
            translated_text=(
                translated_text
            ),
            concepts=tuple(
                concepts
            ),
            unresolved=tuple(
                unresolved
            ),
        )

    @staticmethod
    def _parse_response(
        response: str | dict[str, Any],
    ) -> dict[str, Any]:

        if isinstance(
            response,
            dict,
        ):
            return response

        if not isinstance(
            response,
            str,
        ):
            raise ResponseValidationError(
                "The LLM response must be a JSON string "
                "or a dictionary."
            )

        if not response.strip():
            raise ResponseValidationError(
                "The LLM response cannot be empty."
            )

        try:
            data = json.loads(
                response
            )

        except json.JSONDecodeError as exc:
            raise ResponseValidationError(
                "The LLM response is not valid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ResponseValidationError(
                "The JSON root must be an object."
            )

        return data

    def _validate_fields(
        self,
        data: dict[str, Any],
    ) -> None:

        actual_fields = set(
            data.keys()
        )

        missing_fields = (
            self.REQUIRED_FIELDS
            - actual_fields
        )

        if missing_fields:
            missing = ", ".join(
                sorted(
                    missing_fields
                )
            )

            raise ResponseValidationError(
                f"Missing required fields: "
                f"{missing}"
            )

    @staticmethod
    def _validate_text_field(
        data: dict[str, Any],
        field: str,
    ) -> str:

        value = data.get(
            field
        )

        if not isinstance(
            value,
            str,
        ):
            raise ResponseValidationError(
                f"'{field}' must be a string."
            )

        value = value.strip()

        if not value:
            raise ResponseValidationError(
                f"'{field}' cannot be empty."
            )

        return value

    def _validate_concepts(
        self,
        concepts: Any,
    ) -> list[str]:

        if not isinstance(
            concepts,
            list,
        ):
            raise ResponseValidationError(
                "'concepts' must be a list."
            )

        validated = []

        for position, concept_id in enumerate(
            concepts,
            start=1,
        ):

            if not isinstance(
                concept_id,
                str,
            ):
                raise ResponseValidationError(
                    "Every item in 'concepts' "
                    "must be a string."
                )

            concept_id = (
                concept_id
                .strip()
                .upper()
            )

            if not concept_id:
                raise ResponseValidationError(
                    f"Concept at position "
                    f"{position} is empty."
                )

            if not self.vocabulary.has_concept(
                concept_id
            ):
                raise ResponseValidationError(
                    f"Unknown canonical concept: "
                    f"{concept_id}"
                )

            validated.append(
                concept_id
            )

        return validated

    @staticmethod
    def _validate_unresolved(
        unresolved: Any,
    ) -> list[str]:

        if not isinstance(
            unresolved,
            list,
        ):
            raise ResponseValidationError(
                "'unresolved' must be a list."
            )

        validated = []

        for position, item in enumerate(
            unresolved,
            start=1,
        ):

            if not isinstance(
                item,
                str,
            ):
                raise ResponseValidationError(
                    "Every item in 'unresolved' "
                    "must be a string."
                )

            item = item.strip()

            if not item:
                raise ResponseValidationError(
                    f"Unresolved item at position "
                    f"{position} is empty."
                )

            validated.append(
                item
            )

        return validated
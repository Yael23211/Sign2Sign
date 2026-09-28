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
    - reports unresolved content when appropriate;
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

        # ----------------------------------------------------
        # 1. Parse response
        # ----------------------------------------------------

        data = self._parse_response(
            response
        )

        # ----------------------------------------------------
        # 2. Required fields
        # ----------------------------------------------------

        self._validate_fields(
            data
        )

        # ----------------------------------------------------
        # 3. Text fields
        #
        # Text fields must always be strings.
        #
        # However, they are allowed to be empty when the
        # complete input is unresolved and no canonical
        # concepts could be recovered.
        # ----------------------------------------------------

        corrected_text = (
            self._validate_string_field(
                data,
                "corrected_text",
            )
        )

        translated_text = (
            self._validate_string_field(
                data,
                "translated_text",
            )
        )

        # ----------------------------------------------------
        # 4. Concepts
        # ----------------------------------------------------

        concepts = (
            self._validate_concepts(
                data["concepts"]
            )
        )

        # ----------------------------------------------------
        # 5. Unresolved content
        # ----------------------------------------------------

        unresolved = (
            self._validate_unresolved(
                data["unresolved"]
            )
        )

        # ----------------------------------------------------
        # 6. Semantic consistency
        # ----------------------------------------------------

        # A response cannot resolve nothing and also report
        # nothing as unresolved.
        if not concepts and not unresolved:

            raise ResponseValidationError(
                "The response must contain at least "
                "one canonical concept or one "
                "unresolved item."
            )

        # If at least one canonical concept was recovered,
        # the model must provide corrected and translated
        # natural-language text.
        if concepts:

            if not corrected_text:

                raise ResponseValidationError(
                    "'corrected_text' cannot be empty "
                    "when canonical concepts were "
                    "resolved."
                )

            if not translated_text:

                raise ResponseValidationError(
                    "'translated_text' cannot be empty "
                    "when canonical concepts were "
                    "resolved."
                )

        # If there are no concepts but there is unresolved
        # content, empty corrected/translated text is valid.
        #
        # Example:
        #
        # {
        #   "corrected_text": "",
        #   "translated_text": "",
        #   "concepts": [],
        #   "unresolved": ["ZXQW"]
        # }

        # ----------------------------------------------------
        # 7. Build validated result
        # ----------------------------------------------------

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

    # ========================================================
    # PARSING
    # ========================================================

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
                "The LLM response must be "
                "a JSON string or a dictionary."
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

    # ========================================================
    # REQUIRED FIELDS
    # ========================================================

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

    # ========================================================
    # TEXT FIELDS
    # ========================================================

    @staticmethod
    def _validate_string_field(
        data: dict[str, Any],
        field: str,
    ) -> str:
        """
        Validates that a field is a string.

        Empty strings are allowed at this stage because
        whether they are valid depends on the semantic
        result:

        - concepts found -> text must not be empty;
        - only unresolved content -> text may be empty.
        """

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

        return value.strip()

    # ========================================================
    # CONCEPTS
    # ========================================================

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

    # ========================================================
    # UNRESOLVED
    # ========================================================

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
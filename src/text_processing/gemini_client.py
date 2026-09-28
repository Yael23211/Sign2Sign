from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai

from src.text_processing.contracts import (
    CorrectionRequest,
    TranslationResult,
)

from src.text_processing.prompt_builder import (
    PromptBuilder,
)

from src.text_processing.response_validator import (
    ResponseValidator,
)

from src.text_processing.vocabulary import (
    Vocabulary,
)


@dataclass(frozen=True)
class GeminiCallResult:
    """
    Result of a real Gemini SC-04 call.

    Keeps both the validated Sign2Sign result and
    execution metadata useful for evaluation.
    """

    result: TranslationResult
    raw_response: str
    latency_seconds: float
    model: str


class GeminiClient:
    """
    Real LLM client for SC-04.

    Flow:

    CorrectionRequest
        -> PromptBuilder
        -> Gemini
        -> structured JSON
        -> ResponseValidator
        -> TranslationResult
    """

    DEFAULT_MODEL = (
        "gemini-3.5-flash-lite"
    )

    def __init__(
        self,
        vocabulary: Vocabulary,
        model: str | None = None,
    ):
        load_dotenv()

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found "
                "in the environment."
            )

        self.vocabulary = vocabulary

        self.model = (
            model
            or self.DEFAULT_MODEL
        )

        self.client = genai.Client(
            api_key=api_key
        )

        self.prompt_builder = (
            PromptBuilder(
                vocabulary
            )
        )

        self.response_validator = (
            ResponseValidator(
                vocabulary
            )
        )

    def process(
        self,
        request: CorrectionRequest,
    ) -> GeminiCallResult:
        """
        Executes a complete real SC-04 LLM request.
        """

        prompt = (
            self.prompt_builder
            .build(
                request
            )
        )

        start = time.perf_counter()

        interaction = (
            self.client
            .interactions
            .create(
                model=self.model,
                input=prompt,

                generation_config={
                    "thinking_level":
                        "minimal",
                },

                response_format={
                    "type": "text",
                    "mime_type":
                        "application/json",

                    "schema": {
                        "type": "object",

                        "properties": {
                            "corrected_text": {
                                "type":
                                    "string",
                            },

                            "translated_text": {
                                "type":
                                    "string",
                            },

                            "concepts": {
                                "type":
                                    "array",

                                "items": {
                                    "type":
                                        "string",
                                },
                            },

                            "unresolved": {
                                "type":
                                    "array",

                                "items": {
                                    "type":
                                        "string",
                                },
                            },
                        },

                        "required": [
                            "corrected_text",
                            "translated_text",
                            "concepts",
                            "unresolved",
                        ],

                        "additionalProperties":
                            False,
                    },
                },
            )
        )

        latency = (
            time.perf_counter()
            - start
        )

        raw_response = (
            interaction.output_text
        )

        result = (
            self.response_validator
            .validate(
                raw_response,
                request,
            )
        )

        return GeminiCallResult(
            result=result,
            raw_response=raw_response,
            latency_seconds=latency,
            model=self.model,
        )
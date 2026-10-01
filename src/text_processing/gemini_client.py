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


class GeminiRequestError(RuntimeError):
    """
    Raised when communication with the Gemini service fails.

    This exception represents transport/API communication
    problems only.

    A response that is successfully received but rejected by
    ResponseValidator is NOT converted into this exception.
    """


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

    Network/API retries are delegated to the Google Gen AI SDK.
    A request timeout is explicitly defined by Sign2Sign.
    """

    DEFAULT_MODEL = (
        "gemini-3.5-flash-lite"
    )

    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        vocabulary: Vocabulary,
        model: str | None = None,
        timeout_seconds: float = (
            DEFAULT_TIMEOUT_SECONDS
        ),
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

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be "
                "greater than zero."
            )

        self.vocabulary = vocabulary

        self.model = (
            model
            or self.DEFAULT_MODEL
        )

        self.timeout_seconds = (
            float(timeout_seconds)
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

        The measured latency includes the complete SDK request,
        including any retry behavior performed internally by
        the Google Gen AI SDK.

        Communication/API errors are converted to
        GeminiRequestError.

        Response validation errors are intentionally allowed
        to propagate independently so that the system can
        distinguish:

        - communication failure;
        - invalid semantic/structured response.
        """

        prompt = (
            self.prompt_builder
            .build(
                request
            )
        )

        start = time.perf_counter()

        try:

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

                    timeout=(
                        self.timeout_seconds
                    ),
                )
            )

        except Exception as exc:

            latency = (
                time.perf_counter()
                - start
            )

            raise GeminiRequestError(
                "Gemini request failed before a "
                "valid response was received. "
                f"Elapsed time: "
                f"{latency:.3f} s. "
                f"Cause: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ) from exc

        latency = (
            time.perf_counter()
            - start
        )

        raw_response = (
            interaction.output_text
        )

        # IMPORTANT:
        #
        # Validation intentionally occurs OUTSIDE the try/except
        # used for communication.
        #
        # Therefore an invalid LLM response is distinguishable
        # from a network/API communication failure.
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
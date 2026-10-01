from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

from src.text_processing.contracts import (
    CorrectionRequest,
)

from src.text_processing.gemini_client import (
    GeminiClient,
    GeminiRequestError,
)

from src.text_processing.vocabulary import (
    Vocabulary,
)


ROOT = Path(__file__).resolve().parents[1]

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


class GeminiClientTests(
    unittest.TestCase
):

    def setUp(
        self,
    ):

        self.vocabulary = Vocabulary(
            VOCAB_PATH
        )

        self.request = CorrectionRequest(
            source_language="LSM",
            target_language="ASL",
            recognized_text="HOLA",
            tokens=(),
        )

    def _valid_interaction(
        self,
    ):

        interaction = MagicMock()

        interaction.output_text = (
            '{'
            '"corrected_text":"HOLA",'
            '"translated_text":"HELLO",'
            '"concepts":["HELLO"],'
            '"unresolved":[]'
            '}'
        )

        return interaction

    @patch(
        "src.text_processing."
        "gemini_client.genai.Client"
    )
    def test_process_returns_valid_result(
        self,
        mock_client_class,
    ):

        mock_sdk_client = MagicMock()

        (
            mock_client_class
            .return_value
        ) = mock_sdk_client

        (
            mock_sdk_client
            .interactions
            .create
            .return_value
        ) = self._valid_interaction()

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY":
                    "test-key"
            },
        ):

            client = GeminiClient(
                self.vocabulary
            )

            result = client.process(
                self.request
            )

        self.assertEqual(
            result.result.corrected_text,
            "HOLA",
        )

        self.assertEqual(
            result.result.translated_text,
            "HELLO",
        )

        self.assertEqual(
            result.result.concepts,
            ("HELLO",),
        )

        self.assertEqual(
            result.result.unresolved,
            (),
        )

        self.assertEqual(
            result.model,
            GeminiClient.DEFAULT_MODEL,
        )

        self.assertGreaterEqual(
            result.latency_seconds,
            0.0,
        )

    @patch(
        "src.text_processing."
        "gemini_client.genai.Client"
    )
    def test_timeout_is_sent_to_sdk(
        self,
        mock_client_class,
    ):

        mock_sdk_client = MagicMock()

        (
            mock_client_class
            .return_value
        ) = mock_sdk_client

        (
            mock_sdk_client
            .interactions
            .create
            .return_value
        ) = self._valid_interaction()

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY":
                    "test-key"
            },
        ):

            client = GeminiClient(
                self.vocabulary,
                timeout_seconds=7.5,
            )

            client.process(
                self.request
            )

        call_kwargs = (
            mock_sdk_client
            .interactions
            .create
            .call_args
            .kwargs
        )

        self.assertEqual(
            call_kwargs["timeout"],
            7.5,
        )

    @patch(
        "src.text_processing."
        "gemini_client.genai.Client"
    )
    def test_network_error_becomes_request_error(
        self,
        mock_client_class,
    ):

        mock_sdk_client = MagicMock()

        (
            mock_client_class
            .return_value
        ) = mock_sdk_client

        (
            mock_sdk_client
            .interactions
            .create
            .side_effect
        ) = TimeoutError(
            "Simulated timeout"
        )

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY":
                    "test-key"
            },
        ):

            client = GeminiClient(
                self.vocabulary
            )

            with self.assertRaises(
                GeminiRequestError
            ):

                client.process(
                    self.request
                )

        (
            mock_sdk_client
            .interactions
            .create
            .assert_called_once()
        )

    @patch(
        "src.text_processing."
        "gemini_client.genai.Client"
    )
    def test_invalid_response_is_not_network_error(
        self,
        mock_client_class,
    ):

        mock_sdk_client = MagicMock()

        (
            mock_client_class
            .return_value
        ) = mock_sdk_client

        interaction = MagicMock()

        interaction.output_text = (
            '{"invalid":"response"}'
        )

        (
            mock_sdk_client
            .interactions
            .create
            .return_value
        ) = interaction

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY":
                    "test-key"
            },
        ):

            client = GeminiClient(
                self.vocabulary
            )

            try:

                client.process(
                    self.request
                )

            except GeminiRequestError:

                self.fail(
                    "An invalid Gemini response "
                    "must not be reported as a "
                    "network/API communication "
                    "error."
                )

            except Exception:

                # Expected:
                # ResponseValidator must reject
                # the invalid structure.
                pass

            else:

                self.fail(
                    "Invalid response was "
                    "unexpectedly accepted."
                )

        (
            mock_sdk_client
            .interactions
            .create
            .assert_called_once()
        )

    def test_rejects_zero_timeout(
        self,
    ):

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY":
                    "test-key"
            },
        ):

            with self.assertRaises(
                ValueError
            ):

                GeminiClient(
                    self.vocabulary,
                    timeout_seconds=0,
                )

    def test_rejects_negative_timeout(
        self,
    ):

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY":
                    "test-key"
            },
        ):

            with self.assertRaises(
                ValueError
            ):

                GeminiClient(
                    self.vocabulary,
                    timeout_seconds=-1,
                )


if __name__ == "__main__":
    unittest.main()
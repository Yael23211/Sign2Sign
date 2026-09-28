from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.text import (
    BuiltText,
    PredictionCandidate,
    TextToken,
)

from src.text_processing import (
    CorrectionRequest,
    GeminiClient,
    Vocabulary,
)


VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


def build_test_input() -> BuiltText:

    return BuiltText(
        source_language="LSM",
        text="AGRA",
        tokens=(
            TextToken(
                kind="letter",
                value="A",
                top_k=(
                    PredictionCandidate(
                        "A",
                        0.97,
                    ),
                    PredictionCandidate(
                        "E",
                        0.02,
                    ),
                    PredictionCandidate(
                        "O",
                        0.01,
                    ),
                ),
            ),

            TextToken(
                kind="letter",
                value="G",
                top_k=(
                    PredictionCandidate(
                        "G",
                        0.94,
                    ),
                    PredictionCandidate(
                        "H",
                        0.04,
                    ),
                    PredictionCandidate(
                        "Q",
                        0.02,
                    ),
                ),
            ),

            TextToken(
                kind="letter",
                value="R",
                top_k=(
                    PredictionCandidate(
                        "R",
                        0.54,
                    ),
                    PredictionCandidate(
                        "U",
                        0.44,
                    ),
                    PredictionCandidate(
                        "V",
                        0.02,
                    ),
                ),
            ),

            TextToken(
                kind="letter",
                value="A",
                top_k=(
                    PredictionCandidate(
                        "A",
                        0.98,
                    ),
                    PredictionCandidate(
                        "E",
                        0.01,
                    ),
                    PredictionCandidate(
                        "O",
                        0.01,
                    ),
                ),
            ),
        ),
    )


def main():

    print()
    print("=" * 72)
    print(
        "SIGN2SIGN - PRUEBA REAL DE SC-04 CON GEMINI"
    )
    print("=" * 72)

    vocabulary = Vocabulary(
        VOCAB_PATH
    )

    built_text = (
        build_test_input()
    )

    request = (
        CorrectionRequest
        .from_built_text(
            built_text,
            target_language="ASL",
        )
    )

    client = GeminiClient(
        vocabulary
    )

    print()
    print(
        "Texto reconocido:",
        request.recognized_text,
    )

    print(
        "Flujo:",
        request.source_language,
        "->",
        request.target_language,
    )

    print()
    print(
        "Enviando solicitud a Gemini..."
    )

    call = client.process(
        request
    )

    print()
    print("=" * 72)
    print("RESPUESTA VALIDADA")
    print("=" * 72)

    print(
        json.dumps(
            call.result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        f"Modelo: "
        f"{call.model}"
    )

    print(
        f"Latencia: "
        f"{call.latency_seconds:.3f} s"
    )

    print()
    print(
        "Respuesta cruda:"
    )

    print(
        call.raw_response
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
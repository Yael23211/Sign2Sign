from __future__ import annotations

import json
import sys
import time
from pathlib import Path


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

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


# ============================================================
# UTILIDAD PARA CREAR BuiltText SIMULADO
# ============================================================

def build_built_text(
    text: str,
    source_language: str = "LSM",
) -> BuiltText:
    """
    Builds a synthetic SC-03 output from plain text.

    Letters are assigned a high-confidence Top-1 prediction.
    Spaces are represented as space tokens.

    This test focuses on SC-04 linguistic behavior,
    not SC-03 recognition accuracy.
    """

    tokens = []

    for character in text.upper():

        if character == " ":
            tokens.append(
                TextToken(
                    kind="space",
                    value=" ",
                    top_k=(),
                )
            )

            continue

        # The recognition subsystem works with A-Z.
        # Accents are therefore omitted in the simulated input.
        if not character.isalpha():
            continue

        tokens.append(
            TextToken(
                kind="letter",
                value=character,
                top_k=(
                    PredictionCandidate(
                        character,
                        0.99,
                    ),
                ),
            )
        )

    return BuiltText(
        source_language=source_language,
        text=text.upper(),
        tokens=tuple(tokens),
    )


# ============================================================
# CASOS DE PRUEBA
# ============================================================

TEST_CASES = [
    {
        "name": "Expresion predefinida",
        "recognized_text": "COMO ESTAS",
        "expected_required_concepts": {
            "HOW_ARE_YOU",
        },
        "expect_unresolved": False,
    },
    {
        "name": "Negacion composicional",
        "recognized_text": "NO QUIERO AGUA",
        "expected_required_concepts": {
            "NOT",
            "WANT",
            "WATER",
        },
        "expect_unresolved": False,
    },
    {
        "name": "Concepto fuera del vocabulario",
        "recognized_text": "QUIERO CAFE",
        "expected_required_concepts": {
            "WANT",
        },
        "expect_unresolved": True,
    },
]


# ============================================================
# VALIDACIÓN SEMÁNTICA BÁSICA
# ============================================================

def evaluate_case(
    concepts: tuple[str, ...],
    unresolved: tuple[str, ...],
    expected_required_concepts: set[str],
    expect_unresolved: bool,
) -> list[str]:

    observations = []

    concept_set = set(
        concepts
    )

    missing = (
        expected_required_concepts
        - concept_set
    )

    if missing:
        observations.append(
            "FALTAN conceptos esperados: "
            + ", ".join(
                sorted(missing)
            )
        )

    else:
        observations.append(
            "Conceptos mínimos esperados: OK"
        )

    if expect_unresolved:

        if unresolved:
            observations.append(
                "Contenido fuera de vocabulario "
                "detectado: OK"
            )

        else:
            observations.append(
                "ERROR: se esperaba contenido "
                "sin resolver."
            )

    else:

        if not unresolved:
            observations.append(
                "Sin elementos unresolved: OK"
            )

        else:
            observations.append(
                "ADVERTENCIA: apareció contenido "
                "unresolved."
            )

    return observations


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print()
    print("=" * 76)
    print(
        "SIGN2SIGN - PRUEBAS REALES DE SC-04 CON GEMINI"
    )
    print("=" * 76)

    vocabulary = Vocabulary(
        VOCAB_PATH
    )

    client = GeminiClient(
        vocabulary
    )

    print()
    print(
        f"Modelo: {client.model}"
    )

    print(
        f"Vocabulario: {len(vocabulary)} conceptos"
    )

    total_latency = 0.0
    successful_cases = 0

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print("=" * 76)

        print(
            f"CASO {index}: "
            f"{case['name']}"
        )

        print("=" * 76)

        built_text = build_built_text(
            case["recognized_text"]
        )

        request = (
            CorrectionRequest.from_built_text(
                built_text,
                target_language="ASL",
            )
        )

        print()
        print(
            "Entrada reconocida:",
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
            "Enviando solicitud..."
        )

        try:

            call = client.process(
                request
            )

        except Exception as exc:

            print()
            print(
                "ERROR DURANTE LA LLAMADA:"
            )

            print(
                type(exc).__name__,
                str(exc),
            )

            continue

        total_latency += (
            call.latency_seconds
        )

        successful_cases += 1

        result = call.result

        print()
        print(
            "Respuesta validada:"
        )

        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            f"Latencia: "
            f"{call.latency_seconds:.3f} s"
        )

        print()
        print(
            "Evaluacion del caso:"
        )

        observations = evaluate_case(
            concepts=result.concepts,
            unresolved=result.unresolved,
            expected_required_concepts=(
                case[
                    "expected_required_concepts"
                ]
            ),
            expect_unresolved=(
                case[
                    "expect_unresolved"
                ]
            ),
        )

        for observation in observations:
            print(
                " -",
                observation,
            )

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("=" * 76)
    print("RESUMEN")
    print("=" * 76)

    print(
        f"Casos ejecutados correctamente: "
        f"{successful_cases}/{len(TEST_CASES)}"
    )

    if successful_cases:

        average_latency = (
            total_latency
            / successful_cases
        )

        print(
            f"Latencia promedio: "
            f"{average_latency:.3f} s"
        )

        print(
            f"Latencia acumulada: "
            f"{total_latency:.3f} s"
        )

    print()
    print(
        "Las respuestas mostradas ya pasaron "
        "por ResponseValidator."
    )

    print(
        "Por lo tanto, ningún concept_id "
        "mostrado es ajeno al vocabulario."
    )

    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
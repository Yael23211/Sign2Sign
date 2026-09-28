from __future__ import annotations

import json
import sys
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
# UTILIDADES
# ============================================================

def candidate(
    label: str,
    probability: float,
) -> PredictionCandidate:

    return PredictionCandidate(
        label=label,
        probability=probability,
    )


def default_top_k(
    letter: str,
) -> tuple[PredictionCandidate, ...]:
    """
    Top-k artificial para letras sin ambigüedad especial.
    """

    alternatives = {
        "A": ("E", "O"),
        "B": ("D", "R"),
        "C": ("O", "D"),
        "D": ("Z", "R"),
        "E": ("I", "A"),
        "F": ("P", "T"),
        "G": ("H", "Q"),
        "H": ("G", "N"),
        "I": ("A", "J"),
        "J": ("I", "Y"),
        "K": ("P", "R"),
        "L": ("D", "I"),
        "M": ("N", "P"),
        "N": ("M", "X"),
        "O": ("C", "E"),
        "P": ("K", "M"),
        "Q": ("G", "Z"),
        "R": ("U", "W"),
        "S": ("T", "A"),
        "T": ("S", "F"),
        "U": ("R", "V"),
        "V": ("U", "W"),
        "W": ("V", "U"),
        "X": ("N", "Z"),
        "Y": ("J", "I"),
        "Z": ("D", "Q"),
    }

    alt_1, alt_2 = alternatives.get(
        letter,
        ("X", "Z"),
    )

    return (
        candidate(
            letter,
            0.97,
        ),
        candidate(
            alt_1,
            0.02,
        ),
        candidate(
            alt_2,
            0.01,
        ),
    )


def build_built_text(
    text: str,
    source_language: str,
    custom_top_k: dict[int, tuple] | None = None,
) -> BuiltText:
    """
    Construye una salida sintética de SC-03.

    custom_top_k usa posiciones de LETRA, no índices
    incluyendo espacios.

    Ejemplo:

        custom_top_k = {
            2: (
                ("C", 0.70),
                ("D", 0.20),
                ("X", 0.10),
            )
        }

    significa que para la segunda letra reconocida se
    reemplazará el Top-k normal.
    """

    custom_top_k = (
        custom_top_k
        or {}
    )

    tokens = []

    letter_position = 0

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

        letter_position += 1

        if letter_position in custom_top_k:

            candidates = tuple(
                candidate(
                    label,
                    probability,
                )
                for label, probability
                in custom_top_k[
                    letter_position
                ]
            )

        else:

            candidates = default_top_k(
                character
            )

        tokens.append(
            TextToken(
                kind="letter",
                value=character,
                top_k=candidates,
            )
        )

    return BuiltText(
        source_language=source_language,
        text=text.upper(),
        tokens=tuple(tokens),
    )


# ============================================================
# CASOS
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. CONTROL:
    # La letra correcta está en Top-2.
    # AGRA -> AGUA
    # --------------------------------------------------------

    {
        "name":
            "Correcta presente en Top-2",

        "source_language":
            "LSM",

        "target_language":
            "ASL",

        "recognized_text":
            "AGRA",

        "custom_top_k": {
            3: (
                ("R", 0.54),
                ("U", 0.44),
                ("V", 0.02),
            ),
        },

        "expected_corrected":
            "AGUA",

        "expected_concepts":
            {"WATER"},

        "expect_unresolved":
            False,
    },

    # --------------------------------------------------------
    # 2. CORRECTA FUERA DEL TOP-3:
    #
    # HCLA -> HOLA
    #
    # La segunda letra seleccionada es C.
    # Top-3 = C / D / X.
    # O NO está disponible.
    # --------------------------------------------------------

    {
        "name":
            "HOLA con letra correcta fuera del Top-3",

        "source_language":
            "LSM",

        "target_language":
            "ASL",

        "recognized_text":
            "HCLA",

        "custom_top_k": {
            2: (
                ("C", 0.74),
                ("D", 0.20),
                ("X", 0.06),
            ),
        },

        "expected_corrected":
            "HOLA",

        "expected_concepts":
            {"HELLO"},

        "expect_unresolved":
            False,
    },

    # --------------------------------------------------------
    # 3. PALABRA LARGA:
    #
    # HOSPITAK -> HOSPITAL
    #
    # K es Top-1 y L ni aparece en Top-3.
    # --------------------------------------------------------

    {
        "name":
            "HOSPITAL con letra final fuera del Top-3",

        "source_language":
            "LSM",

        "target_language":
            "ASL",

        "recognized_text":
            "HOSPITAK",

        "custom_top_k": {
            8: (
                ("K", 0.81),
                ("P", 0.12),
                ("R", 0.07),
            ),
        },

        "expected_corrected":
            "HOSPITAL",

        "expected_concepts":
            {"HOSPITAL"},

        "expect_unresolved":
            False,
    },

    # --------------------------------------------------------
    # 4. DIRECCIÓN INVERSA:
    #
    # ASL -> LSM
    #
    # HELLC -> HELLO
    #
    # O tampoco aparece en Top-3.
    # --------------------------------------------------------

    {
        "name":
            "ASL a LSM con correcta fuera del Top-3",

        "source_language":
            "ASL",

        "target_language":
            "LSM",

        "recognized_text":
            "HELLC",

        "custom_top_k": {
            5: (
                ("C", 0.69),
                ("D", 0.24),
                ("X", 0.07),
            ),
        },

        "expected_corrected":
            "HELLO",

        "expected_concepts":
            {"HELLO"},

        "expect_unresolved":
            False,
    },

    # --------------------------------------------------------
    # 5. PARCIALMENTE RESOLVIBLE:
    #
    # HCLA CAFE
    #
    # HOLA sí pertenece al dominio.
    # CAFÉ no existe.
    # --------------------------------------------------------

    {
        "name":
            "Parte válida y parte fuera de vocabulario",

        "source_language":
            "LSM",

        "target_language":
            "ASL",

        "recognized_text":
            "HCLA CAFE",

        "custom_top_k": {
            2: (
                ("C", 0.75),
                ("D", 0.18),
                ("X", 0.07),
            ),
        },

        "expected_corrected_contains":
            "HOLA",

        "expected_concepts":
            {"HELLO"},

        "expect_unresolved":
            True,
    },

    # --------------------------------------------------------
    # 6. ENTRADA SIN EVIDENCIA SUFICIENTE:
    #
    # No queremos que el LLM fuerce una palabra cualquiera
    # solamente porque está obligado a usar vocabulario.
    # --------------------------------------------------------

    {
        "name":
            "Entrada arbitraria no debe forzarse",

        "source_language":
            "LSM",

        "target_language":
            "ASL",

        "recognized_text":
            "ZXQW",

        "custom_top_k": {},

        "expected_concepts":
            set(),

        "expect_unresolved":
            True,
    },
]


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_case(
    case: dict,
    result,
) -> list[str]:

    observations = []

    result_concepts = set(
        result.concepts
    )

    expected_concepts = (
        case["expected_concepts"]
    )

    missing = (
        expected_concepts
        - result_concepts
    )

    if missing:

        observations.append(
            "ERROR - faltan conceptos: "
            + ", ".join(
                sorted(missing)
            )
        )

    else:

        observations.append(
            "Conceptos esperados: OK"
        )

    expected_corrected = (
        case.get(
            "expected_corrected"
        )
    )

    if expected_corrected:

        if (
            result.corrected_text
            == expected_corrected
        ):

            observations.append(
                "Corrección exacta: OK"
            )

        else:

            observations.append(
                "OBSERVAR - corrección obtenida: "
                + result.corrected_text
            )

    expected_contains = (
        case.get(
            "expected_corrected_contains"
        )
    )

    if expected_contains:

        if (
            expected_contains
            in result.corrected_text
        ):

            observations.append(
                "Corrección parcial esperada: OK"
            )

        else:

            observations.append(
                "ERROR - no apareció la corrección "
                "esperada."
            )

    expect_unresolved = (
        case[
            "expect_unresolved"
        ]
    )

    if expect_unresolved:

        if result.unresolved:

            observations.append(
                "Unresolved esperado: OK"
            )

        else:

            observations.append(
                "ERROR - se esperaba unresolved."
            )

    else:

        if not result.unresolved:

            observations.append(
                "Sin unresolved: OK"
            )

        else:

            observations.append(
                "OBSERVAR - apareció unresolved: "
                + str(
                    result.unresolved
                )
            )

    # Caso especial:
    # entrada arbitraria.
    #
    # No debería producir conceptos sólo por intentar
    # acomodarla al vocabulario.
    if (
        case["recognized_text"]
        == "ZXQW"
    ):

        if not result.concepts:

            observations.append(
                "No forzó conceptos: OK"
            )

        else:

            observations.append(
                "ADVERTENCIA - el modelo forzó: "
                + ", ".join(
                    result.concepts
                )
            )

    return observations


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "SIGN2SIGN - ROBUSTEZ DE SC-04 CON GEMINI"
    )
    print("=" * 78)

    vocabulary = Vocabulary(
        VOCAB_PATH
    )

    client = GeminiClient(
        vocabulary
    )

    total_latency = 0.0
    completed = 0

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print("=" * 78)

        print(
            f"CASO {index}: "
            f"{case['name']}"
        )

        print("=" * 78)

        built_text = build_built_text(
            text=case[
                "recognized_text"
            ],
            source_language=case[
                "source_language"
            ],
            custom_top_k=case[
                "custom_top_k"
            ],
        )

        request = (
            CorrectionRequest
            .from_built_text(
                built_text,
                target_language=case[
                    "target_language"
                ],
            )
        )

        print()
        print(
            "Reconocido:",
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
            "Consultando Gemini..."
        )

        try:

            call = client.process(
                request
            )

        except Exception as exc:

            print()
            print(
                "ERROR:",
                type(exc).__name__,
                str(exc),
            )

            continue

        completed += 1

        total_latency += (
            call.latency_seconds
        )

        result = call.result

        print()
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
            "Evaluación:"
        )

        for observation in evaluate_case(
            case,
            result,
        ):

            print(
                " -",
                observation,
            )

    print()
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)

    print(
        f"Casos completados: "
        f"{completed}/{len(TEST_CASES)}"
    )

    if completed:

        average = (
            total_latency
            / completed
        )

        print(
            f"Latencia promedio: "
            f"{average:.3f} s"
        )

        print(
            f"Latencia total: "
            f"{total_latency:.3f} s"
        )

    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
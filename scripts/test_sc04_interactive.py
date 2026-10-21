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


SUPPORTED_LANGUAGES = {
    "LSM",
    "ASL",
}


def normalize_text(
    text: str,
) -> str:
    """
    Normalizes manually entered recognized text.

    Consecutive spaces are collapsed and
    the text is converted to uppercase.
    """

    return " ".join(
        text
        .strip()
        .upper()
        .split()
    )


def build_manual_input(
    text: str,
    source_language: str,
) -> BuiltText:
    """
    Builds a BuiltText object from manually entered text.

    Because this script tests SC-04 in isolation,
    each letter is treated as the selected prediction
    with probability 1.0.

    This means the script is useful for testing:
    - correction behavior
    - semantic preservation
    - translation
    - canonical concepts
    - unresolved content

    It is NOT intended to evaluate real Top-k evidence
    from SC-03.
    """

    tokens = []

    for character in text:

        if character == " ":

            tokens.append(
                TextToken(
                    kind="space",
                    value=" ",
                    top_k=(),
                )
            )

            continue

        tokens.append(
            TextToken(
                kind="letter",
                value=character,
                top_k=(
                    PredictionCandidate(
                        character,
                        1.0,
                    ),
                ),
            )
        )

    return BuiltText(
        source_language=source_language,
        text=text,
        tokens=tuple(
            tokens
        ),
    )


def ask_source_language() -> str:

    while True:

        value = (
            input(
                "Lengua origen [LSM/ASL]: "
            )
            .strip()
            .upper()
        )

        if value in SUPPORTED_LANGUAGES:
            return value

        print(
            "Valor inválido. "
            "Escribe LSM o ASL."
        )


def get_target_language(
    source_language: str,
) -> str:

    if source_language == "LSM":
        return "ASL"

    return "LSM"


def print_result(
    call,
) -> None:

    print()
    print("=" * 72)
    print(
        "RESPUESTA VALIDADA"
    )
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


def main():

    print()
    print("=" * 72)
    print(
        "SIGN2SIGN - PRUEBA INTERACTIVA DE SC-04"
    )
    print("=" * 72)

    print()
    print(
        "Esta prueba omite SC-03 y permite "
        "introducir manualmente el texto reconocido."
    )

    print(
        "Escribe SALIR como texto para terminar."
    )

    vocabulary = Vocabulary(
        VOCAB_PATH
    )

    client = GeminiClient(
        vocabulary
    )

    while True:

        print()
        print("-" * 72)

        source_language = (
            ask_source_language()
        )

        target_language = (
            get_target_language(
                source_language
            )
        )

        print(
            f"Flujo seleccionado: "
            f"{source_language} -> "
            f"{target_language}"
        )

        print()

        raw_text = input(
            "Texto reconocido: "
        )

        if (
            raw_text
            .strip()
            .upper()
            == "SALIR"
        ):
            print()
            print(
                "Prueba finalizada."
            )
            break

        recognized_text = (
            normalize_text(
                raw_text
            )
        )

        if not recognized_text:

            print(
                "El texto no puede "
                "estar vacío."
            )

            continue

        built_text = (
            build_manual_input(
                recognized_text,
                source_language,
            )
        )

        request = (
            CorrectionRequest
            .from_built_text(
                built_text,
                target_language=target_language,
            )
        )

        print()
        print(
            "Texto enviado:",
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
            "Enviando solicitud "
            "a Gemini..."
        )

        try:

            call = client.process(
                request
            )

        except Exception as exc:

            print()
            print(
                "ERROR DURANTE SC-04"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

        print_result(
            call
        )


if __name__ == "__main__":
    main()
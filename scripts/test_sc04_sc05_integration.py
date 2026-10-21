from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.text_processing.contracts import TranslationResult
from src.visual_mapping import (
    VisualMappingError,
    VisualResourceMapper,
)


def main() -> int:

    print()
    print(
        "============================================"
    )
    print(
        " Sign2Sign - Integración SC-04 -> SC-05"
    )
    print(
        "============================================"
    )
    print()

    # Salida estructurada equivalente a la generada
    # y validada por SC-04.
    translation_result = TranslationResult(
        source_language="LSM",
        target_language="ASL",
        recognized_text="HOLA COMT ESTAA",
        corrected_text="HOLA CÓMO ESTÁS",
        translated_text="HELLO HOW ARE YOU",
        concepts=[
            "HELLO",
            "HOW_ARE_YOU",
        ],
        unresolved=[],
    )

    print("SC-04 TranslationResult")
    print("-----------------------")
    print(
        f"Source language : "
        f"{translation_result.source_language}"
    )
    print(
        f"Target language : "
        f"{translation_result.target_language}"
    )
    print(
        f"Recognized text : "
        f"{translation_result.recognized_text}"
    )
    print(
        f"Corrected text  : "
        f"{translation_result.corrected_text}"
    )
    print(
        f"Translated text : "
        f"{translation_result.translated_text}"
    )
    print(
        f"Concepts        : "
        f"{translation_result.concepts}"
    )
    print(
        f"Unresolved      : "
        f"{translation_result.unresolved}"
    )

    print()
    print(
        "Enviando TranslationResult a SC-05..."
    )
    print()

    mapper = VisualResourceMapper()

    try:
        mapping_result = (
            mapper.resolve_translation_result(
                translation_result,
                delivery="local",
            )
        )

    except VisualMappingError as exc:
        print(
            "ERROR EN SC-05"
        )

        print(
            str(exc)
        )

        return 1

    print(
        "SC-05 ResourceMappingResult"
    )
    print(
        "---------------------------"
    )

    print(
        json.dumps(
            mapping_result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    expected = (
        "resources/asl/HELLO.mp4",
        "resources/asl/HOW_ARE_YOU.mp4",
    )

    if mapping_result.locations != expected:

        print(
            "ERROR: las rutas obtenidas "
            "no coinciden con las esperadas."
        )

        print(
            "Esperadas:"
        )

        for path in expected:
            print(
                f"  {path}"
            )

        print(
            "Obtenidas:"
        )

        for path in mapping_result.locations:
            print(
                f"  {path}"
            )

        return 1

    print(
        "OK: integración SC-04 -> SC-05 correcta."
    )

    print()
    print(
        "Secuencia visual resuelta:"
    )

    for index, resource in enumerate(
        mapping_result.resources,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{resource.concept_id} "
            f"-> "
            f"{resource.location}"
        )

    print()

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
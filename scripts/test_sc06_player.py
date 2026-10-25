from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.visual_mapping import (
    VisualResourceMapper,
)

from src.visual_output import (
    VisualOutputError,
    VisualSequencePlayer,
)


def main() -> int:

    print()
    print("=" * 60)
    print(
        "Sign2Sign - Smoke test SC-05 -> SC-06"
    )
    print("=" * 60)

    mapper = (
        VisualResourceMapper()
    )

    player = (
        VisualSequencePlayer()
    )

    concepts = [
        "HELLO",
        "HOW_ARE_YOU",
    ]

    print()
    print(
        "Resolviendo recursos..."
    )

    mapping_result = (
        mapper.resolve(
            concepts=concepts,
            target_language="ASL",
            delivery="local",
        )
    )

    print(
        "Secuencia:"
    )

    for resource in (
        mapping_result.resources
    ):

        print(
            f"  {resource.concept_id}"
            f" -> "
            f"{resource.location}"
        )

    print()
    print(
        "Iniciando reproducción..."
    )
    print(
        "SPACE = pausa | Q/ESC = detener"
    )

    try:

        result = player.play(
            mapping_result,
            translated_text=(
                "HELLO, HOW ARE YOU"
            ),
        )

    except VisualOutputError as exc:

        print()
        print(
            "[ERROR SC-06]"
        )
        print(
            str(exc)
        )

        return 1

    print()
    print(
        "Resultado:"
    )

    print(
        f"  Completado: "
        f"{result.completed}"
    )

    print(
        f"  Interrumpido: "
        f"{result.interrupted}"
    )

    print(
        f"  Videos reproducidos: "
        f"{result.played_count}/"
        f"{result.total_count}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
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


from src.visual_mapping import (
    VisualMappingError,
    VisualResourceMapper,
)


def main() -> int:

    mapper = VisualResourceMapper()

    target_language = "LSM"

    concepts = [
        "FIRST_PERSON_SINGULAR",
        "WANT",
        "WATER",
    ]

    print()
    print(
        "============================================"
    )
    print(
        " Sign2Sign - Smoke test SC-05"
    )
    print(
        "============================================"
    )
    print()

    print(
        f"Target language: {target_language}"
    )

    print(
        f"Concepts: {concepts}"
    )

    print()

    try:
        result = mapper.resolve(
            concepts=concepts,
            target_language=target_language,
            delivery="local",
        )

    except VisualMappingError as exc:
        print(
            "ERROR SC-05"
        )

        print(
            str(exc)
        )

        return 1

    print(
        "Resolved resources:"
    )

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
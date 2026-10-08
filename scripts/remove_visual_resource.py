from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOURCE_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "visual_resources.json"
)

RESOURCE_SOURCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "resource_sources.csv"
)

SUPPORTED_LANGUAGES = {
    "ASL",
    "LSM",
}

SUPPORTED_DELIVERIES = {
    "local",
    "online",
}

CSV_FIELDS = [
    "concept_id",
    "language",
    "delivery",
    "source_name",
    "source_url",
    "notes",
]


class RemovalError(Exception):
    pass


def load_resource_catalog() -> dict:

    if not RESOURCE_CATALOG_PATH.is_file():
        raise RemovalError(
            "Visual resource catalog was not found:\n"
            f"{RESOURCE_CATALOG_PATH}"
        )

    try:
        with RESOURCE_CATALOG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise RemovalError(
            "visual_resources.json "
            "is not valid JSON."
        ) from exc

    resources = data.get(
        "resources"
    )

    if not isinstance(
        resources,
        list,
    ):
        raise RemovalError(
            "visual_resources.json must contain "
            "a 'resources' array."
        )

    return data


def load_source_rows() -> list[dict[str, str]]:

    if not RESOURCE_SOURCES_PATH.exists():
        return []

    if (
        RESOURCE_SOURCES_PATH.stat().st_size
        == 0
    ):
        return []

    with RESOURCE_SOURCES_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            return []

        if reader.fieldnames != CSV_FIELDS:
            raise RemovalError(
                "resource_sources.csv has "
                "an unexpected header."
            )

        return [
            dict(row)
            for row in reader
        ]


def find_resource_index(
    resources: list,
    concept_id: str,
    language: str,
    delivery: str,
) -> int | None:

    for index, resource in enumerate(
        resources
    ):

        if not isinstance(
            resource,
            dict,
        ):
            continue

        if (
            str(
                resource.get(
                    "concept_id",
                    "",
                )
            ).strip().upper()
            == concept_id
            and
            str(
                resource.get(
                    "language",
                    "",
                )
            ).strip().upper()
            == language
            and
            str(
                resource.get(
                    "delivery",
                    "",
                )
            ).strip().lower()
            == delivery
        ):
            return index

    return None


def find_source_index(
    rows: list[dict[str, str]],
    concept_id: str,
    language: str,
    delivery: str,
) -> int | None:

    for index, row in enumerate(
        rows
    ):

        if (
            row.get(
                "concept_id",
                "",
            ).strip().upper()
            == concept_id
            and
            row.get(
                "language",
                "",
            ).strip().upper()
            == language
            and
            row.get(
                "delivery",
                "",
            ).strip().lower()
            == delivery
        ):
            return index

    return None


def save_resource_catalog(
    data: dict,
) -> None:

    temporary_path = (
        RESOURCE_CATALOG_PATH
        .with_suffix(
            ".json.tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    temporary_path.replace(
        RESOURCE_CATALOG_PATH
    )


def save_source_rows(
    rows: list[dict[str, str]],
) -> None:

    temporary_path = (
        RESOURCE_SOURCES_PATH
        .with_suffix(
            ".csv.tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    temporary_path.replace(
        RESOURCE_SOURCES_PATH
    )


def calculate_progress(
    resources: list,
) -> dict[str, int]:

    groups = {
        ("ASL", "local"): set(),
        ("ASL", "online"): set(),
        ("LSM", "local"): set(),
        ("LSM", "online"): set(),
    }

    for resource in resources:

        if not isinstance(
            resource,
            dict,
        ):
            continue

        concept_id = str(
            resource.get(
                "concept_id",
                "",
            )
        ).strip().upper()

        language = str(
            resource.get(
                "language",
                "",
            )
        ).strip().upper()

        delivery = str(
            resource.get(
                "delivery",
                "",
            )
        ).strip().lower()

        key = (
            language,
            delivery,
        )

        if (
            concept_id
            and key in groups
        ):
            groups[key].add(
                concept_id
            )

    return {
        "asl_local":
            len(
                groups[
                    ("ASL", "local")
                ]
            ),
        "lsm_local":
            len(
                groups[
                    ("LSM", "local")
                ]
            ),
        "asl_online":
            len(
                groups[
                    ("ASL", "online")
                ]
            ),
        "lsm_online":
            len(
                groups[
                    ("LSM", "online")
                ]
            ),
        "local_pairs":
            len(
                groups[
                    ("ASL", "local")
                ]
                &
                groups[
                    ("LSM", "local")
                ]
            ),
        "online_pairs":
            len(
                groups[
                    ("ASL", "online")
                ]
                &
                groups[
                    ("LSM", "online")
                ]
            ),
    }


def ask_nonempty(
    message: str,
) -> str:

    while True:

        value = input(
            message
        ).strip()

        if value:
            return value

        print(
            "Este campo no puede "
            "quedar vacío."
        )


def main() -> int:

    print()
    print(
        "========================================"
    )
    print(
        " Sign2Sign - Eliminar registro visual"
    )
    print(
        "========================================"
    )
    print()

    try:

        catalog = (
            load_resource_catalog()
        )

        source_rows = (
            load_source_rows()
        )

        concept_id = (
            ask_nonempty(
                "Concept ID: "
            )
            .upper()
        )

        language = (
            ask_nonempty(
                "Language [ASL/LSM]: "
            )
            .upper()
        )

        if (
            language
            not in SUPPORTED_LANGUAGES
        ):
            raise RemovalError(
                "Language debe ser "
                "ASL o LSM."
            )

        delivery = (
            ask_nonempty(
                "Delivery [local/online]: "
            )
            .lower()
        )

        if (
            delivery
            not in SUPPORTED_DELIVERIES
        ):
            raise RemovalError(
                "Delivery debe ser "
                "local u online."
            )

        resources = catalog[
            "resources"
        ]

        resource_index = (
            find_resource_index(
                resources,
                concept_id,
                language,
                delivery,
            )
        )

        source_index = (
            find_source_index(
                source_rows,
                concept_id,
                language,
                delivery,
            )
        )

        if (
            resource_index is None
            and source_index is None
        ):
            raise RemovalError(
                f"{concept_id} / "
                f"{language} / "
                f"{delivery} "
                "no está registrado."
            )

        print()
        print(
            "Registro localizado:"
        )
        print()

        resource = None
        source = None

        if resource_index is not None:

            resource = resources[
                resource_index
            ]

            print(
                "visual_resources.json:"
            )

            print(
                f"  Concept ID : "
                f"{resource.get('concept_id')}"
            )

            print(
                f"  Language   : "
                f"{resource.get('language')}"
            )

            print(
                f"  Delivery   : "
                f"{resource.get('delivery')}"
            )

            print(
                f"  Location   : "
                f"{resource.get('location')}"
            )

        else:

            print(
                "ADVERTENCIA: no existe "
                "entrada en visual_resources.json."
            )

        print()

        if source_index is not None:

            source = source_rows[
                source_index
            ]

            print(
                "resource_sources.csv:"
            )

            print(
                f"  Source     : "
                f"{source.get('source_name', '')}"
            )

            print(
                f"  URL        : "
                f"{source.get('source_url', '')}"
            )

            print(
                f"  Notes      : "
                f"{source.get('notes', '')}"
            )

        else:

            print(
                "ADVERTENCIA: no existe "
                "entrada en resource_sources.csv."
            )

        print()

        if (
            resource is not None
            and delivery == "local"
        ):
            print(
                "El archivo local NO será "
                "eliminado físicamente."
            )

        if delivery == "online":
            print(
                "Sólo se eliminará el registro "
                "del recurso online."
            )

        print()

        confirmation = (
            input(
                "Confirmar eliminación "
                "del registro [S/N]: "
            )
            .strip()
            .upper()
        )

        if confirmation not in {
            "S",
            "SI",
            "SÍ",
            "Y",
            "YES",
        }:
            print()
            print(
                "Eliminación cancelada."
            )

            return 0

        removed_resource = None
        removed_source = None

        if resource_index is not None:

            removed_resource = (
                resources.pop(
                    resource_index
                )
            )

        if source_index is not None:

            removed_source = (
                source_rows.pop(
                    source_index
                )
            )

        save_resource_catalog(
            catalog
        )

        try:

            save_source_rows(
                source_rows
            )

        except Exception:

            if (
                removed_resource
                is not None
            ):
                resources.insert(
                    resource_index,
                    removed_resource,
                )

            if (
                removed_source
                is not None
            ):
                source_rows.insert(
                    source_index,
                    removed_source,
                )

            save_resource_catalog(
                catalog
            )

            raise

        progress = (
            calculate_progress(
                resources
            )
        )

        print()
        print(
            "========================================"
        )
        print(
            " Registro eliminado correctamente"
        )
        print(
            "========================================"
        )
        print()

        print(
            f"{concept_id} / "
            f"{language} / "
            f"{delivery}"
        )

        print()
        print(
            "Recursos locales:"
        )
        print(
            f"  ASL : "
            f"{progress['asl_local']}/71"
        )
        print(
            f"  LSM : "
            f"{progress['lsm_local']}/71"
        )
        print(
            f"  Pares : "
            f"{progress['local_pairs']}/71"
        )

        print()
        print(
            "Recursos online:"
        )
        print(
            f"  ASL : "
            f"{progress['asl_online']}/71"
        )
        print(
            f"  LSM : "
            f"{progress['lsm_online']}/71"
        )
        print(
            f"  Pares : "
            f"{progress['online_pairs']}/71"
        )

        if (
            resource is not None
            and delivery == "local"
        ):

            location = (
                resource.get(
                    "location",
                    ""
                )
            )

            if location:

                physical_path = (
                    PROJECT_ROOT
                    / location
                )

                print()
                print(
                    "Archivo físico conservado:"
                )
                print(
                    physical_path
                )

        print()

        return 0

    except RemovalError as exc:

        print()
        print(
            "ERROR DE ELIMINACIÓN"
        )
        print(
            str(exc)
        )
        print()

        return 1

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Eliminación cancelada."
        )

        return 130

    except Exception as exc:

        print()
        print(
            "ERROR INESPERADO"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "visual_resources.json"
)

SOURCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "resource_sources.csv"
)

OLD_CSV_FIELDS = [
    "concept_id",
    "language",
    "source_name",
    "source_url",
    "notes",
]

NEW_CSV_FIELDS = [
    "concept_id",
    "language",
    "delivery",
    "source_name",
    "source_url",
    "notes",
]


class MigrationError(Exception):
    pass


def load_catalog() -> dict:
    if not CATALOG_PATH.is_file():
        raise MigrationError(
            f"No se encontró:\n{CATALOG_PATH}"
        )

    try:
        with CATALOG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise MigrationError(
            "visual_resources.json no es JSON válido."
        ) from exc

    if not isinstance(data, dict):
        raise MigrationError(
            "La raíz de visual_resources.json "
            "debe ser un objeto."
        )

    resources = data.get("resources")

    if not isinstance(resources, list):
        raise MigrationError(
            "visual_resources.json debe contener "
            "un arreglo 'resources'."
        )

    return data


def migrate_catalog(
    data: dict,
) -> tuple[dict, int]:
    migrated_count = 0
    new_resources = []

    for index, resource in enumerate(
        data["resources"]
    ):
        if not isinstance(resource, dict):
            raise MigrationError(
                f"El recurso #{index + 1} "
                "no es un objeto válido."
            )

        # Ya usa el esquema nuevo.
        if (
            "delivery" in resource
            and "location" in resource
        ):
            new_resources.append(resource)
            continue

        # Esquema anterior.
        required_old_fields = {
            "concept_id",
            "language",
            "path",
            "type",
            "available",
        }

        if not required_old_fields.issubset(
            resource.keys()
        ):
            raise MigrationError(
                f"El recurso #{index + 1} no coincide "
                "con el esquema anterior ni con el nuevo."
            )

        new_resource = {
            "concept_id":
                resource["concept_id"],
            "language":
                resource["language"],
            "delivery":
                "local",
            "location":
                resource["path"],
            "type":
                resource["type"],
            "available":
                resource["available"],
        }

        new_resources.append(
            new_resource
        )

        migrated_count += 1

    return (
        {
            "resources": new_resources
        },
        migrated_count,
    )


def load_and_migrate_sources(
) -> tuple[list[dict[str, str]], int]:

    if not SOURCES_PATH.exists():
        return [], 0

    if SOURCES_PATH.stat().st_size == 0:
        return [], 0

    with SOURCES_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return [], 0

        fieldnames = reader.fieldnames

        if fieldnames == NEW_CSV_FIELDS:
            return (
                [dict(row) for row in reader],
                0,
            )

        if fieldnames != OLD_CSV_FIELDS:
            raise MigrationError(
                "resource_sources.csv tiene "
                "un encabezado inesperado.\n\n"
                f"Encontrado:\n{fieldnames}\n\n"
                f"Esperado antiguo:\n"
                f"{OLD_CSV_FIELDS}\n\n"
                f"O esperado nuevo:\n"
                f"{NEW_CSV_FIELDS}"
            )

        migrated_rows = []

        for row in reader:
            migrated_rows.append(
                {
                    "concept_id":
                        row.get(
                            "concept_id",
                            "",
                        ),
                    "language":
                        row.get(
                            "language",
                            "",
                        ),
                    "delivery":
                        "local",
                    "source_name":
                        row.get(
                            "source_name",
                            "",
                        ),
                    "source_url":
                        row.get(
                            "source_url",
                            "",
                        ),
                    "notes":
                        row.get(
                            "notes",
                            "",
                        ),
                }
            )

        return (
            migrated_rows,
            len(migrated_rows),
        )


def create_backup(
    original: Path,
) -> None:
    if not original.exists():
        return

    backup = original.with_name(
        original.name + ".v1.bak"
    )

    if backup.exists():
        return

    shutil.copy2(
        original,
        backup,
    )


def save_catalog(
    data: dict,
) -> None:
    temp_path = CATALOG_PATH.with_suffix(
        ".json.tmp"
    )

    with temp_path.open(
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

    temp_path.replace(
        CATALOG_PATH
    )


def save_sources(
    rows: list[dict[str, str]],
) -> None:
    SOURCES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = SOURCES_PATH.with_suffix(
        ".csv.tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=NEW_CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(
        SOURCES_PATH
    )


def main() -> int:
    print()
    print(
        "========================================"
    )
    print(
        " Sign2Sign - Migración SC-07 V2"
    )
    print(
        " Local + Online"
    )
    print(
        "========================================"
    )
    print()

    try:
        catalog = load_catalog()

        (
            new_catalog,
            catalog_count,
        ) = migrate_catalog(
            catalog
        )

        (
            source_rows,
            source_count,
        ) = load_and_migrate_sources()

        print(
            "Cambios detectados:"
        )
        print(
            f"  Recursos a migrar : "
            f"{catalog_count}"
        )
        print(
            f"  Fuentes a migrar  : "
            f"{source_count}"
        )
        print()

        if (
            catalog_count == 0
            and source_count == 0
        ):
            print(
                "Los archivos ya utilizan "
                "el esquema V2."
            )
            print()

            return 0

        print(
            "Se crearán respaldos antes "
            "de modificar los archivos."
        )
        print()

        confirmation = (
            input(
                "Continuar con la migración [S/N]: "
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
                "Migración cancelada."
            )
            return 0

        create_backup(
            CATALOG_PATH
        )

        create_backup(
            SOURCES_PATH
        )

        save_catalog(
            new_catalog
        )

        save_sources(
            source_rows
        )

        print()
        print(
            "========================================"
        )
        print(
            " Migración completada"
        )
        print(
            "========================================"
        )
        print()

        print(
            f"Recursos migrados : "
            f"{catalog_count}"
        )
        print(
            f"Fuentes migradas  : "
            f"{source_count}"
        )

        print()
        print(
            "Respaldos:"
        )
        print(
            "  visual_resources.json.v1.bak"
        )
        print(
            "  resource_sources.csv.v1.bak"
        )

        print()
        print(
            "Los archivos de video NO "
            "fueron modificados."
        )
        print()

        return 0

    except MigrationError as exc:
        print()
        print(
            "ERROR DE MIGRACIÓN"
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
            "Migración cancelada."
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
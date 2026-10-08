from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VOCABULARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)

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


class RegistrationError(Exception):
    pass


def load_allowed_concepts() -> set[str]:

    if not VOCABULARY_PATH.is_file():
        raise RegistrationError(
            "Canonical vocabulary was not found:\n"
            f"{VOCABULARY_PATH}"
        )

    try:
        with VOCABULARY_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise RegistrationError(
            "Canonical vocabulary is not valid JSON."
        ) from exc

    concepts = data.get(
        "concepts"
    )

    if not isinstance(
        concepts,
        list,
    ):
        raise RegistrationError(
            "Canonical vocabulary must contain "
            "a 'concepts' array."
        )

    allowed = set()

    for item in concepts:

        if not isinstance(
            item,
            dict,
        ):
            continue

        concept_id = item.get(
            "concept_id"
        )

        if (
            isinstance(
                concept_id,
                str,
            )
            and concept_id.strip()
        ):
            allowed.add(
                concept_id
                .strip()
                .upper()
            )

    if not allowed:
        raise RegistrationError(
            "No concept_id values were found."
        )

    return allowed


def load_resource_catalog() -> dict:

    if not RESOURCE_CATALOG_PATH.is_file():
        raise RegistrationError(
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
        raise RegistrationError(
            "visual_resources.json "
            "is not valid JSON."
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise RegistrationError(
            "visual_resources.json root "
            "must be an object."
        )

    resources = data.get(
        "resources"
    )

    if not isinstance(
        resources,
        list,
    ):
        raise RegistrationError(
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
            raise RegistrationError(
                "resource_sources.csv has "
                "an unexpected header.\n"
                "Expected:\n"
                + ",".join(
                    CSV_FIELDS
                )
            )

        return [
            dict(row)
            for row in reader
        ]


def normalize_concept_id(
    value: str,
) -> str:

    return (
        value
        .strip()
        .upper()
    )


def normalize_language(
    value: str,
) -> str:

    return (
        value
        .strip()
        .upper()
    )


def normalize_delivery(
    value: str,
) -> str:

    return (
        value
        .strip()
        .lower()
    )


def build_local_location(
    concept_id: str,
    language: str,
) -> str:

    return (
        f"resources/"
        f"{language.lower()}/"
        f"{concept_id}.mp4"
    )


def validate_online_url(
    value: str,
) -> str:

    value = value.strip()

    parsed = urlparse(
        value
    )

    if (
        parsed.scheme
        not in {
            "http",
            "https",
        }
    ):
        raise RegistrationError(
            "Online location must use "
            "http or https."
        )

    if not parsed.netloc:
        raise RegistrationError(
            "Online location must contain "
            "a valid host."
        )

    return value


def resource_is_registered(
    resources: list,
    concept_id: str,
    language: str,
    delivery: str,
) -> bool:

    for resource in resources:

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
            return True

    return False


def source_is_registered(
    rows: list[dict[str, str]],
    concept_id: str,
    language: str,
    delivery: str,
) -> bool:

    for row in rows:

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
            return True

    return False


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

    RESOURCE_SOURCES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    local_pairs = (
        groups[
            ("ASL", "local")
        ]
        &
        groups[
            ("LSM", "local")
        ]
    )

    online_pairs = (
        groups[
            ("ASL", "online")
        ]
        &
        groups[
            ("LSM", "online")
        ]
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
                local_pairs
            ),
        "online_pairs":
            len(
                online_pairs
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
        " Sign2Sign - Registro de recurso visual"
    )
    print(
        "========================================"
    )
    print()

    try:

        allowed_concepts = (
            load_allowed_concepts()
        )

        catalog = (
            load_resource_catalog()
        )

        source_rows = (
            load_source_rows()
        )

        concept_id = (
            normalize_concept_id(
                ask_nonempty(
                    "Concept ID: "
                )
            )
        )

        if (
            concept_id
            not in allowed_concepts
        ):
            raise RegistrationError(
                f"'{concept_id}' no existe "
                "en el vocabulario canónico."
            )

        language = (
            normalize_language(
                ask_nonempty(
                    "Language [ASL/LSM]: "
                )
            )
        )

        if (
            language
            not in SUPPORTED_LANGUAGES
        ):
            raise RegistrationError(
                "Language debe ser ASL o LSM."
            )

        delivery = (
            normalize_delivery(
                ask_nonempty(
                    "Delivery [local/online]: "
                )
            )
        )

        if (
            delivery
            not in SUPPORTED_DELIVERIES
        ):
            raise RegistrationError(
                "Delivery debe ser "
                "local u online."
            )

        resources = catalog[
            "resources"
        ]

        if resource_is_registered(
            resources,
            concept_id,
            language,
            delivery,
        ):
            raise RegistrationError(
                f"{concept_id} / "
                f"{language} / "
                f"{delivery} "
                "ya está registrado."
            )

        if source_is_registered(
            source_rows,
            concept_id,
            language,
            delivery,
        ):
            raise RegistrationError(
                f"{concept_id} / "
                f"{language} / "
                f"{delivery} "
                "ya está registrado "
                "en resource_sources.csv."
            )

        if delivery == "local":

            location = (
                build_local_location(
                    concept_id,
                    language,
                )
            )

            physical_path = (
                PROJECT_ROOT
                / location
            )

            if not physical_path.is_file():
                raise RegistrationError(
                    "No se encontró el video esperado:\n"
                    f"{physical_path}\n\n"
                    "Guarda primero el archivo "
                    "con el nombre correcto."
                )

            print()
            print(
                "Video local encontrado:"
            )
            print(
                location
            )

        else:

            print()

            location = (
                validate_online_url(
                    ask_nonempty(
                        "Online location URL: "
                    )
                )
            )

            print()
            print(
                "Recurso online:"
            )
            print(
                location
            )

        print()

        source_name = ask_nonempty(
            "Source name: "
        )

        source_url = (
            validate_online_url(
                ask_nonempty(
                    "Source URL: "
                )
            )
        )

        notes = input(
            "Notes [opcional]: "
        ).strip()

        new_resource = {
            "concept_id":
                concept_id,
            "language":
                language,
            "delivery":
                delivery,
            "location":
                location,
            "type":
                "video",
            "available":
                True,
        }

        new_source = {
            "concept_id":
                concept_id,
            "language":
                language,
            "delivery":
                delivery,
            "source_name":
                source_name,
            "source_url":
                source_url,
            "notes":
                notes,
        }

        print()
        print(
            "Se registrará:"
        )
        print(
            f"  Concept ID : {concept_id}"
        )
        print(
            f"  Language   : {language}"
        )
        print(
            f"  Delivery   : {delivery}"
        )
        print(
            f"  Location   : {location}"
        )
        print(
            f"  Source     : {source_name}"
        )
        print(
            f"  Source URL : {source_url}"
        )

        if notes:
            print(
                f"  Notes      : {notes}"
            )

        print()

        confirmation = (
            input(
                "Confirmar registro [S/N]: "
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
                "Registro cancelado."
            )

            return 0

        resources.append(
            new_resource
        )

        source_rows.append(
            new_source
        )

        save_resource_catalog(
            catalog
        )

        try:
            save_source_rows(
                source_rows
            )

        except Exception:

            resources.pop()

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
            " Recurso registrado correctamente"
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
            f"  Pares locales : "
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
            f"  Pares online : "
            f"{progress['online_pairs']}/71"
        )

        print()

        return 0

    except RegistrationError as exc:

        print()
        print(
            "ERROR DE REGISTRO"
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
            "Registro cancelado."
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
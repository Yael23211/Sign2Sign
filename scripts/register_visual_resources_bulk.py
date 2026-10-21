from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

from register_visual_resource import (
    PROJECT_ROOT,
    RESOURCE_CATALOG_PATH,
    RESOURCE_SOURCES_PATH,
    RegistrationError,
    build_local_location,
    calculate_progress,
    load_allowed_concepts,
    load_resource_catalog,
    load_source_rows,
    resource_is_registered,
    save_resource_catalog,
    save_source_rows,
    source_is_registered,
)


ASL_OWN_RECORDINGS = {
    "HE",
    "SHE",
    "YOU_ALL",
    "SUBWAY",
    "HOW_ARE_YOU",
    "NICE_TO_MEET_YOU",
    "WHAT_IS_YOUR_NAME",
    "HOW_DO_YOU_FEEL",
}


def get_folder_concepts(language: str) -> set[str]:
    folder = PROJECT_ROOT / "resources" / language.lower()

    if not folder.is_dir():
        raise RegistrationError(
            f"No existe la carpeta de recursos:\n{folder}"
        )

    return {
        path.stem.strip().upper()
        for path in folder.glob("*.mp4")
        if path.is_file()
    }


def validate_physical_repository(allowed_concepts: set[str]) -> None:
    print("Validando archivos físicos...")

    has_errors = False

    for language in ("ASL", "LSM"):
        actual = get_folder_concepts(language)

        missing = sorted(allowed_concepts - actual)
        extra = sorted(actual - allowed_concepts)

        print()
        print(f"{language}")
        print(f"  Esperados   : {len(allowed_concepts)}")
        print(f"  Encontrados : {len(actual)}")

        if missing:
            has_errors = True
            print("  Faltantes:")
            for concept_id in missing:
                print(f"    - {concept_id}")
        else:
            print("  Faltantes   : ninguno")

        if extra:
            has_errors = True
            print("  Extras / nombres no canónicos:")
            for concept_id in extra:
                print(f"    - {concept_id}")
        else:
            print("  Extras      : ninguno")

    if has_errors:
        raise RegistrationError(
            "El repositorio físico no coincide con el vocabulario. "
            "Corrige los nombres/archivos antes de registrar."
        )


def build_source_metadata(
    concept_id: str,
    language: str,
) -> tuple[str, str, str]:
    source_url = ""

    if language == "ASL":
        source_name = "Grabación propia"

        if concept_id in {"HE", "SHE"}:
            notes = (
                "Grabación propia para el prototipo Sign2Sign. "
                "HE y SHE comparten la misma realización visual deíctica "
                "de tercera persona. Recurso no validado por especialista "
                "certificado en ASL."
            )
        else:
            notes = (
                "Grabación propia para el prototipo Sign2Sign, realizada "
                "a partir de material de referencia externo. Recurso no "
                "validado por especialista certificado en ASL."
            )

    else:
        source_name = "Grabación propia"
        notes = (
            "Grabación propia para el prototipo Sign2Sign, basada en "
            "material de referencia de LSM. Recurso no validado por "
            "especialista certificado en LSM."
        )

    return source_name, source_url, notes


def create_backup() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_dir = (
        RESOURCE_CATALOG_PATH.parent
        / "backups"
        / f"before_bulk_{timestamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    if RESOURCE_CATALOG_PATH.is_file():
        shutil.copy2(
            RESOURCE_CATALOG_PATH,
            backup_dir / RESOURCE_CATALOG_PATH.name,
        )

    if RESOURCE_SOURCES_PATH.is_file():
        shutil.copy2(
            RESOURCE_SOURCES_PATH,
            backup_dir / RESOURCE_SOURCES_PATH.name,
        )

    return backup_dir


def main() -> int:
    print()
    print("====================================================")
    print(" Sign2Sign - Registro masivo de recursos visuales")
    print("====================================================")
    print()

    try:
        allowed_concepts = load_allowed_concepts()

        if len(allowed_concepts) != 71:
            raise RegistrationError(
                "Se esperaban 71 concept_id en el vocabulario canónico, "
                f"pero se encontraron {len(allowed_concepts)}."
            )

        unknown_asl = ASL_OWN_RECORDINGS - allowed_concepts
        if unknown_asl:
            raise RegistrationError(
                "Hay conceptos ASL propios que no existen en el vocabulario:\n"
                + "\n".join(sorted(unknown_asl))
            )

        validate_physical_repository(allowed_concepts)

        catalog = load_resource_catalog()
        source_rows = load_source_rows()
        resources = catalog["resources"]

        targets: list[tuple[str, str]] = []

        for concept_id in sorted(ASL_OWN_RECORDINGS):
            targets.append((concept_id, "ASL"))

        for concept_id in sorted(allowed_concepts):
            targets.append((concept_id, "LSM"))

        pending: list[tuple[str, str]] = []
        already_complete: list[tuple[str, str]] = []

        for concept_id, language in targets:
            resource_exists = resource_is_registered(
                resources,
                concept_id,
                language,
                "local",
            )

            source_exists = source_is_registered(
                source_rows,
                concept_id,
                language,
                "local",
            )

            if resource_exists and source_exists:
                already_complete.append((concept_id, language))
                continue

            if resource_exists != source_exists:
                raise RegistrationError(
                    "Se detectó un registro inconsistente:\n"
                    f"{concept_id} / {language} / local\n"
                    f"visual_resources.json: {resource_exists}\n"
                    f"resource_sources.csv: {source_exists}\n\n"
                    "No se modificó ningún archivo. Revisa este caso antes "
                    "de continuar."
                )

            pending.append((concept_id, language))

        print()
        print("Resumen previo")
        print("-------------")
        print(f"ASL propios esperados : {len(ASL_OWN_RECORDINGS)}")
        print(f"LSM esperados         : {len(allowed_concepts)}")
        print(f"Ya registrados        : {len(already_complete)}")
        print(f"Por registrar         : {len(pending)}")

        if pending:
            print()
            print("Se agregarán:")
            for concept_id, language in pending:
                print(f"  - {concept_id} / {language} / local")
        else:
            print()
            print("No hay registros nuevos por agregar.")

        print()
        confirmation = input(
            "¿Continuar con el registro masivo? [S/N]: "
        ).strip().upper()

        if confirmation not in {"S", "SI", "SÍ", "Y", "YES"}:
            print()
            print("Registro cancelado. No se modificó ningún archivo.")
            return 0

        if not pending:
            print()
            print("Todo ya estaba registrado.")
            return 0

        backup_dir = create_backup()

        try:
            for concept_id, language in pending:
                location = build_local_location(
                    concept_id,
                    language,
                )

                physical_path = PROJECT_ROOT / location

                if not physical_path.is_file():
                    raise RegistrationError(
                        "No se encontró el archivo esperado:\n"
                        f"{physical_path}"
                    )

                resources.append(
                    {
                        "concept_id": concept_id,
                        "language": language,
                        "delivery": "local",
                        "location": location,
                        "type": "video",
                        "available": True,
                    }
                )

                source_name, source_url, notes = build_source_metadata(
                    concept_id,
                    language,
                )

                source_rows.append(
                    {
                        "concept_id": concept_id,
                        "language": language,
                        "delivery": "local",
                        "source_name": source_name,
                        "source_url": source_url,
                        "notes": notes,
                    }
                )

            save_resource_catalog(catalog)
            save_source_rows(source_rows)

        except Exception:
            backup_catalog = backup_dir / RESOURCE_CATALOG_PATH.name
            backup_sources = backup_dir / RESOURCE_SOURCES_PATH.name

            if backup_catalog.is_file():
                shutil.copy2(
                    backup_catalog,
                    RESOURCE_CATALOG_PATH,
                )

            if backup_sources.is_file():
                shutil.copy2(
                    backup_sources,
                    RESOURCE_SOURCES_PATH,
                )

            raise

        progress = calculate_progress(resources)

        print()
        print("====================================================")
        print(" Registro masivo completado")
        print("====================================================")
        print()
        print(f"Respaldo: {backup_dir}")
        print()
        print("Recursos locales:")
        print(f"  ASL : {progress['asl_local']}/71")
        print(f"  LSM : {progress['lsm_local']}/71")
        print(f"  Pares locales : {progress['local_pairs']}/71")
        print()

        if (
            progress["asl_local"] == 71
            and progress["lsm_local"] == 71
            and progress["local_pairs"] == 71
        ):
            print("OK: repositorio visual local completo (71/71 por idioma).")
        else:
            print(
                "ADVERTENCIA: el registro terminó, pero el catálogo local "
                "todavía no está completo."
            )

        print()
        return 0

    except RegistrationError as exc:
        print()
        print("ERROR DE REGISTRO")
        print(str(exc))
        print()
        return 1

    except KeyboardInterrupt:
        print()
        print()
        print("Registro cancelado.")
        return 130

    except Exception as exc:
        print()
        print("ERROR INESPERADO")
        print(f"{type(exc).__name__}: {exc}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

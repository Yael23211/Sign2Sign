import csv
import re
import sys
import zipfile
from collections import Counter
from pathlib import PurePosixPath


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def detect_group(filename):
    """
    Detecta un posible bloque/sesión a partir del nombre.

    IMPORTANTE:
    La letra encontrada en el nombre NO se usa como etiqueta.
    La clase oficial siempre proviene de la carpeta:
        TT/dataset/<CLASE>/
    """

    stem = PurePosixPath(filename).stem.upper()

    # Normalizamos sufijos que parecen copias de archivos.
    has_copy_marker = "COPY" in stem

    clean_stem = re.sub(
        r"\s*[-_]\s*COPY(?:\s*\(\d+\))?$",
        "",
        stem
    )

    # Fram_N5_123
    # Frame_K6_14
    match = re.match(
        r"^(FRAM|FRAME)_([A-ZÑ])(\d+)_\d+$",
        clean_stem
    )

    if match:
        style, letter, session = match.groups()

        return {
            "group": f"{style}_{letter}{session}",
            "pattern": "FRAM_CLASS_SESSION_NUMBER",
            "filename_label_hint": letter,
            "session_hint": session,
            "group_quality": "explicit",
            "copy_marker": int(has_copy_marker),
        }

    # N5_Frame123
    # K7_Frame99
    match = re.match(
        r"^([A-ZÑ])(\d+)_FRAME\d+$",
        clean_stem
    )

    if match:
        letter, session = match.groups()

        return {
            "group": f"{letter}{session}_FRAME",
            "pattern": "CLASS_SESSION_FRAME",
            "filename_label_hint": letter,
            "session_hint": session,
            "group_quality": "explicit",
            "copy_marker": int(has_copy_marker),
        }

    # N1_0155
    # K3_0000
    match = re.match(
        r"^([A-ZÑ])(\d+)_\d+$",
        clean_stem
    )

    if match:
        letter, session = match.groups()

        return {
            "group": f"{letter}{session}_NUM",
            "pattern": "CLASS_SESSION_NUMBER",
            "filename_label_hint": letter,
            "session_hint": session,
            "group_quality": "explicit",
            "copy_marker": int(has_copy_marker),
        }

    # P_normal_0092
    # S_normal_0099 - Copy
    match = re.match(
        r"^([A-ZÑ])_NORMAL_\d+$",
        clean_stem
    )

    if match:
        letter = match.group(1)

        return {
            "group": f"{letter}_NORMAL",
            "pattern": "CLASS_NORMAL_NUMBER",
            "filename_label_hint": letter,
            "session_hint": "NORMAL",
            "group_quality": "explicit",
            "copy_marker": int(has_copy_marker),
        }

    # frame_D0
    # frame_D100
    # Frame_K_60
    match = re.match(
        r"^(FRAM|FRAME)_([A-ZÑ])_?\d+$",
        clean_stem
    )

    if match:
        style, letter = match.groups()

        return {
            "group": f"{style}_{letter}",
            "pattern": "FRAME_CLASS_NUMBER",
            "filename_label_hint": letter,
            "session_hint": "",
            "group_quality": "partial",
            "copy_marker": int(has_copy_marker),
        }

    # frame_111
    if re.match(
        r"^(FRAM|FRAME)_\d+$",
        clean_stem
    ):
        return {
            "group": "GENERIC_FRAME",
            "pattern": "GENERIC_FRAME_NUMBER",
            "filename_label_hint": "",
            "session_hint": "",
            "group_quality": "generic",
            "copy_marker": int(has_copy_marker),
        }

    # IMG_..._BURST001
    if (
        clean_stem.startswith("IMG_")
        and "BURST" in clean_stem
    ):
        return {
            "group": "PHONE_BURST",
            "pattern": "PHONE_BURST",
            "filename_label_hint": "",
            "session_hint": "BURST",
            "group_quality": "partial",
            "copy_marker": int(has_copy_marker),
        }

    return {
        "group": "UNRESOLVED",
        "pattern": "UNRESOLVED",
        "filename_label_hint": "",
        "session_hint": "",
        "group_quality": "unresolved",
        "copy_marker": int(has_copy_marker),
    }


def main(zip_path, output_csv):

    rows = []

    class_counts = Counter()
    pattern_counts = Counter()
    quality_counts = Counter()
    copy_count = 0

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = PurePosixPath(item.filename)

            # Solamente imágenes del dataset real.
            #
            # TT/dataset/A/...
            # TT/dataset/B/...
            # etc.
            if (
                len(path.parts) < 4
                or path.parts[0] != "TT"
                or path.parts[1] != "dataset"
            ):
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # La carpeta es la etiqueta oficial.
            label = path.parts[2].upper()

            group_info = detect_group(
                path.name
            )

            filename_hint = (
                group_info["filename_label_hint"]
            )

            filename_folder_mismatch = int(
                bool(filename_hint)
                and filename_hint != label
            )

            row = {
                "dataset": "LSM",
                "source_archive": "LSM_original.zip",

                "path": item.filename,
                "class": label,

                "group": group_info["group"],
                "group_quality": group_info["group_quality"],
                "filename_pattern": group_info["pattern"],

                # Solo información auxiliar.
                # NO modifica la clase real.
                "filename_label_hint": filename_hint,
                "filename_folder_mismatch":
                    filename_folder_mismatch,

                "session_hint":
                    group_info["session_hint"],

                "copy_marker":
                    group_info["copy_marker"],

                # LSM todavía no trae splits oficiales.
                "split_original": "",
                "split_final": "",

                "sample_type": "letter",

                "extension":
                    path.suffix.lower(),

                "size_bytes":
                    item.file_size,

                # La validación anterior comprobó
                # el 100 % de estas imágenes.
                "valid_image": 1,

                "use_letter_classifier": 1,

                # Se llenará durante 25-31 agosto.
                "landmarks_ok": "",

                # Originalmente viene de LSM.
                "source_language": "LSM",
                "provisional_source": 0,

                "status": "accepted",
                "reject_reason": "",
                "notes": "",
            }

            rows.append(row)

            class_counts[label] += 1

            pattern_counts[
                group_info["pattern"]
            ] += 1

            quality_counts[
                group_info["group_quality"]
            ] += 1

            copy_count += (
                group_info["copy_marker"]
            )

    fieldnames = [
        "dataset",
        "source_archive",
        "path",
        "class",

        "group",
        "group_quality",
        "filename_pattern",

        "filename_label_hint",
        "filename_folder_mismatch",

        "session_hint",
        "copy_marker",

        "split_original",
        "split_final",

        "sample_type",

        "extension",
        "size_bytes",

        "valid_image",
        "use_letter_classifier",
        "landmarks_ok",

        "source_language",
        "provisional_source",

        "status",
        "reject_reason",
        "notes",
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------
    # RESUMEN
    # ---------------------------------------

    print("\n=== LSM MANIFEST ===\n")

    print(
        f"Muestras registradas : "
        f"{len(rows):,}"
    )

    print("\nPor clase:")

    for label in sorted(class_counts):

        print(
            f"  {label:<4}: "
            f"{class_counts[label]:,}"
        )

    print("\nPor patrón:")

    for pattern, count in pattern_counts.most_common():

        print(
            f"  {pattern:<30}"
            f"{count:>8,}"
        )

    print("\nPor calidad de grupo:")

    for quality, count in quality_counts.most_common():

        print(
            f"  {quality:<12}: "
            f"{count:,}"
        )

    mismatch_count = sum(
        row["filename_folder_mismatch"]
        for row in rows
    )

    unresolved_count = sum(
        1
        for row in rows
        if row["group"] == "UNRESOLVED"
    )

    print("\n=== RESUMEN ===")

    print(
        f"Imágenes para clasificador : "
        f"{len(rows):,}"
    )

    print(
        f"Marcadas como COPY         : "
        f"{copy_count:,}"
    )

    print(
        f"Nombre/carpeta diferentes  : "
        f"{mismatch_count:,}"
    )

    print(
        f"Grupo no resuelto          : "
        f"{unresolved_count:,}"
    )

    print(
        f"Manifest generado          : "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python build_lsm_manifest.py "
            "\"LSM_original.zip\" "
            "\"lsm_manifest.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
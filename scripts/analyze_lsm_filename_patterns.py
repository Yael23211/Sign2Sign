import csv
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def detect_pattern(filename):
    """
    Clasifica el formato del nombre del archivo sin asumir todavía
    que el identificador representa necesariamente una sesión.
    """

    stem = PurePosixPath(filename).stem.upper()

    patterns = [
        (
            "FRAM_CLASS_SESSION_NUMBER",
            r"^(?:FRAM|FRAME)_([A-ZÑ])(\d+)_([0-9]+)$"
        ),

        (
            "CLASS_SESSION_NUMBER",
            r"^([A-ZÑ])(\d+)_([0-9]+)$"
        ),

        (
            "CLASS_SESSION_FRAME",
            r"^([A-ZÑ])(\d+)_FRAME([0-9]+)$"
        ),

        (
            "GENERIC_FRAME_NUMBER",
            r"^(?:FRAM|FRAME)_([0-9]+)$"
        ),

        (
            "FRAME_CLASS_NUMBER",
            r"^(?:FRAM|FRAME)_([A-ZÑ])([0-9]+)$"
        ),
    ]

    for pattern_name, regex in patterns:

        match = re.match(regex, stem)

        if match:
            return pattern_name, match.groups()

    return "OTHER", ()


def main(zip_path, output_csv):

    pattern_counts = Counter()
    class_pattern_counts = Counter()

    examples = defaultdict(list)

    prefix_mismatches = Counter()
    mismatch_examples = defaultdict(list)

    total_images = 0

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = PurePosixPath(item.filename)

            if (
                len(path.parts) < 4
                or path.parts[0] != "TT"
                or path.parts[1] != "dataset"
            ):
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            folder_class = path.parts[2].upper()

            pattern_name, groups = detect_pattern(
                path.name
            )

            total_images += 1

            pattern_counts[pattern_name] += 1
            class_pattern_counts[
                (folder_class, pattern_name)
            ] += 1

            if len(
                examples[(folder_class, pattern_name)]
            ) < 5:
                examples[
                    (folder_class, pattern_name)
                ].append(path.name)

            # -----------------------------------------
            # Revisar si el nombre contiene una letra
            # -----------------------------------------

            filename_class = None

            if pattern_name in {
                "FRAM_CLASS_SESSION_NUMBER",
                "CLASS_SESSION_NUMBER",
                "CLASS_SESSION_FRAME",
                "FRAME_CLASS_NUMBER"
            }:
                filename_class = groups[0]

            if (
                filename_class is not None
                and filename_class != folder_class
            ):

                key = (
                    folder_class,
                    filename_class,
                    pattern_name
                )

                prefix_mismatches[key] += 1

                if len(mismatch_examples[key]) < 5:
                    mismatch_examples[key].append(
                        path.name
                    )

    # ==========================================
    # CONSOLA
    # ==========================================

    print("\n=== PATRONES DE NOMBRES LSM ===\n")

    for pattern, count in pattern_counts.most_common():

        percentage = (
            count / total_images * 100
        )

        print(
            f"{pattern:<30}"
            f"{count:>8,} "
            f"({percentage:6.2f} %)"
        )

    print(
        f"\nTOTAL: {total_images:,} imágenes"
    )

    # ==========================================
    # POR CLASE
    # ==========================================

    print("\n=== PATRONES POR CLASE ===")

    classes = sorted({
        label
        for label, _
        in class_pattern_counts
    })

    for label in classes:

        print(f"\n--- {label} ---")

        rows = [
            (pattern, count)
            for (class_label, pattern), count
            in class_pattern_counts.items()
            if class_label == label
        ]

        rows.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for pattern, count in rows:

            print(
                f"{pattern:<30}"
                f"{count:>6,}"
            )

            for example in examples[
                (label, pattern)
            ]:
                print(
                    f"    {example}"
                )

    # ==========================================
    # POSIBLES INCONSISTENCIAS
    # ==========================================

    print(
        "\n=== PREFIJO DE ARCHIVO != "
        "CARPETA ==="
    )

    if not prefix_mismatches:

        print(
            "No se encontraron discrepancias."
        )

    else:

        for (
            folder_class,
            filename_class,
            pattern
        ), count in sorted(
            prefix_mismatches.items()
        ):

            print(
                f"\nCarpeta {folder_class} "
                f"| Nombre {filename_class} "
                f"| {count:,} imágenes"
            )

            print(
                f"Patrón: {pattern}"
            )

            for example in mismatch_examples[
                (
                    folder_class,
                    filename_class,
                    pattern
                )
            ]:
                print(
                    f"    {example}"
                )

    # ==========================================
    # CSV
    # ==========================================

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "clase",
            "patron",
            "cantidad",
            "ejemplos"
        ])

        for (
            label,
            pattern
        ), count in sorted(
            class_pattern_counts.items()
        ):

            writer.writerow([
                label,
                pattern,
                count,
                " | ".join(
                    examples[(label, pattern)]
                )
            ])

    print(
        f"\nCSV generado: {output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python "
            "analyze_lsm_filename_patterns.py "
            "\"dataset.zip\" "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
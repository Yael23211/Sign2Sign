import csv
import os
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


def detect_group(filename, label):
    """
    Intenta identificar la sesión/grupo de captura a partir
    del nombre del archivo.

    Ejemplos detectados:
        Fram_N5_129.jpg  -> N5
        Fram_N6_30.jpg   -> N6
        N1_0155.jpg      -> N1
        N3_0050.jpg      -> N3
        N5_Frame154.jpg  -> N5

    Archivos como:
        frame_111.jpg
        frame_246.jpg

    se agrupan como GENERIC_FRAME.
    """

    stem = PurePosixPath(filename).stem.upper()

    # Ejemplo: FRAM_N5_129 / FRAME_N5_129
    match = re.search(
        r"^(?:FRAM|FRAME)[_-]([A-ZÑ]\d+)[_-]",
        stem
    )

    if match:
        return match.group(1)

    # Ejemplo:
    # N1_0155
    # N3_0050
    # N5_FRAME154
    match = re.search(
        r"^([A-ZÑ]\d+)[_-]",
        stem
    )

    if match:
        return match.group(1)

    # Ejemplo:
    # FRAME_111
    # FRAM_200
    if re.match(
        r"^(?:FRAM|FRAME)[_-]?\d+",
        stem
    ):
        return "GENERIC_FRAME"

    return "UNRESOLVED"


def main(zip_path, output_csv):

    group_counts = Counter()
    class_totals = Counter()

    examples = defaultdict(list)
    unresolved_examples = []

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = PurePosixPath(item.filename)

            # Estructura esperada:
            # TT/dataset/N/archivo.jpg

            if (
                len(path.parts) < 4
                or path.parts[0] != "TT"
                or path.parts[1] != "dataset"
            ):
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = path.parts[2].upper()

            group = detect_group(
                path.name,
                label
            )

            group_counts[(label, group)] += 1
            class_totals[label] += 1

            if len(examples[(label, group)]) < 3:
                examples[(label, group)].append(
                    path.name
                )

            if (
                group == "UNRESOLVED"
                and len(unresolved_examples) < 30
            ):
                unresolved_examples.append(
                    item.filename
                )

    # -----------------------------
    # CSV
    # -----------------------------

    os.makedirs(
        os.path.dirname(output_csv),
        exist_ok=True
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "clase",
            "grupo",
            "cantidad_imagenes",
            "porcentaje_clase",
            "ejemplos"
        ])

        for label in sorted(class_totals):

            groups = [
                (group, count)
                for (group_label, group), count
                in group_counts.items()
                if group_label == label
            ]

            groups.sort(
                key=lambda x: x[1],
                reverse=True
            )

            for group, count in groups:

                percentage = (
                    count
                    / class_totals[label]
                    * 100
                )

                writer.writerow([
                    label,
                    group,
                    count,
                    f"{percentage:.2f}",
                    " | ".join(
                        examples[(label, group)]
                    )
                ])

    # -----------------------------
    # CONSOLA
    # -----------------------------

    print("\n=== GRUPOS / SESIONES LSM ===")

    for label in sorted(class_totals):

        print(
            f"\n--- CLASE {label} "
            f"({class_totals[label]:,} imágenes) ---"
        )

        groups = [
            (group, count)
            for (group_label, group), count
            in group_counts.items()
            if group_label == label
        ]

        groups.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for group, count in groups:

            percentage = (
                count
                / class_totals[label]
                * 100
            )

            print(
                f"{group:<18}"
                f"{count:>6,} "
                f"({percentage:>6.2f} %)"
            )

    print("\n=== RESUMEN ===")

    print(
        f"Clases detectadas : "
        f"{len(class_totals)}"
    )

    print(
        f"Imágenes analizadas: "
        f"{sum(class_totals.values()):,}"
    )

    explicit_groups = {
        group
        for (_, group) in group_counts
        if group not in {
            "GENERIC_FRAME",
            "UNRESOLVED"
        }
    }

    print(
        f"Grupos explícitos detectados: "
        f"{len(explicit_groups)}"
    )

    generic = sum(
        count
        for (label, group), count
        in group_counts.items()
        if group == "GENERIC_FRAME"
    )

    unresolved = sum(
        count
        for (label, group), count
        in group_counts.items()
        if group == "UNRESOLVED"
    )

    print(
        f"GENERIC_FRAME: "
        f"{generic:,}"
    )

    print(
        f"No resueltos: "
        f"{unresolved:,}"
    )

    print(
        f"CSV generado: "
        f"{output_csv}"
    )

    if unresolved_examples:

        print(
            "\n=== EJEMPLOS NO RESUELTOS ==="
        )

        for filename in unresolved_examples:
            print(filename)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python analyze_lsm_groups.py "
            "\"dataset.zip\" "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
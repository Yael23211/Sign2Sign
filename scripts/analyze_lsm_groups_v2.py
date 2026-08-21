import csv
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def detect_group(filename):
    stem = PurePosixPath(filename).stem.upper()

    # Fram_N5_123 / Frame_K6_14
    m = re.match(
        r"^(FRAM|FRAME)_([A-ZÑ])(\d+)_\d+$",
        stem
    )
    if m:
        style, letter, session = m.groups()
        return f"{style}_{letter}{session}", "FRAM_CLASS_SESSION_NUMBER"

    # N5_Frame123 / K7_Frame99
    m = re.match(
        r"^([A-ZÑ])(\d+)_FRAME\d+$",
        stem
    )
    if m:
        letter, session = m.groups()
        return f"{letter}{session}_FRAME", "CLASS_SESSION_FRAME"

    # N1_0155 / K3_0000
    m = re.match(
        r"^([A-ZÑ])(\d+)_\d+$",
        stem
    )
    if m:
        letter, session = m.groups()
        return f"{letter}{session}_NUM", "CLASS_SESSION_NUMBER"

    # P_normal_0092
    m = re.match(
        r"^([A-ZÑ])_NORMAL_\d+$",
        stem
    )
    if m:
        letter = m.group(1)
        return f"{letter}_NORMAL", "CLASS_NORMAL_NUMBER"

    # frame_D0 / frame_D100 / Frame_K_60
    m = re.match(
        r"^(FRAM|FRAME)_([A-ZÑ])_?\d+$",
        stem
    )
    if m:
        style, letter = m.groups()
        return f"{style}_{letter}", "FRAME_CLASS_NUMBER"

    # frame_111
    if re.match(r"^(FRAM|FRAME)_\d+$", stem):
        return "GENERIC_FRAME", "GENERIC_FRAME_NUMBER"

    # IMG_..._BURST001
    if stem.startswith("IMG_") and "BURST" in stem:
        return "PHONE_BURST", "PHONE_BURST"

    return "UNRESOLVED", "UNRESOLVED"


def main(zip_path, output_csv):

    counts = Counter()
    class_totals = Counter()
    pattern_counts = Counter()
    examples = defaultdict(list)

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

            label = path.parts[2].upper()

            group, pattern = detect_group(path.name)

            counts[(label, group, pattern)] += 1
            class_totals[label] += 1
            pattern_counts[pattern] += 1

            key = (label, group, pattern)

            if len(examples[key]) < 3:
                examples[key].append(path.name)

    # CSV
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
            "patron",
            "cantidad",
            "porcentaje_clase",
            "ejemplos"
        ])

        for label in sorted(class_totals):

            rows = [
                (group, pattern, count)
                for (row_label, group, pattern), count
                in counts.items()
                if row_label == label
            ]

            rows.sort(
                key=lambda x: x[2],
                reverse=True
            )

            for group, pattern, count in rows:

                pct = count / class_totals[label] * 100

                writer.writerow([
                    label,
                    group,
                    pattern,
                    count,
                    f"{pct:.2f}",
                    " | ".join(
                        examples[(label, group, pattern)]
                    )
                ])

    # Consola
    print("\n=== GRUPOS LSM V2 ===")

    for label in sorted(class_totals):

        print(
            f"\n--- {label} "
            f"({class_totals[label]:,}) ---"
        )

        rows = [
            (group, pattern, count)
            for (row_label, group, pattern), count
            in counts.items()
            if row_label == label
        ]

        rows.sort(
            key=lambda x: x[2],
            reverse=True
        )

        for group, pattern, count in rows:

            pct = count / class_totals[label] * 100

            print(
                f"{group:<20}"
                f"{count:>6,} "
                f"({pct:6.2f} %) "
                f"[{pattern}]"
            )

    print("\n=== PATRONES ===")

    total = sum(pattern_counts.values())

    for pattern, count in pattern_counts.most_common():

        print(
            f"{pattern:<30}"
            f"{count:>8,} "
            f"({count / total * 100:6.2f} %)"
        )

    unresolved = pattern_counts["UNRESOLVED"]

    print("\n=== RESUMEN ===")
    print(f"Clases               : {len(class_totals)}")
    print(f"Imágenes             : {total:,}")
    print(f"No resueltas         : {unresolved:,}")
    print(f"CSV                   : {output_csv}")


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python analyze_lsm_groups_v2.py "
            "\"LSM_original.zip\" "
            "\"salida.csv\""
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
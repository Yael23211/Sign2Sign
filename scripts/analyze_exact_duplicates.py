import csv
import sys
from collections import defaultdict


def main(input_csv):

    groups = defaultdict(list)

    with open(
        input_csv,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            groups[row["duplicate_group"]].append(row)

    same_class_groups = 0
    cross_class_groups = 0

    groups_with_copy = 0
    groups_without_copy = 0

    cross_class_details = []

    print("\n=== ANÁLISIS DE DUPLICADOS ===\n")

    for group_id, rows in groups.items():

        classes = {
            row["class"]
            for row in rows
        }

        has_copy = any(
            row["copy_in_name"] == "1"
            for row in rows
        )

        if len(classes) == 1:
            same_class_groups += 1
        else:
            cross_class_groups += 1
            cross_class_details.append(
                (group_id, rows)
            )

        if has_copy:
            groups_with_copy += 1
        else:
            groups_without_copy += 1

    print(
        f"Grupos totales                 : "
        f"{len(groups):,}"
    )

    print(
        f"Duplicados dentro misma clase  : "
        f"{same_class_groups:,}"
    )

    print(
        f"Duplicados entre clases        : "
        f"{cross_class_groups:,}"
    )

    print(
        f"Grupos con archivo COPY        : "
        f"{groups_with_copy:,}"
    )

    print(
        f"Grupos sin marcador COPY       : "
        f"{groups_without_copy:,}"
    )

    if cross_class_details:

        print(
            "\n=== DUPLICADOS ENTRE CLASES ==="
        )

        for group_id, rows in cross_class_details:

            print(
                f"\nGrupo {group_id}:"
            )

            for row in rows:

                print(
                    f"  Clase {row['class']} | "
                    f"{row['path']}"
                )

    else:

        print(
            "\nNo existen imágenes idénticas "
            "etiquetadas como clases diferentes."
        )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Uso: python analyze_exact_duplicates.py "
            "\"duplicados.csv\""
        )

        sys.exit(1)

    main(sys.argv[1])
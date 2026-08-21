import csv
import sys
from collections import defaultdict


def main(manifest_path, duplicates_path, output_path):

    # --------------------------------------------------
    # Leer grupos de duplicados
    # --------------------------------------------------

    duplicate_groups = defaultdict(list)

    with open(
        duplicates_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            duplicate_groups[
                row["duplicate_group"]
            ].append(row)

    # --------------------------------------------------
    # Elegir qué archivo conservar en cada grupo
    # --------------------------------------------------

    reject_map = {}
    keeper_map = {}

    for group_id, rows in duplicate_groups.items():

        # Preferimos conservar el que NO tenga COPY
        non_copy = [
            row
            for row in rows
            if row["copy_in_name"] != "1"
        ]

        if non_copy:

            # Si existe uno sin COPY, conservarlo.
            keeper = sorted(
                non_copy,
                key=lambda x: x["path"]
            )[0]

        else:

            # Si ambos parecen equivalentes,
            # elegimos de forma determinista.
            keeper = sorted(
                rows,
                key=lambda x: x["path"]
            )[0]

        keeper_path = keeper["path"]

        keeper_map[group_id] = keeper_path

        for row in rows:

            if row["path"] != keeper_path:

                reject_map[row["path"]] = {
                    "duplicate_group": group_id,
                    "keeper": keeper_path
                }

    # --------------------------------------------------
    # Leer manifest original
    # --------------------------------------------------

    rows_out = []

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        for row in reader:

            path = row["path"]

            if path in reject_map:

                info = reject_map[path]

                row["status"] = "rejected"
                row["reject_reason"] = "exact_duplicate"

                note = (
                    f"duplicate_group="
                    f"{info['duplicate_group']}; "
                    f"duplicate_of="
                    f"{info['keeper']}"
                )

                if row["notes"]:
                    row["notes"] += " | " + note
                else:
                    row["notes"] = note

            rows_out.append(row)

    # --------------------------------------------------
    # Guardar manifest limpio
    # --------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows_out)

    # --------------------------------------------------
    # Resumen
    # --------------------------------------------------

    accepted = sum(
        1
        for row in rows_out
        if row["status"] == "accepted"
    )

    rejected = sum(
        1
        for row in rows_out
        if row["status"] == "rejected"
    )

    exact_duplicates = sum(
        1
        for row in rows_out
        if row["reject_reason"] == "exact_duplicate"
    )

    print("\n=== LSM MANIFEST LIMPIO ===\n")

    print(
        f"Muestras originales       : "
        f"{len(rows_out):,}"
    )

    print(
        f"Muestras aceptadas        : "
        f"{accepted:,}"
    )

    print(
        f"Muestras rechazadas       : "
        f"{rejected:,}"
    )

    print(
        f"Duplicados exactos fuera  : "
        f"{exact_duplicates:,}"
    )

    print(
        f"Grupos procesados         : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Manifest limpio           : "
        f"{output_path}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:

        print(
            "Uso: python apply_lsm_duplicate_filter.py "
            "\"lsm_manifest.csv\" "
            "\"lsm_exact_duplicates.csv\" "
            "\"lsm_manifest_clean.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3]
    )
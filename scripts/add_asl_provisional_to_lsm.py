import csv
import sys
from collections import Counter


PROVISIONAL_CLASSES = {"E", "L", "Z"}


def main(
    lsm_manifest_path,
    asl_manifest_path,
    output_path
):

    # ==================================================
    # 1. LEER MANIFEST LSM LIMPIO
    # ==================================================

    with open(
        lsm_manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        lsm_rows = list(reader)

    # ==================================================
    # 2. LEER ASL Y SELECCIONAR E / L / Z
    # ==================================================

    provisional_rows = []

    with open(
        asl_manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for asl_row in reader:

            label = asl_row["class"]

            if label not in PROVISIONAL_CLASSES:
                continue

            if asl_row["sample_type"] != "letter":
                continue

            # ------------------------------------------
            # Convertimos la fila ASL al formato LSM
            # ------------------------------------------

            row = {
                "dataset": "LSM",

                "source_archive":
                    asl_row["source_archive"],

                "path":
                    asl_row["path"],

                # La clase objetivo dentro del
                # reconocedor LSM.
                "class": label,

                # No fingimos una sesión LSM.
                "group":
                    f"ASL_PROVISIONAL_{label}_"
                    f"{asl_row['split_original'].upper()}",

                "group_quality":
                    "external",

                "filename_pattern":
                    "ASL_PROVISIONAL",

                "filename_label_hint":
                    label,

                "filename_folder_mismatch":
                    0,

                "session_hint":
                    asl_row["split_original"],

                "copy_marker":
                    0,

                # Conservamos la separación
                # original de Kaggle.
                "split_original":
                    asl_row["split_original"],

                # Se decidirá durante landmarks.
                "split_final":
                    "",

                "sample_type":
                    "letter",

                "extension":
                    asl_row["extension"],

                "size_bytes":
                    asl_row["size_bytes"],

                "valid_image":
                    asl_row["valid_image"],

                "use_letter_classifier":
                    1,

                "landmarks_ok":
                    "",

                # Esto es lo importante:
                "source_language":
                    "ASL",

                "provisional_source":
                    1,

                "status":
                    "accepted",

                "reject_reason":
                    "",

                "notes":
                    (
                        "Provisional ASL sample used "
                        f"for missing LSM class {label}"
                    ),
            }

            provisional_rows.append(row)

    # ==================================================
    # 3. COMBINAR
    # ==================================================

    combined_rows = (
        lsm_rows
        + provisional_rows
    )

    # ==================================================
    # 4. GUARDAR
    # ==================================================

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
        writer.writerows(combined_rows)

    # ==================================================
    # 5. RESUMEN
    # ==================================================

    provisional_counts = Counter(
        row["class"]
        for row in provisional_rows
    )

    accepted_lsm = sum(
        1
        for row in lsm_rows
        if row["status"] == "accepted"
    )

    rejected_lsm = sum(
        1
        for row in lsm_rows
        if row["status"] == "rejected"
    )

    print(
        "\n=== LSM + MUESTRAS "
        "PROVISIONALES ASL ===\n"
    )

    print(
        f"LSM original aceptado     : "
        f"{accepted_lsm:,}"
    )

    print(
        f"LSM rechazado             : "
        f"{rejected_lsm:,}"
    )

    print(
        f"Muestras ASL añadidas     : "
        f"{len(provisional_rows):,}"
    )

    print("\nPor clase provisional:")

    for label in sorted(
        PROVISIONAL_CLASSES
    ):

        print(
            f"  {label}: "
            f"{provisional_counts[label]:,}"
        )

    print(
        f"\nTotal filas manifest      : "
        f"{len(combined_rows):,}"
    )

    print(
        f"Manifest generado         : "
        f"{output_path}"
    )

    print(
        "\nNOTA: Ñ permanece sin "
        "muestras de entrenamiento."
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:

        print(
            "Uso: python "
            "add_asl_provisional_to_lsm.py "
            "\"lsm_manifest_clean.csv\" "
            "\"asl_manifest.csv\" "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3]
    )
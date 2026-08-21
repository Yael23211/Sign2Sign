import csv
import sys
import zipfile
from collections import Counter
from pathlib import PurePosixPath


LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def main(zip_path, output_csv):

    rows = []

    split_counts = Counter()
    class_counts = Counter()
    type_counts = Counter()

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = PurePosixPath(item.filename)

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            parts = path.parts

            # Estructura esperada:
            # Train_Alphabet/A/imagen.png
            # Test_Alphabet/A/imagen.png
            # Train_Alphabet/Blank/imagen.png
            # Test_Alphabet/Blank/imagen.png

            if len(parts) < 3:
                continue

            root = parts[0]
            label = parts[1]

            if root == "Train_Alphabet":
                split_original = "train"

            elif root == "Test_Alphabet":
                split_original = "test"

            else:
                # Por ejemplo alphabet.jpg
                continue

            # -----------------------------
            # Tipo de muestra
            # -----------------------------

            if label in LETTERS:

                sample_type = "letter"

                # Sí participa en el clasificador alfabético
                use_letter_classifier = 1

                # No es una muestra negativa
                use_negative_detection = 0

            elif label == "Blank":

                sample_type = "negative"

                # Blank NO será clase del clasificador
                use_letter_classifier = 0

                # Sí puede utilizarse para evaluar rechazo
                # cuando no existe una mano.
                use_negative_detection = 1

            else:

                sample_type = "unknown"
                use_letter_classifier = 0
                use_negative_detection = 0

            row = {
                "dataset": "ASL",
                "source_archive": "ASL_original.zip",
                "path": item.filename,
                "class": label,
                "split_original": split_original,
                "split_final": "",
                "sample_type": sample_type,
                "extension": path.suffix.lower(),
                "size_bytes": item.file_size,
                "valid_image": 1,
                "use_letter_classifier": use_letter_classifier,
                "use_negative_detection": use_negative_detection,
                "landmarks_ok": "",
                "status": "accepted",
                "reject_reason": "",
                "notes": "",
            }

            rows.append(row)

            split_counts[split_original] += 1
            class_counts[label] += 1
            type_counts[sample_type] += 1

    # ---------------------------------
    # Guardar CSV
    # ---------------------------------

    fieldnames = [
        "dataset",
        "source_archive",
        "path",
        "class",
        "split_original",
        "split_final",
        "sample_type",
        "extension",
        "size_bytes",
        "valid_image",
        "use_letter_classifier",
        "use_negative_detection",
        "landmarks_ok",
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

    # ---------------------------------
    # Resumen
    # ---------------------------------

    print("\n=== ASL MANIFEST ===\n")

    print(f"Muestras registradas : {len(rows):,}")

    print("\nPor split original:")

    for split, count in sorted(split_counts.items()):
        print(
            f"  {split:<10}: {count:,}"
        )

    print("\nPor tipo:")

    for sample_type, count in sorted(type_counts.items()):
        print(
            f"  {sample_type:<10}: {count:,}"
        )

    print("\nPor clase:")

    for label in sorted(class_counts):

        print(
            f"  {label:<6}: "
            f"{class_counts[label]:,}"
        )

    classifier_samples = sum(
        row["use_letter_classifier"]
        for row in rows
    )

    negative_samples = sum(
        row["use_negative_detection"]
        for row in rows
    )

    print("\n=== RESUMEN ===")

    print(
        f"Imágenes para clasificador : "
        f"{classifier_samples:,}"
    )

    print(
        f"Imágenes Blank negativas   : "
        f"{negative_samples:,}"
    )

    print(
        f"Manifest generado          : "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python build_asl_manifest.py "
            "\"ASL_original.zip\" "
            "\"asl_manifest.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
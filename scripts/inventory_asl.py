import csv
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath


EXPECTED_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NEGATIVE_CLASSES = ["Blank"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

SPLIT_FOLDERS = {
    "Train_Alphabet": "train",
    "Test_Alphabet": "test"
}


def main(zip_path, output_csv):

    # Conteo por split y letra
    class_counts = Counter()
    class_sizes = defaultdict(int)

    extension_counts = Counter()

    total_files = 0
    total_images = 0

    auxiliary_files = []
    unexpected_classes = Counter()

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            total_files += 1

            path = PurePosixPath(item.filename)
            extension = path.suffix.lower()

            extension_counts[extension or "[sin extensión]"] += 1

            parts = path.parts

            # Estructura esperada:
            #
            # Train_Alphabet/A/imagen.png
            # Test_Alphabet/A/imagen.png

            if (
                len(parts) >= 3
                and parts[0] in SPLIT_FOLDERS
                and extension in IMAGE_EXTENSIONS
            ):

                split = SPLIT_FOLDERS[parts[0]]
                label = parts[1]

                if label in EXPECTED_CLASSES:

                    class_counts[(split, label)] += 1
                    class_sizes[(split, label)] += item.file_size

                    total_images += 1

                else:

                    unexpected_classes[
                        (split, label)
                    ] += 1

            else:

                auxiliary_files.append(
                    item.filename
                )

    # --------------------------------------------------
    # GENERAR CSV
    # --------------------------------------------------

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "clase",
            "train_imagenes",
            "test_imagenes",
            "total_imagenes",
            "train_mb",
            "test_mb",
            "total_mb",
            "estado"
        ])

        for label in EXPECTED_CLASSES:

            train_count = class_counts.get(
                ("train", label),
                0
            )

            test_count = class_counts.get(
                ("test", label),
                0
            )

            total_count = (
                train_count
                + test_count
            )

            train_size = (
                class_sizes.get(
                    ("train", label),
                    0
                )
                / (1024 ** 2)
            )

            test_size = (
                class_sizes.get(
                    ("test", label),
                    0
                )
                / (1024 ** 2)
            )

            total_size = (
                train_size
                + test_size
            )

            writer.writerow([
                label,
                train_count,
                test_count,
                total_count,
                f"{train_size:.2f}",
                f"{test_size:.2f}",
                f"{total_size:.2f}",
                (
                    "Disponible"
                    if total_count > 0
                    else "Faltante"
                )
            ])

    # --------------------------------------------------
    # MOSTRAR INVENTARIO
    # --------------------------------------------------

    print("\n=== INVENTARIO ASL ===\n")

    print(
        f"{'Clase':<7}"
        f"{'Train':>10}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print("-" * 37)

    for label in EXPECTED_CLASSES:

        train_count = class_counts.get(
            ("train", label),
            0
        )

        test_count = class_counts.get(
            ("test", label),
            0
        )

        total_count = (
            train_count
            + test_count
        )

        print(
            f"{label:<7}"
            f"{train_count:>10,}"
            f"{test_count:>10,}"
            f"{total_count:>10,}"
        )

    # --------------------------------------------------
    # TOTALES
    # --------------------------------------------------

    total_train = sum(
        class_counts.get(
            ("train", label),
            0
        )
        for label in EXPECTED_CLASSES
    )

    total_test = sum(
        class_counts.get(
            ("test", label),
            0
        )
        for label in EXPECTED_CLASSES
    )

    print("-" * 37)

    print(
        f"{'TOTAL':<7}"
        f"{total_train:>10,}"
        f"{total_test:>10,}"
        f"{total_train + total_test:>10,}"
    )

    # --------------------------------------------------
    # EXTENSIONES
    # --------------------------------------------------

    print("\n=== EXTENSIONES ===\n")

    for extension, count in extension_counts.most_common():

        print(
            f"{extension:10} : "
            f"{count:,}"
        )

    # --------------------------------------------------
    # CLASES EXTRAÑAS
    # --------------------------------------------------

    if unexpected_classes:

        print(
            "\n=== CLASES NO ESPERADAS ===\n"
        )

        for (
            split,
            label
        ), count in unexpected_classes.items():

            print(
                f"{split}/{label:15} : "
                f"{count:,}"
            )

    # --------------------------------------------------
    # ARCHIVOS AUXILIARES
    # --------------------------------------------------

    if auxiliary_files:

        print(
            "\n=== ARCHIVOS AUXILIARES ===\n"
        )

        for filename in auxiliary_files:
            print(filename)

    # --------------------------------------------------
    # RESUMEN
    # --------------------------------------------------

    print("\n=== RESUMEN ===")

    print(
        f"Archivos totales       : "
        f"{total_files:,}"
    )

    print(
        f"Imágenes Train         : "
        f"{total_train:,}"
    )

    print(
        f"Imágenes Test          : "
        f"{total_test:,}"
    )

    print(
        f"Imágenes identificadas : "
        f"{total_images:,}"
    )

    print(
        f"Archivos auxiliares    : "
        f"{len(auxiliary_files):,}"
    )

    print(
        f"CSV generado en        : "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python inventory_asl.py "
            "\"ruta_dataset.zip\" "
            "\"ruta_salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
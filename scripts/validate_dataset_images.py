import csv
import io
import sys
import zipfile
from collections import Counter
from pathlib import PurePosixPath

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_metadata(dataset_type, path):
    """
    Regresa:
        (es_imagen_dataset, split, clase)

    ASL:
        Train_Alphabet/A/...
        Test_Alphabet/A/...
        Train_Alphabet/Blank/...
        Test_Alphabet/Blank/...

    LSM:
        TT/dataset/A/...
        TT/dataset/N/...
    """

    parts = path.parts

    if dataset_type == "asl":

        if len(parts) < 3:
            return False, None, None

        root = parts[0]

        if root == "Train_Alphabet":
            split = "train"
        elif root == "Test_Alphabet":
            split = "test"
        else:
            return False, None, None

        label = parts[1]

        return True, split, label

    elif dataset_type == "lsm":

        if (
            len(parts) >= 4
            and parts[0] == "TT"
            and parts[1] == "dataset"
        ):
            label = parts[2]
            return True, "original", label

        return False, None, None

    else:
        raise ValueError(
            "dataset_type debe ser 'asl' o 'lsm'"
        )


def main(zip_path, dataset_type, output_csv):

    total_images = 0
    valid_images = 0
    invalid_images = 0

    class_total = Counter()
    class_valid = Counter()
    class_invalid = Counter()

    invalid_rows = []

    with zipfile.ZipFile(zip_path, "r") as z:

        items = [
            item
            for item in z.infolist()
            if not item.is_dir()
        ]

        for index, item in enumerate(items, start=1):

            path = PurePosixPath(item.filename)

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            is_dataset_image, split, label = get_metadata(
                dataset_type,
                path
            )

            if not is_dataset_image:
                continue

            total_images += 1
            class_total[(split, label)] += 1

            try:

                image_bytes = z.read(item)

                with Image.open(
                    io.BytesIO(image_bytes)
                ) as img:

                    # Verifica estructura interna del archivo
                    img.verify()

                valid_images += 1
                class_valid[(split, label)] += 1

            except Exception as e:

                invalid_images += 1
                class_invalid[(split, label)] += 1

                invalid_rows.append([
                    item.filename,
                    split,
                    label,
                    type(e).__name__,
                    str(e)
                ])

            if total_images % 1000 == 0:

                print(
                    f"Procesadas: "
                    f"{total_images:,} | "
                    f"Válidas: "
                    f"{valid_images:,} | "
                    f"Inválidas: "
                    f"{invalid_images:,}"
                )

    # -----------------------------------------
    # CSV DE ARCHIVOS INVÁLIDOS
    # -----------------------------------------

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "ruta",
            "split",
            "clase",
            "tipo_error",
            "detalle"
        ])

        writer.writerows(invalid_rows)

    # -----------------------------------------
    # RESULTADO POR CLASE
    # -----------------------------------------

    print("\n=== RESULTADO POR CLASE ===\n")

    for split, label in sorted(class_total):

        total = class_total[(split, label)]
        valid = class_valid[(split, label)]
        invalid = class_invalid[(split, label)]

        print(
            f"{split:<10} "
            f"{label:<8} "
            f"total={total:>6,} | "
            f"válidas={valid:>6,} | "
            f"inválidas={invalid:>4,}"
        )

    print("\n=== RESUMEN ===")

    print(
        f"Dataset                 : "
        f"{dataset_type.upper()}"
    )

    print(
        f"Imágenes analizadas     : "
        f"{total_images:,}"
    )

    print(
        f"Imágenes válidas        : "
        f"{valid_images:,}"
    )

    print(
        f"Imágenes inválidas      : "
        f"{invalid_images:,}"
    )

    if total_images:

        percentage = (
            valid_images / total_images * 100
        )

        print(
            f"Porcentaje válido       : "
            f"{percentage:.4f}%"
        )

    print(
        f"CSV de errores generado : "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:

        print(
            "Uso:\n"
            "python validate_dataset_images.py "
            "\"dataset.zip\" "
            "asl|lsm "
            "\"errores.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2].lower(),
        sys.argv[3]
    )
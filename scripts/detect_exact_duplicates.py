import csv
import sys
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def is_dataset_image(dataset_type, path):
    parts = path.parts

    if dataset_type == "asl":
        return (
            len(parts) >= 3
            and parts[0] in {
                "Train_Alphabet",
                "Test_Alphabet"
            }
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if dataset_type == "lsm":
        return (
            len(parts) >= 4
            and parts[0] == "TT"
            and parts[1] == "dataset"
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise ValueError(
        "dataset_type debe ser asl o lsm"
    )


def get_metadata(dataset_type, path):
    parts = path.parts

    if dataset_type == "asl":
        split = (
            "train"
            if parts[0] == "Train_Alphabet"
            else "test"
        )

        label = parts[1]

        return split, label

    label = parts[2]

    return "", label


def main(zip_path, dataset_type, output_csv):

    # CRC + tamaño es una forma muy rápida de
    # identificar candidatos a duplicado exacto.
    candidates = defaultdict(list)

    total_images = 0

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = PurePosixPath(
                item.filename
            )

            if not is_dataset_image(
                dataset_type,
                path
            ):
                continue

            total_images += 1

            key = (
                item.CRC,
                item.file_size
            )

            candidates[key].append(item)

        duplicate_groups = [
            items
            for items in candidates.values()
            if len(items) > 1
        ]

        rows = []

        group_id = 0
        duplicate_files = 0

        for candidate_group in duplicate_groups:

            # Confirmación byte a byte para evitar
            # depender únicamente del CRC.
            confirmed = defaultdict(list)

            for item in candidate_group:

                data = z.read(item)

                confirmed[data].append(item)

            for exact_group in confirmed.values():

                if len(exact_group) <= 1:
                    continue

                group_id += 1
                duplicate_files += len(exact_group)

                for item in exact_group:

                    path = PurePosixPath(
                        item.filename
                    )

                    split, label = get_metadata(
                        dataset_type,
                        path
                    )

                    rows.append({
                        "duplicate_group": group_id,
                        "path": item.filename,
                        "class": label,
                        "split": split,
                        "size_bytes": item.file_size,
                        "copy_in_name": int(
                            "COPY"
                            in path.stem.upper()
                        ),
                    })

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "duplicate_group",
                "path",
                "class",
                "split",
                "size_bytes",
                "copy_in_name",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # Número de imágenes redundantes que sobran
    # si conservamos una muestra de cada grupo.
    redundant_images = (
        duplicate_files - group_id
    )

    print("\n=== DUPLICADOS EXACTOS ===\n")

    print(
        f"Dataset                 : "
        f"{dataset_type.upper()}"
    )

    print(
        f"Imágenes analizadas     : "
        f"{total_images:,}"
    )

    print(
        f"Grupos duplicados       : "
        f"{group_id:,}"
    )

    print(
        f"Archivos en grupos      : "
        f"{duplicate_files:,}"
    )

    print(
        f"Imágenes redundantes    : "
        f"{redundant_images:,}"
    )

    print(
        f"CSV generado            : "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:

        print(
            "Uso: python detect_exact_duplicates.py "
            "\"dataset.zip\" "
            "asl|lsm "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2].lower(),
        sys.argv[3]
    )
import csv
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath


EXPECTED_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "Ñ", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(zip_path, output_csv):
    class_counts = Counter()
    class_sizes = defaultdict(int)
    extension_counts = Counter()

    total_files = 0
    total_images = 0
    other_files = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        for item in z.infolist():
            if item.is_dir():
                continue

            total_files += 1

            path = PurePosixPath(item.filename)
            extension = path.suffix.lower()
            extension_counts[extension or "[sin extensión]"] += 1

            # Estructura esperada:
            # TT/dataset/LETRA/archivo.jpg
            parts = path.parts

            if len(parts) >= 4 and parts[0] == "TT" and parts[1] == "dataset":
                label = parts[2]

                if extension in IMAGE_EXTENSIONS:
                    class_counts[label] += 1
                    class_sizes[label] += item.file_size
                    total_images += 1
                else:
                    other_files += 1
            else:
                other_files += 1

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "clase",
            "cantidad_imagenes",
            "tamano_descomprimido_mb",
            "estado"
        ])

        for label in EXPECTED_CLASSES:
            count = class_counts.get(label, 0)
            size_mb = class_sizes.get(label, 0) / (1024 ** 2)

            writer.writerow([
                label,
                count,
                f"{size_mb:.2f}",
                "Disponible" if count > 0 else "Faltante"
            ])

    print("\n=== INVENTARIO LSM ===\n")

    for label in EXPECTED_CLASSES:
        count = class_counts.get(label, 0)

        if count:
            print(f"{label:2} : {count:>6,} imágenes")
        else:
            print(f"{label:2} : FALTANTE")

    print("\n=== EXTENSIONES ===\n")

    for extension, count in extension_counts.most_common():
        print(f"{extension:10} : {count:,}")

    print("\n=== RESUMEN ===")
    print(f"Archivos totales       : {total_files:,}")
    print(f"Imágenes identificadas : {total_images:,}")
    print(f"Otros archivos         : {other_files:,}")
    print(f"CSV generado en        : {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Uso: python inventory_lsm.py "
            "\"ruta_dataset.zip\" \"ruta_salida.csv\""
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
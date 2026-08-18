import os
import sys
import zipfile


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def extract_samples(zip_path, split, letter, output_dir, n_samples=20):

    prefix = f"{split}/{letter}/"

    with zipfile.ZipFile(zip_path, "r") as z:

        files = [
            item
            for item in z.infolist()
            if not item.is_dir()
            and item.filename.startswith(prefix)
            and item.filename.lower().endswith(IMAGE_EXTENSIONS)
        ]

        files.sort(key=lambda x: x.filename)

        if not files:
            print(f"No se encontraron imágenes para {split}/{letter}")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Tomar muestras distribuidas a lo largo del conjunto
        if len(files) <= n_samples:
            selected = files
        else:
            step = (len(files) - 1) / (n_samples - 1)
            indices = [round(i * step) for i in range(n_samples)]
            selected = [files[i] for i in indices]

        for item in selected:

            destination = os.path.join(
                output_dir,
                os.path.basename(item.filename)
            )

            with z.open(item) as source, open(destination, "wb") as target:
                target.write(source.read())

        print(
            f"{split}/{letter}: "
            f"{len(selected)} muestras extraídas "
            f"de {len(files):,}"
        )


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Uso: python sample_asl.py "
            "\"dataset.zip\" "
            "\"Train_Alphabet|Test_Alphabet\" "
            "LETRA "
            "\"directorio_salida\""
        )
        sys.exit(1)

    extract_samples(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3].upper(),
        sys.argv[4]
    )
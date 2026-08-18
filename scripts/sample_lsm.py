import os
import re
import sys
import zipfile


def frame_number(filename):
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else -1


def main(zip_path, letter, output_dir):
    prefix = f"TT/dataset/{letter}/"

    with zipfile.ZipFile(zip_path, "r") as z:
        files = [
            x for x in z.infolist()
            if not x.is_dir()
            and x.filename.startswith(prefix)
            and x.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        files.sort(key=lambda x: frame_number(x.filename))

        if not files:
            print(f"No se encontraron imágenes para {letter}")
            return

        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------
        # 1. Primeras muestras de la secuencia
        # -------------------------------------------------
        consecutive = files[:30]

        # -------------------------------------------------
        # 2. Muestras distribuidas por toda la clase
        # -------------------------------------------------
        n_samples = 30

        if len(files) <= n_samples:
            distributed = files
        else:
            step = (len(files) - 1) / (n_samples - 1)
            indices = [round(i * step) for i in range(n_samples)]
            distributed = [files[i] for i in indices]

        groups = {
            "consecutive": consecutive,
            "distributed": distributed
        }

        for group_name, samples in groups.items():
            group_dir = os.path.join(output_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            for item in samples:
                destination = os.path.join(
                    group_dir,
                    os.path.basename(item.filename)
                )

                with z.open(item) as source, open(destination, "wb") as target:
                    target.write(source.read())

        print(f"Clase analizada: {letter}")
        print(f"Total de imágenes: {len(files):,}")
        print(f"Muestras consecutivas: {len(consecutive)}")
        print(f"Muestras distribuidas: {len(distributed)}")
        print(f"Resultado: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Uso: python sample_lsm.py "
            "\"dataset.zip\" LETRA \"directorio_salida\""
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2].upper(), sys.argv[3])
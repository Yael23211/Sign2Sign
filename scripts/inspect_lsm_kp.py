import os
import sys
import zipfile
from pathlib import PurePosixPath


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def classify_sample(folder, filename):

    name = filename.upper()

    if folder == "K":

        if name.startswith(("K", "FRAM_K", "FRAME_K")):
            return "K_folder_K_name"

        if name.startswith(("P", "FRAM_P", "FRAME_P")):
            return "K_folder_P_name"

        if name.startswith(("FRAME_", "FRAM_")):
            return "K_folder_generic"

    if folder == "P":

        if name.startswith(("K", "FRAM_K", "FRAME_K")):
            return "P_folder_K_name"

        if name.startswith(("P", "FRAM_P", "FRAME_P")):
            return "P_folder_P_name"

        if name.startswith(("FRAME_", "FRAM_")):
            return "P_folder_generic"

    return None


def main(zip_path, output_dir, max_per_group=15):

    groups = {}

    with zipfile.ZipFile(zip_path, "r") as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = PurePosixPath(item.filename)

            if (
                len(path.parts) < 4
                or path.parts[0] != "TT"
                or path.parts[1] != "dataset"
            ):
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            folder = path.parts[2].upper()

            if folder not in {"K", "P"}:
                continue

            group = classify_sample(
                folder,
                path.name
            )

            if group is None:
                continue

            groups.setdefault(
                group,
                []
            ).append(item)

        for group, files in groups.items():

            files.sort(
                key=lambda x: x.filename
            )

            if len(files) > max_per_group:

                step = (
                    (len(files) - 1)
                    / (max_per_group - 1)
                )

                indices = [
                    round(i * step)
                    for i in range(max_per_group)
                ]

                selected = [
                    files[i]
                    for i in indices
                ]

            else:
                selected = files

            group_dir = os.path.join(
                output_dir,
                group
            )

            os.makedirs(
                group_dir,
                exist_ok=True
            )

            for item in selected:

                destination = os.path.join(
                    group_dir,
                    os.path.basename(
                        item.filename
                    )
                )

                with z.open(item) as source, open(
                    destination,
                    "wb"
                ) as target:

                    target.write(
                        source.read()
                    )

            print(
                f"{group}: "
                f"{len(selected)} muestras "
                f"de {len(files):,}"
            )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python inspect_lsm_kp.py "
            "\"dataset.zip\" "
            "\"directorio_salida\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
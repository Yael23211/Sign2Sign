import csv
import sys
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath

import cv2
import mediapipe as mp
import numpy as np


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

CLASSES = ["A", "M", "N"]

VARIANTS = (
    "baseline",
    "resize_640x480",
    "flip",
    "resize_flip",
)


def get_class_images(z, target_class):

    files = []

    for item in z.infolist():

        if item.is_dir():
            continue

        path = PurePosixPath(item.filename)

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = path.parts

        if (
            len(parts) >= 4
            and parts[0] == "TT"
            and parts[1] == "dataset"
            and parts[2].upper() == target_class
        ):
            files.append(item)

    return files


def distributed_sample(files, n):

    if len(files) <= n:
        return files

    step = (len(files) - 1) / (n - 1)

    indices = [
        round(i * step)
        for i in range(n)
    ]

    return [
        files[i]
        for i in indices
    ]


def load_manifest_groups(manifest_path):

    groups = {}

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            groups[row["path"]] = row["group"]

    return groups


def transform_image(image, variant):

    if variant == "baseline":
        return image

    if variant == "resize_640x480":

        return cv2.resize(
            image,
            (640, 480),
            interpolation=cv2.INTER_LINEAR
        )

    if variant == "flip":

        return cv2.flip(
            image,
            1
        )

    if variant == "resize_flip":

        resized = cv2.resize(
            image,
            (640, 480),
            interpolation=cv2.INTER_LINEAR
        )

        return cv2.flip(
            resized,
            1
        )

    raise ValueError(
        f"Variante desconocida: {variant}"
    )


def detect(landmarker, image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )

    return bool(
        result.hand_landmarks
    )


def main(
    zip_path,
    manifest_path,
    model_path,
    sample_size,
    output_csv
):

    manifest_groups = load_manifest_groups(
        manifest_path
    )

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=model_path
        ),

        running_mode=RunningMode.IMAGE,

        num_hands=1,

        # Mantenemos exactamente el mismo baseline.
        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,
    )

    rows = []

    class_summary = defaultdict(
        lambda: {
            "total": 0,
            "detected": 0
        }
    )

    group_summary = defaultdict(
        lambda: {
            "total": 0,
            "detected": 0
        }
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            for label in CLASSES:

                files = get_class_images(
                    z,
                    label
                )

                samples = distributed_sample(
                    files,
                    sample_size
                )

                print(
                    f"\n=== CLASE {label} ==="
                )

                print(
                    f"Imágenes disponibles : "
                    f"{len(files):,}"
                )

                print(
                    f"Muestras evaluadas   : "
                    f"{len(samples)}"
                )

                for index, item in enumerate(
                    samples,
                    start=1
                ):

                    image_bytes = z.read(
                        item
                    )

                    arr = np.frombuffer(
                        image_bytes,
                        dtype=np.uint8
                    )

                    original = cv2.imdecode(
                        arr,
                        cv2.IMREAD_COLOR
                    )

                    if original is None:

                        print(
                            f"[WARN] No se pudo leer "
                            f"{item.filename}"
                        )

                        continue

                    group = manifest_groups.get(
                        item.filename,
                        "UNKNOWN"
                    )

                    result_text = []

                    for variant in VARIANTS:

                        processed = transform_image(
                            original,
                            variant
                        )

                        success = detect(
                            landmarker,
                            processed
                        )

                        # -----------------------
                        # GLOBAL POR CLASE
                        # -----------------------

                        key_class = (
                            label,
                            variant
                        )

                        class_summary[
                            key_class
                        ]["total"] += 1

                        if success:

                            class_summary[
                                key_class
                            ]["detected"] += 1

                        # -----------------------
                        # POR GRUPO
                        # -----------------------

                        key_group = (
                            label,
                            group,
                            variant
                        )

                        group_summary[
                            key_group
                        ]["total"] += 1

                        if success:

                            group_summary[
                                key_group
                            ]["detected"] += 1

                        rows.append({
                            "class": label,
                            "path": item.filename,
                            "group": group,
                            "variant": variant,
                            "original_width":
                                original.shape[1],
                            "original_height":
                                original.shape[0],
                            "detected":
                                int(success),
                        })

                        result_text.append(
                            f"{variant}="
                            f"{'OK' if success else 'FAIL'}"
                        )

                    print(
                        f"[{index:02d}/"
                        f"{len(samples)}] "
                        + " | ".join(
                            result_text
                        )
                    )

    # =============================================
    # GUARDAR CSV
    # =============================================

    with open(
        output_csv,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "path",
                "group",
                "variant",
                "original_width",
                "original_height",
                "detected",
            ]
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # =============================================
    # GLOBAL
    # =============================================

    print(
        "\n\n=== COMPARACIÓN GLOBAL ===\n"
    )

    for label in CLASSES:

        print(
            f"--- {label} ---"
        )

        for variant in VARIANTS:

            data = class_summary[
                (label, variant)
            ]

            total = data["total"]

            detected = data[
                "detected"
            ]

            rate = (
                detected / total * 100
                if total
                else 0
            )

            print(
                f"{variant:<17} "
                f"{detected:>2}/{total:<2} "
                f"{rate:6.2f}%"
            )

        print()

    # =============================================
    # POR GRUPO
    # =============================================

    print(
        "\n=== COMPARACIÓN POR GRUPO ===\n"
    )

    groups_by_class = defaultdict(
        set
    )

    for (
        label,
        group,
        variant
    ) in group_summary:

        groups_by_class[
            label
        ].add(
            group
        )

    for label in CLASSES:

        print(
            f"\n--- CLASE {label} ---"
        )

        for group in sorted(
            groups_by_class[label]
        ):

            print(
                f"\n{group}"
            )

            for variant in VARIANTS:

                data = group_summary[
                    (
                        label,
                        group,
                        variant
                    )
                ]

                total = data["total"]

                if total == 0:
                    continue

                detected = data[
                    "detected"
                ]

                rate = (
                    detected
                    / total
                    * 100
                )

                print(
                    f"  {variant:<17} "
                    f"{detected:>2}/{total:<2} "
                    f"{rate:6.2f}%"
                )

    print(
        f"\nCSV generado: "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 6:

        print(
            "Uso:\n"
            "python "
            "compare_mediapipe_webcam_transforms.py "
            "\"LSM_original.zip\" "
            "\"lsm_manifest_clean.csv\" "
            "\"hand_landmarker.task\" "
            "N_MUESTRAS "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        sys.argv[5]
    )
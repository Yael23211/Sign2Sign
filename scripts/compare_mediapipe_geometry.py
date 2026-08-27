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
    "stretch",
    "letterbox",
    "center_4_3",
)

TARGET_W = 640
TARGET_H = 480


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


# ======================================================
# TRANSFORMACIONES
# ======================================================

def stretch(image):

    return cv2.resize(
        image,
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_LINEAR
    )


def letterbox(image):

    h, w = image.shape[:2]

    scale = min(
        TARGET_W / w,
        TARGET_H / h
    )

    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Usamos el color mediano de la imagen como padding
    # para evitar barras negras artificiales.
    median_color = np.median(
        image.reshape(-1, 3),
        axis=0
    ).astype(np.uint8)

    canvas = np.full(
        (TARGET_H, TARGET_W, 3),
        median_color,
        dtype=np.uint8
    )

    x1 = (TARGET_W - new_w) // 2
    y1 = (TARGET_H - new_h) // 2

    canvas[
        y1:y1 + new_h,
        x1:x1 + new_w
    ] = resized

    return canvas


def center_crop_4_3(image):

    h, w = image.shape[:2]

    target_ratio = 4 / 3
    current_ratio = w / h

    if current_ratio > target_ratio:

        # Imagen demasiado ancha:
        # recortar laterales.
        new_w = round(h * target_ratio)

        x1 = (w - new_w) // 2

        cropped = image[
            :,
            x1:x1 + new_w
        ]

    elif current_ratio < target_ratio:

        # Imagen demasiado alta:
        # recortar arriba/abajo.
        new_h = round(w / target_ratio)

        y1 = (h - new_h) // 2

        cropped = image[
            y1:y1 + new_h,
            :
        ]

    else:

        cropped = image

    return cv2.resize(
        cropped,
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_LINEAR
    )


def transform_image(image, variant):

    if variant == "baseline":
        return image

    if variant == "stretch":
        return stretch(image)

    if variant == "letterbox":
        return letterbox(image)

    if variant == "center_4_3":
        return center_crop_4_3(image)

    raise ValueError(
        f"Variante desconocida: {variant}"
    )


# ======================================================
# MEDIAPIPE
# ======================================================

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

        # Seguimos con el baseline.
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

                        class_key = (
                            label,
                            variant
                        )

                        class_summary[
                            class_key
                        ]["total"] += 1

                        if success:

                            class_summary[
                                class_key
                            ]["detected"] += 1

                        group_key = (
                            label,
                            group,
                            variant
                        )

                        group_summary[
                            group_key
                        ]["total"] += 1

                        if success:

                            group_summary[
                                group_key
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

    # ==================================================
    # CSV
    # ==================================================

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

    # ==================================================
    # RESUMEN GLOBAL
    # ==================================================

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
            detected = data["detected"]

            rate = (
                detected / total * 100
                if total
                else 0
            )

            print(
                f"{variant:<12} "
                f"{detected:>2}/{total:<2} "
                f"{rate:6.2f}%"
            )

        print()

    # ==================================================
    # POR GRUPO
    # ==================================================

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
                    detected / total * 100
                )

                print(
                    f"  {variant:<12} "
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
            "python compare_mediapipe_geometry.py "
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
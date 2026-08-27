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

CROP_VARIANTS = {
    "baseline": 1.00,
    "crop_85": 0.85,
    "crop_70": 0.70,
}


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


def center_crop(image, ratio):

    if ratio >= 1.0:
        return image

    h, w = image.shape[:2]

    crop_h = int(h * ratio)
    crop_w = int(w * ratio)

    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2

    cropped = image[
        y1:y1 + crop_h,
        x1:x1 + crop_w
    ]

    # Volvemos al tamaño original.
    # MediaPipe recibe así una imagen con la
    # región central "ampliada".
    return cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


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


def detect(landmarker, image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    return bool(result.hand_landmarks)


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

        # IMPORTANTE:
        # mantenemos exactamente el baseline.
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    rows = []

    summary = defaultdict(
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

    with zipfile.ZipFile(zip_path, "r") as z:

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
                    f"Muestras: {len(samples)}"
                )

                for index, item in enumerate(
                    samples,
                    start=1
                ):

                    image_bytes = z.read(item)

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

                    results_text = []

                    for variant, ratio in (
                        CROP_VARIANTS.items()
                    ):

                        processed = center_crop(
                            original,
                            ratio
                        )

                        success = detect(
                            landmarker,
                            processed
                        )

                        summary[
                            (label, variant)
                        ]["total"] += 1

                        group_summary[
                            (
                                label,
                                group,
                                variant
                            )
                        ]["total"] += 1

                        if success:

                            summary[
                                (label, variant)
                            ]["detected"] += 1

                            group_summary[
                                (
                                    label,
                                    group,
                                    variant
                                )
                            ]["detected"] += 1

                        rows.append({
                            "class": label,
                            "path": item.filename,
                            "group": group,
                            "variant": variant,
                            "crop_ratio": ratio,
                            "detected": int(success),
                        })

                        results_text.append(
                            f"{variant}="
                            f"{'OK' if success else 'FAIL'}"
                        )

                    print(
                        f"[{index:02d}/{len(samples)}] "
                        + " | ".join(results_text)
                    )

    # ==============================================
    # CSV
    # ==============================================

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
                "crop_ratio",
                "detected",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # RESUMEN GLOBAL
    # ==============================================

    print(
        "\n\n=== COMPARACIÓN GLOBAL ===\n"
    )

    for label in CLASSES:

        print(f"--- {label} ---")

        for variant in CROP_VARIANTS:

            data = summary[
                (label, variant)
            ]

            total = data["total"]
            detected = data["detected"]

            rate = (
                detected / total * 100
                if total else 0
            )

            print(
                f"{variant:<10} "
                f"{detected:>2}/{total:<2} "
                f"{rate:6.2f}%"
            )

        print()

    # ==============================================
    # RESUMEN POR GRUPO
    # ==============================================

    print(
        "\n=== COMPARACIÓN POR GRUPO ===\n"
    )

    groups_by_class = defaultdict(set)

    for (
        label,
        group,
        variant
    ) in group_summary:

        groups_by_class[label].add(
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

            for variant in CROP_VARIANTS:

                data = group_summary[
                    (
                        label,
                        group,
                        variant
                    )
                ]

                total = data["total"]
                detected = data["detected"]

                if total == 0:
                    continue

                rate = (
                    detected / total * 100
                )

                print(
                    f"  {variant:<10} "
                    f"{detected:>2}/{total:<2} "
                    f"{rate:6.2f}%"
                )

    print(
        f"\nCSV generado: {output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 6:

        print(
            "Uso:\n"
            "python compare_mediapipe_crops.py "
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
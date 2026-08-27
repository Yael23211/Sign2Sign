import csv
import sys
import zipfile

import cv2
import mediapipe as mp
import numpy as np


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)


def get_class_images(z, dataset_type, target_class):

    target_class = target_class.upper()
    files = []

    for item in z.infolist():

        if item.is_dir():
            continue

        if not item.filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        parts = item.filename.split("/")

        if dataset_type == "asl":

            if (
                len(parts) >= 3
                and parts[0] == "Train_Alphabet"
                and parts[1].upper() == target_class
            ):
                files.append(item)

        elif dataset_type == "lsm":

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


def main(
    zip_path,
    dataset_type,
    target_class,
    model_path,
    sample_size,
    output_csv
):

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    rows = []

    detected = 0
    failed = 0

    with zipfile.ZipFile(zip_path, "r") as z:

        all_files = get_class_images(
            z,
            dataset_type,
            target_class
        )

        samples = distributed_sample(
            all_files,
            sample_size
        )

        print(
            f"\nClase {target_class}: "
            f"{len(all_files):,} imágenes"
        )

        print(
            f"Muestra a evaluar: "
            f"{len(samples)}"
        )

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            for index, item in enumerate(
                samples,
                start=1
            ):

                image_bytes = z.read(item)

                arr = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    arr,
                    cv2.IMREAD_COLOR
                )

                success = False

                if image is not None:

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

                    success = bool(
                        result.hand_landmarks
                    )

                if success:
                    detected += 1
                else:
                    failed += 1

                rows.append({
                    "path": item.filename,
                    "detected": int(success),
                })

                print(
                    f"[{index:02d}/{len(samples)}] "
                    f"{'OK' if success else 'FAIL'} "
                    f"{item.filename}"
                )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "detected"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    total = detected + failed

    print("\n=== RESULTADO ===")

    print(
        f"Detectadas     : {detected}/{total}"
    )

    print(
        f"No detectadas  : {failed}/{total}"
    )

    print(
        f"Tasa detección : "
        f"{detected / total * 100:.2f}%"
    )

    print(
        f"CSV            : {output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 7:

        print(
            "Uso:\n"
            "python test_mediapipe_class.py "
            "\"dataset.zip\" "
            "asl|lsm "
            "CLASE "
            "\"hand_landmarker.task\" "
            "N_MUESTRAS "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2].lower(),
        sys.argv[3],
        sys.argv[4],
        int(sys.argv[5]),
        sys.argv[6]
    )
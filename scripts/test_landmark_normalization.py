import csv
import sys
import zipfile
from pathlib import PurePosixPath

import cv2
import mediapipe as mp
import numpy as np


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

CLASSES = ["A", "M", "N"]

CASCADE_ORDER = [
    "baseline",
    "center_4_3",
    "crop_85",
    "crop_70",
    "letterbox",
]

TARGET_W = 640
TARGET_H = 480


# =========================================================
# DATASET
# =========================================================

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

            groups[row["path"]] = (
                row.get("group", "UNKNOWN")
            )

    return groups


# =========================================================
# TRANSFORMACIONES DEL CASCADE
# =========================================================

def center_crop_ratio(image, ratio):

    if ratio >= 1.0:
        return image.copy()

    h, w = image.shape[:2]

    crop_h = int(h * ratio)
    crop_w = int(w * ratio)

    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2

    cropped = image[
        y1:y1 + crop_h,
        x1:x1 + crop_w
    ]

    # Mantiene la misma relación de aspecto
    # de la imagen original.
    return cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


def center_crop_4_3(image):

    h, w = image.shape[:2]

    target_ratio = 4 / 3
    current_ratio = w / h

    if current_ratio > target_ratio:

        new_w = round(
            h * target_ratio
        )

        x1 = (w - new_w) // 2

        cropped = image[
            :,
            x1:x1 + new_w
        ]

    elif current_ratio < target_ratio:

        new_h = round(
            w / target_ratio
        )

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

    median_color = np.median(
        image.reshape(-1, 3),
        axis=0
    ).astype(np.uint8)

    canvas = np.full(
        (TARGET_H, TARGET_W, 3),
        median_color,
        dtype=np.uint8
    )

    x1 = (
        TARGET_W - new_w
    ) // 2

    y1 = (
        TARGET_H - new_h
    ) // 2

    canvas[
        y1:y1 + new_h,
        x1:x1 + new_w
    ] = resized

    return canvas


def transform_image(image, variant):

    if variant == "baseline":

        return image.copy()

    if variant == "center_4_3":

        return center_crop_4_3(
            image
        )

    if variant == "crop_85":

        return center_crop_ratio(
            image,
            0.85
        )

    if variant == "crop_70":

        return center_crop_ratio(
            image,
            0.70
        )

    if variant == "letterbox":

        return letterbox(
            image
        )

    raise ValueError(
        f"Variante desconocida: "
        f"{variant}"
    )


# =========================================================
# MEDIAPIPE
# =========================================================

def detect_with_cascade(
    landmarker,
    original
):

    for variant in CASCADE_ORDER:

        processed = transform_image(
            original,
            variant
        )

        rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=
                mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(
            mp_image
        )

        if result.hand_landmarks:

            return (
                True,
                variant,
                processed,
                result
            )

    return (
        False,
        None,
        None,
        None
    )


# =========================================================
# NORMALIZACIÓN
# =========================================================

def normalize_image_landmarks(
    landmarks,
    width,
    height
):

    """
    MediaPipe entrega x/y normalizados
    respecto a ancho/alto.

    Los llevamos primero a una escala
    espacial comparable:

        X = x * width
        Y = y * height
        Z = z * width

    Después:
      1. muñeca (0) como origen
      2. escala = distancia 0 -> 9
    """

    points = np.array([
        [
            lm.x * width,
            lm.y * height,
            lm.z * width,
        ]
        for lm in landmarks
    ], dtype=np.float64)

    origin = points[0].copy()

    centered = (
        points - origin
    )

    scale = np.linalg.norm(
        centered[9]
    )

    if scale < 1e-8:

        raise ValueError(
            "Escala inválida para "
            "image landmarks."
        )

    normalized = (
        centered / scale
    )

    return (
        points,
        normalized,
        scale
    )


def normalize_world_landmarks(
    landmarks
):

    """
    World landmarks se reciben en
    coordenadas 3D.

    Aplicamos exactamente la misma idea:

      1. landmark 0 como origen
      2. distancia 0 -> 9 como escala
    """

    points = np.array([
        [
            lm.x,
            lm.y,
            lm.z,
        ]
        for lm in landmarks
    ], dtype=np.float64)

    origin = points[0].copy()

    centered = (
        points - origin
    )

    scale = np.linalg.norm(
        centered[9]
    )

    if scale < 1e-8:

        raise ValueError(
            "Escala inválida para "
            "world landmarks."
        )

    normalized = (
        centered / scale
    )

    return (
        points,
        normalized,
        scale
    )


# =========================================================
# VALIDACIONES
# =========================================================

def validate_normalization(
    normalized
):

    wrist_error = np.linalg.norm(
        normalized[0]
    )

    scale_error = abs(
        np.linalg.norm(
            normalized[9]
        ) - 1.0
    )

    return (
        wrist_error,
        scale_error
    )


# =========================================================
# CSV
# =========================================================

def build_fieldnames():

    fields = [
        "class",
        "path",
        "group",
        "landmarks_ok",
        "detection_variant",
        "processed_width",
        "processed_height",
        "handedness",
        "handedness_score",
        "image_scale_0_9",
        "world_scale_0_9",
        "image_wrist_error",
        "image_scale_error",
        "world_wrist_error",
        "world_scale_error",
    ]

    # Raw image landmarks
    for i in range(21):
        fields += [
            f"img_raw_x{i}",
            f"img_raw_y{i}",
            f"img_raw_z{i}",
        ]

    # Normalized image landmarks
    for i in range(21):
        fields += [
            f"img_norm_x{i}",
            f"img_norm_y{i}",
            f"img_norm_z{i}",
        ]

    # Raw world landmarks
    for i in range(21):
        fields += [
            f"world_raw_x{i}",
            f"world_raw_y{i}",
            f"world_raw_z{i}",
        ]

    # Normalized world landmarks
    for i in range(21):
        fields += [
            f"world_norm_x{i}",
            f"world_norm_y{i}",
            f"world_norm_z{i}",
        ]

    return fields


def add_landmarks_to_row(
    row,
    prefix,
    points
):

    for i in range(21):

        row[
            f"{prefix}_x{i}"
        ] = points[i][0]

        row[
            f"{prefix}_y{i}"
        ] = points[i][1]

        row[
            f"{prefix}_z{i}"
        ] = points[i][2]


# =========================================================
# MAIN
# =========================================================

def main(
    zip_path,
    manifest_path,
    model_path,
    samples_per_class,
    output_csv
):

    manifest_groups = (
        load_manifest_groups(
            manifest_path
        )
    )

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision
        .HandLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=
                model_path
        ),

        running_mode=
            RunningMode.IMAGE,

        num_hands=1,

        min_hand_detection_confidence=
            0.5,

        min_hand_presence_confidence=
            0.5,
    )

    rows = []

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with (
            HandLandmarker
            .create_from_options(
                options
            )
        ) as landmarker:

            for label in CLASSES:

                files = get_class_images(
                    z,
                    label
                )

                samples = (
                    distributed_sample(
                        files,
                        samples_per_class
                    )
                )

                print(
                    f"\n=== CLASE {label} ==="
                )

                for index, item in enumerate(
                    samples,
                    start=1
                ):

                    data = z.read(
                        item
                    )

                    arr = np.frombuffer(
                        data,
                        dtype=np.uint8
                    )

                    original = (
                        cv2.imdecode(
                            arr,
                            cv2.IMREAD_COLOR
                        )
                    )

                    if original is None:

                        print(
                            f"[FAIL] "
                            f"{item.filename}"
                        )

                        continue

                    (
                        success,
                        variant,
                        processed,
                        result
                    ) = detect_with_cascade(
                        landmarker,
                        original
                    )

                    row = {
                        "class": label,
                        "path":
                            item.filename,
                        "group":
                            manifest_groups.get(
                                item.filename,
                                "UNKNOWN"
                            ),
                        "landmarks_ok":
                            int(success),
                    }

                    if not success:

                        row[
                            "detection_variant"
                        ] = "FAILED"

                        rows.append(
                            row
                        )

                        print(
                            f"[{index:02d}/"
                            f"{len(samples)}] "
                            f"FAIL | "
                            f"{item.filename}"
                        )

                        continue

                    h, w = (
                        processed.shape[:2]
                    )

                    row[
                        "detection_variant"
                    ] = variant

                    row[
                        "processed_width"
                    ] = w

                    row[
                        "processed_height"
                    ] = h

                    # -------------------------
                    # HANDEDNESS
                    # -------------------------

                    handedness = ""
                    handedness_score = ""

                    if result.handedness:

                        category = (
                            result
                            .handedness[0][0]
                        )

                        handedness = (
                            category.category_name
                        )

                        handedness_score = (
                            category.score
                        )

                    row[
                        "handedness"
                    ] = handedness

                    row[
                        "handedness_score"
                    ] = handedness_score

                    # -------------------------
                    # IMAGE LANDMARKS
                    # -------------------------

                    (
                        img_raw,
                        img_norm,
                        img_scale
                    ) = (
                        normalize_image_landmarks(
                            result
                            .hand_landmarks[0],
                            w,
                            h
                        )
                    )

                    # -------------------------
                    # WORLD LANDMARKS
                    # -------------------------

                    (
                        world_raw,
                        world_norm,
                        world_scale
                    ) = (
                        normalize_world_landmarks(
                            result
                            .hand_world_landmarks[0]
                        )
                    )

                    # -------------------------
                    # VALIDACIONES
                    # -------------------------

                    (
                        img_wrist_error,
                        img_scale_error
                    ) = validate_normalization(
                        img_norm
                    )

                    (
                        world_wrist_error,
                        world_scale_error
                    ) = validate_normalization(
                        world_norm
                    )

                    row[
                        "image_scale_0_9"
                    ] = img_scale

                    row[
                        "world_scale_0_9"
                    ] = world_scale

                    row[
                        "image_wrist_error"
                    ] = img_wrist_error

                    row[
                        "image_scale_error"
                    ] = img_scale_error

                    row[
                        "world_wrist_error"
                    ] = world_wrist_error

                    row[
                        "world_scale_error"
                    ] = world_scale_error

                    add_landmarks_to_row(
                        row,
                        "img_raw",
                        img_raw
                    )

                    add_landmarks_to_row(
                        row,
                        "img_norm",
                        img_norm
                    )

                    add_landmarks_to_row(
                        row,
                        "world_raw",
                        world_raw
                    )

                    add_landmarks_to_row(
                        row,
                        "world_norm",
                        world_norm
                    )

                    rows.append(
                        row
                    )

                    print(
                        f"[{index:02d}/"
                        f"{len(samples)}] "
                        f"OK | "
                        f"{variant:<10} | "
                        f"img d09="
                        f"{np.linalg.norm(img_norm[9]):.6f} | "
                        f"world d09="
                        f"{np.linalg.norm(world_norm[9]):.6f}"
                    )

    # =================================================
    # CSV
    # =================================================

    fieldnames = (
        build_fieldnames()
    )

    with open(
        output_csv,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # =================================================
    # RESUMEN
    # =================================================

    valid_rows = [
        r
        for r in rows
        if r.get(
            "landmarks_ok"
        ) == 1
    ]

    print(
        "\n=== NORMALIZACIÓN COMPLETADA ==="
    )

    print(
        f"Muestras totales  : "
        f"{len(rows)}"
    )

    print(
        f"Landmarks válidos : "
        f"{len(valid_rows)}"
    )

    print(
        f"No detectadas     : "
        f"{len(rows) - len(valid_rows)}"
    )

    if valid_rows:

        max_img_wrist = max(
            float(
                r["image_wrist_error"]
            )
            for r in valid_rows
        )

        max_img_scale = max(
            float(
                r["image_scale_error"]
            )
            for r in valid_rows
        )

        max_world_wrist = max(
            float(
                r["world_wrist_error"]
            )
            for r in valid_rows
        )

        max_world_scale = max(
            float(
                r["world_scale_error"]
            )
            for r in valid_rows
        )

        print(
            "\nValidación numérica:"
        )

        print(
            "Máx error muñeca image : "
            f"{max_img_wrist:.10f}"
        )

        print(
            "Máx error escala image : "
            f"{max_img_scale:.10f}"
        )

        print(
            "Máx error muñeca world : "
            f"{max_world_wrist:.10f}"
        )

        print(
            "Máx error escala world : "
            f"{max_world_scale:.10f}"
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
            "test_landmark_normalization.py "
            "\"LSM_original.zip\" "
            "\"lsm_manifest_clean.csv\" "
            "\"hand_landmarker.task\" "
            "MUESTRAS_POR_CLASE "
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
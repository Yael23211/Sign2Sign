import cv2
import mediapipe as mp
import numpy as np


TARGET_W = 640
TARGET_H = 480

CASCADE_ORDER = [
    "baseline",
    "center_4_3",
    "crop_85",
    "crop_70",
    "letterbox",
]


# ==========================================================
# TRANSFORMACIONES
# ==========================================================

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

    x1 = (TARGET_W - new_w) // 2
    y1 = (TARGET_H - new_h) // 2

    canvas[
        y1:y1 + new_h,
        x1:x1 + new_w
    ] = resized

    return canvas


def transform_image(image, variant):

    if variant == "baseline":
        return image.copy()

    if variant == "center_4_3":
        return center_crop_4_3(image)

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
        return letterbox(image)

    raise ValueError(
        f"Variante desconocida: {variant}"
    )


# ==========================================================
# MEDIAPIPE
# ==========================================================

def create_hand_landmarker(model_path):

    options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=(
                mp.tasks.vision.RunningMode.IMAGE
            ),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
    )

    return (
        mp.tasks.vision.HandLandmarker
        .create_from_options(options)
    )


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
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(
            mp_image
        )

        if result.hand_landmarks:

            return {
                "ok": True,
                "variant": variant,
                "processed": processed,
                "result": result,
            }

    return {
        "ok": False,
        "variant": "FAILED",
        "processed": None,
        "result": None,
    }


# ==========================================================
# NORMALIZACIÓN
# ==========================================================

def normalize_points(points):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    origin = points[0].copy()

    centered = (
        points - origin
    )

    # Landmark 9 = MCP del dedo medio
    scale = np.linalg.norm(
        centered[9]
    )

    if scale < 1e-8:
        raise ValueError(
            "Escala 0->9 inválida."
        )

    normalized = (
        centered / scale
    )

    return (
        points,
        normalized,
        scale
    )


def image_landmarks_to_arrays(
    landmarks,
    width,
    height
):

    # z utiliza una escala comparable
    # a x, por eso usamos width.
    points = np.array([
        [
            lm.x * width,
            lm.y * height,
            lm.z * width,
        ]
        for lm in landmarks
    ])

    return normalize_points(
        points
    )


def world_landmarks_to_arrays(
    landmarks
):

    points = np.array([
        [
            lm.x,
            lm.y,
            lm.z,
        ]
        for lm in landmarks
    ])

    return normalize_points(
        points
    )


def add_points_to_row(
    row,
    prefix,
    points
):

    for i in range(21):

        row[f"{prefix}_x{i}"] = (
            float(points[i][0])
        )

        row[f"{prefix}_y{i}"] = (
            float(points[i][1])
        )

        row[f"{prefix}_z{i}"] = (
            float(points[i][2])
        )


def landmark_fieldnames():

    fields = []

    for prefix in [
        "img_raw",
        "img_norm",
        "world_raw",
        "world_norm",
    ]:

        for i in range(21):

            fields += [
                f"{prefix}_x{i}",
                f"{prefix}_y{i}",
                f"{prefix}_z{i}",
            ]

    return fields
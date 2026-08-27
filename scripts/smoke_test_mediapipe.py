import os
import sys
import zipfile

import cv2
import mediapipe as mp
import numpy as np


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# Conexiones estándar de los 21 landmarks de la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),       # índice
    (5, 9), (9, 10), (10, 11), (11, 12),  # medio
    (9, 13), (13, 14), (14, 15), (15, 16),# anular
    (13, 17), (17, 18), (18, 19), (19, 20),# meñique
    (0, 17),
]


def find_image(zip_file, dataset_type, target_class):

    target_class = target_class.upper()

    for item in zip_file.infolist():

        if item.is_dir():
            continue

        name = item.filename

        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        parts = name.split("/")

        if dataset_type == "asl":

            if (
                len(parts) >= 3
                and parts[0] == "Train_Alphabet"
                and parts[1].upper() == target_class
            ):
                return item

        elif dataset_type == "lsm":

            if (
                len(parts) >= 4
                and parts[0] == "TT"
                and parts[1] == "dataset"
                and parts[2].upper() == target_class
            ):
                return item

    return None


def draw_landmarks(image, landmarks):

    annotated = image.copy()

    height, width = annotated.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Dibujar conexiones
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            annotated,
            points[start],
            points[end],
            (255, 255, 255),
            2
        )

    # Dibujar landmarks
    for index, point in enumerate(points):

        cv2.circle(
            annotated,
            point,
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            annotated,
            str(index),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1
        )

    return annotated


def main(
    zip_path,
    dataset_type,
    target_class,
    model_path,
    output_path
):

    # --------------------------------------------
    # LEER IMAGEN DESDE ZIP
    # --------------------------------------------

    with zipfile.ZipFile(zip_path, "r") as z:

        item = find_image(
            z,
            dataset_type,
            target_class
        )

        if item is None:

            print(
                f"No se encontró una imagen "
                f"de la clase {target_class}"
            )

            return

        image_bytes = z.read(item)

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(
            "No se pudo decodificar la imagen."
        )

        return

    print("\n=== MEDIAPIPE TASKS SMOKE TEST ===\n")

    print(
        f"Dataset : {dataset_type.upper()}"
    )

    print(
        f"Clase   : {target_class.upper()}"
    )

    print(
        f"Imagen  : {item.filename}"
    )

    print(
        f"Tamaño  : "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # --------------------------------------------
    # BGR → RGB
    # --------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # --------------------------------------------
    # MEDIAPIPE TASKS
    # --------------------------------------------

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

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5
    )

    # --------------------------------------------
    # DETECCIÓN
    # --------------------------------------------

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(
            mp_image
        )

    if not result.hand_landmarks:

        print(
            "\n❌ No se detectó ninguna mano."
        )

        return

    hand = result.hand_landmarks[0]

    print("\n✅ Mano detectada.")

    print(
        f"Landmarks encontrados: "
        f"{len(hand)}"
    )

    # --------------------------------------------
    # HANDEDNESS
    # --------------------------------------------

    if result.handedness:

        handedness = (
            result.handedness[0][0]
        )

        print(
            f"Mano detectada como: "
            f"{handedness.category_name}"
        )

        print(
            f"Confianza handedness: "
            f"{handedness.score:.4f}"
        )

    # --------------------------------------------
    # MOSTRAR COORDENADAS
    # --------------------------------------------

    print("\nPrimeros landmarks:")

    for index, landmark in enumerate(
        hand[:5]
    ):

        print(
            f"{index:02d} → "
            f"x={landmark.x:.4f}, "
            f"y={landmark.y:.4f}, "
            f"z={landmark.z:.4f}"
        )

    # --------------------------------------------
    # DIBUJAR
    # --------------------------------------------

    annotated = draw_landmarks(
        image,
        hand
    )

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    cv2.imwrite(
        output_path,
        annotated
    )

    print(
        f"\nImagen anotada: "
        f"{output_path}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 6:

        print(
            "Uso:\n"
            "python smoke_test_mediapipe.py "
            "\"dataset.zip\" "
            "asl|lsm "
            "CLASE "
            "\"hand_landmarker.task\" "
            "\"salida.jpg\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2].lower(),
        sys.argv[3],
        sys.argv[4],
        sys.argv[5]
    )
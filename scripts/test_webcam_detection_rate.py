import sys
import time

import cv2
import mediapipe as mp


# ==============================
# CONFIGURACIÓN
# ==============================

TOTAL_FRAMES = 100
COUNTDOWN_SECONDS = 3

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

GUIA_X1 = 170
GUIA_Y1 = 60
GUIA_X2 = 470
GUIA_Y2 = 360


def dibujar_guia(frame):

    cv2.rectangle(
        frame,
        (GUIA_X1, GUIA_Y1),
        (GUIA_X2, GUIA_Y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Coloca la mano dentro del cuadro",
        (GUIA_X1 - 30, GUIA_Y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def main(label, model_path):

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

    detector = HandLandmarker.create_from_options(
        options
    )

    camara = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    camara.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camara.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not camara.isOpened():

        print(
            "No se pudo abrir la cámara."
        )

        detector.close()

        return

    print("\n=== PRUEBA DE DETECCIÓN WEBCAM ===")
    print(f"Seña: {label}")
    print(
        "Mantén la seña relativamente estable "
        "durante toda la prueba."
    )
    print(
        "No intentes corregirla si los landmarks "
        "desaparecen."
    )

    # ==============================
    # CUENTA REGRESIVA
    # ==============================

    start = time.time()

    while True:

        ret, frame = camara.read()

        if not ret:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        dibujar_guia(frame)

        elapsed = time.time() - start

        remaining = (
            COUNTDOWN_SECONDS
            - int(elapsed)
        )

        cv2.putText(
            frame,
            f"Preparando prueba de {label}",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            str(max(remaining, 1)),
            (290, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 255),
            5
        )

        cv2.imshow(
            "Webcam Detection Rate",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):

            detector.close()
            camara.release()
            cv2.destroyAllWindows()

            return

        if elapsed >= COUNTDOWN_SECONDS:
            break

    # ==============================
    # MEDICIÓN
    # ==============================

    detectados = 0
    no_detectados = 0

    frame_index = 0

    print(
        "\nIniciando medición..."
    )

    while frame_index < TOTAL_FRAMES:

        ret, frame = camara.read()

        if not ret:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        resultado = detector.detect(
            mp_image
        )

        success = bool(
            resultado.hand_landmarks
        )

        if success:
            detectados += 1
        else:
            no_detectados += 1

        frame_index += 1

        # ==========================
        # INTERFAZ
        # ==========================

        dibujar_guia(frame)

        estado = (
            "DETECTADA"
            if success
            else "NO DETECTADA"
        )

        cv2.putText(
            frame,
            f"Sena: {label}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_index}/{TOTAL_FRAMES}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            estado,
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                (0, 255, 0)
                if success
                else (0, 0, 255)
            ),
            2
        )

        cv2.imshow(
            "Webcam Detection Rate",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

    detector.close()

    camara.release()

    cv2.destroyAllWindows()

    total = (
        detectados
        + no_detectados
    )

    tasa = (
        detectados
        / total
        * 100
        if total
        else 0
    )

    print(
        "\n=== RESULTADO WEBCAM ==="
    )

    print(
        f"Seña              : {label}"
    )

    print(
        f"Frames analizados : {total}"
    )

    print(
        f"Detectados        : {detectados}"
    )

    print(
        f"No detectados     : {no_detectados}"
    )

    print(
        f"Tasa detección    : {tasa:.2f}%"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso:\n"
            "python test_webcam_detection_rate.py "
            "CLASE "
            "\"hand_landmarker.task\""
        )

        sys.exit(1)

    main(
        sys.argv[1].upper(),
        sys.argv[2]
    )
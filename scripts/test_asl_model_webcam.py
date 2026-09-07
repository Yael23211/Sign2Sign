import os

# Reducir logs de TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import inspect
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf


# ==========================================================
# ROOT DEL PROYECTO
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==========================================================
# PIPELINE COMPARTIDO
# ==========================================================

from src.landmarks.pipeline import (
    create_hand_landmarker,
    detect_with_cascade,
    image_landmarks_to_arrays,
)


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


DEFAULT_HAND_MODEL = (
    ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)


DEFAULT_ASL_MODEL = (
    ROOT
    / "models"
    / "asl"
    / "A3_mlp_img_norm.keras"
)


DEFAULT_SCALER = (
    ROOT
    / "models"
    / "asl"
    / "scaler_img_norm.joblib"
)


# ==========================================================
# LANDMARKS -> IMG_NORM
# ==========================================================

def landmarks_to_feature_vector(
    result,
    processed_image
):
    """
    Convierte la detección de MediaPipe al mismo vector
    img_norm de 63 características utilizado para entrenar A3.
    """

    if result is None:
        return None

    if not hasattr(result, "hand_landmarks"):
        raise RuntimeError(
            "El resultado recibido no parece ser "
            "un resultado válido de MediaPipe."
        )

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    height, width = processed_image.shape[:2]

    # ------------------------------------------------------
    # Llamamos a la función REAL del pipeline.
    #
    # Dependiendo de cómo quedó definida localmente,
    # puede recibir solo landmarks o landmarks,width,height.
    # ------------------------------------------------------

    signature = inspect.signature(
        image_landmarks_to_arrays
    )

    param_count = len(signature.parameters)

    if param_count == 1:

        output = image_landmarks_to_arrays(
            landmarks
        )

    elif param_count >= 3:

        output = image_landmarks_to_arrays(
            landmarks,
            width,
            height
        )

    else:

        raise RuntimeError(
            "Firma inesperada de "
            "image_landmarks_to_arrays(): "
            f"{signature}"
        )

    # ------------------------------------------------------
    # Esperamos:
    # raw, normalized, scale
    # ------------------------------------------------------

    if not isinstance(
        output,
        (tuple, list)
    ):

        raise RuntimeError(
            "image_landmarks_to_arrays() "
            "no devolvió una tupla/lista."
        )

    if len(output) < 2:

        raise RuntimeError(
            "image_landmarks_to_arrays() "
            "no devolvió raw + normalized."
        )

    raw = output[0]
    normalized = output[1]

    features = np.asarray(
        normalized,
        dtype=np.float32
    ).reshape(-1)

    # ------------------------------------------------------
    # Validaciones
    # ------------------------------------------------------

    if features.shape != (63,):

        raise RuntimeError(
            "\nVector img_norm inválido.\n"
            f"Esperado : (63,)\n"
            f"Obtenido : {features.shape}"
        )

    if not np.isfinite(features).all():

        raise RuntimeError(
            "El vector img_norm contiene "
            "NaN o valores infinitos."
        )

    return features


# ==========================================================
# PREDICCIÓN A3
# ==========================================================

def predict_top3(
    model,
    scaler,
    features
):
    """
    StandardScaler -> A3 -> Top-3.
    """

    X = features.reshape(
        1,
        -1
    )

    # IMPORTANTE:
    # El scaler ya fue ajustado con TRAIN.
    # Aquí JAMÁS hacemos fit().
    X_scaled = scaler.transform(
        X
    ).astype(
        np.float32
    )

    probabilities = model.predict(
        X_scaled,
        verbose=0
    )[0]

    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    top3 = []

    for idx in top3_indices:

        idx = int(idx)

        top3.append(
            (
                CLASSES[idx],
                float(probabilities[idx])
            )
        )

    return top3


# ==========================================================
# INTERPRETAR SALIDA DEL CASCADE
# ==========================================================

def unpack_detection(detection):
    """
    Extrae los elementos de detect_with_cascade().

    El pipeline actual devuelve un diccionario.
    """

    if not isinstance(
        detection,
        dict
    ):

        raise RuntimeError(
            "\ndetect_with_cascade() no devolvió "
            "un diccionario.\n"
            f"Tipo recibido: {type(detection)}\n"
            f"Valor: {detection}"
        )

    if "ok" not in detection:

        raise RuntimeError(
            "La salida del cascade no contiene "
            "la clave 'ok'.\n"
            f"Claves disponibles: "
            f"{list(detection.keys())}"
        )

    detected = bool(
        detection["ok"]
    )

    if not detected:

        return (
            False,
            None,
            None,
            None,
        )

    required = [
        "variant",
        "processed",
        "result",
    ]

    missing = [
        key
        for key in required
        if key not in detection
    ]

    if missing:

        raise RuntimeError(
            "Faltan claves en la salida del cascade: "
            f"{missing}\n"
            f"Claves existentes: "
            f"{list(detection.keys())}"
        )

    return (
        True,
        detection["variant"],
        detection["processed"],
        detection["result"],
    )


# ==========================================================
# INTERFAZ
# ==========================================================

def draw_interface(
    frame,
    top3=None,
    detection_variant=None,
    message=None
):

    output = frame.copy()

    width = output.shape[1]

    # Fondo superior
    cv2.rectangle(
        output,
        (0, 0),
        (width, 175),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        "Sign2Sign - ASL A3",
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "ENTER: clasificar | ESC: salir",
        (15, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1
    )

    if top3 is not None:

        label1, prob1 = top3[0]

        cv2.putText(
            output,
            (
                f"Top-1: {label1} "
                f"{prob1 * 100:.2f}%"
            ),
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        label2, prob2 = top3[1]
        label3, prob3 = top3[2]

        cv2.putText(
            output,
            (
                f"Top-2: {label2} "
                f"{prob2 * 100:.2f}%"
            ),
            (15, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1
        )

        cv2.putText(
            output,
            (
                f"Top-3: {label3} "
                f"{prob3 * 100:.2f}%"
            ),
            (15, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1
        )

        if detection_variant is not None:

            cv2.putText(
                output,
                (
                    "MediaPipe: "
                    f"{detection_variant}"
                ),
                (15, 163),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (190, 190, 190),
                1
            )

    elif message is not None:

        cv2.putText(
            output,
            message,
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return output


# ==========================================================
# CONSOLA
# ==========================================================

def print_prediction(
    top3,
    detection_variant
):

    print()
    print("=== PREDICCIÓN ASL ===")

    print(
        f"Método detección: "
        f"{detection_variant}"
    )

    print()
    print("Top-3:")

    for rank, (
        label,
        probability
    ) in enumerate(
        top3,
        start=1
    ):

        print(
            f"{rank}. "
            f"{label} "
            f"{probability * 100:7.3f}%"
        )


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    hand_model_path = Path(
        args.hand_model
    )

    asl_model_path = Path(
        args.asl_model
    )

    scaler_path = Path(
        args.scaler
    )

    # ------------------------------------------------------
    # Validación de archivos
    # ------------------------------------------------------

    required_files = [
        (
            hand_model_path,
            "Modelo MediaPipe"
        ),
        (
            asl_model_path,
            "Modelo ASL A3"
        ),
        (
            scaler_path,
            "Scaler ASL"
        ),
    ]

    for path, name in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"\n{name} no encontrado:\n"
                f"{path}"
            )

    # ------------------------------------------------------
    # Archivos
    # ------------------------------------------------------

    print()
    print("=== ARCHIVOS ===")

    print()
    print("MediaPipe:")
    print(
        f"  {hand_model_path}"
    )

    print()
    print("Modelo A3:")
    print(
        f"  {asl_model_path}"
    )

    print()
    print("Scaler:")
    print(
        f"  {scaler_path}"
    )

    # ------------------------------------------------------
    # Modelo A3
    # ------------------------------------------------------

    print()
    print("=== CARGANDO A3 ===")

    model = tf.keras.models.load_model(
        asl_model_path,
        compile=False
    )

    scaler = joblib.load(
        scaler_path
    )

    print("✅ Modelo A3 cargado.")

    print(
        "Input:",
        model.input_shape
    )

    print(
        "Output:",
        model.output_shape
    )

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None
    )

    print(
        "Scaler features:",
        scaler_features
    )

    if model.input_shape[-1] != 63:

        raise RuntimeError(
            "A3 no espera 63 características."
        )

    if model.output_shape[-1] != 26:

        raise RuntimeError(
            "A3 no tiene 26 clases."
        )

    if (
        scaler_features is not None
        and scaler_features != 63
    ):

        raise RuntimeError(
            "El StandardScaler no espera "
            "63 características."
        )

    # ------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------

    print()
    print("=== CARGANDO MEDIAPIPE ===")

    landmarker = create_hand_landmarker(
        str(hand_model_path)
    )

    print(
        "✅ MediaPipe cargado."
    )

    # ------------------------------------------------------
    # Cámara
    # ------------------------------------------------------

    print()
    print("=== ABRIENDO WEBCAM ===")

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():

        landmarker.close()

        raise RuntimeError(
            f"No se pudo abrir "
            f"la cámara {args.camera}."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # ------------------------------------------------------
    # Instrucciones
    # ------------------------------------------------------

    print()
    print(
        "=== ASL WEBCAM SMOKE TEST ==="
    )

    print()
    print(
        "ENTER -> capturar y clasificar"
    )

    print(
        "ESC   -> salir"
    )

    print()
    print("Pipeline:")

    print("webcam")
    print("  -> cascade MediaPipe")
    print("  -> image_landmarks_to_arrays")
    print("  -> img_norm")
    print("  -> StandardScaler.transform")
    print("  -> A3 MLP")
    print("  -> Top-3")

    print()
    print(
        "IMPORTANTE: no se aplica flip."
    )

    # ------------------------------------------------------
    # Estado
    # ------------------------------------------------------

    last_top3 = None
    last_variant = None

    last_message = (
        "Presiona ENTER para clasificar"
    )

    capture_number = 0

    # ------------------------------------------------------
    # Loop
    # ------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "\n❌ No se pudo leer "
                    "un frame."
                )

                break

            display = draw_interface(
                frame,
                top3=last_top3,
                detection_variant=last_variant,
                message=last_message
            )

            cv2.imshow(
                "Sign2Sign - ASL Webcam",
                display
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ==================================================
            # ESC
            # ==================================================

            if key == 27:
                break

            # ==================================================
            # ENTER
            # ==================================================

            if key in (
                10,
                13
            ):

                capture_number += 1

                captured = frame.copy()

                print()
                print(
                    "================================"
                )

                print(
                    f"Captura #{capture_number}"
                )

                print(
                    "Analizando captura..."
                )

                # ----------------------------------------------
                # CASCADE
                # ----------------------------------------------

                detection = detect_with_cascade(
                    landmarker,
                    captured
                )

                (
                    detected,
                    variant,
                    processed,
                    result,
                ) = unpack_detection(
                    detection
                )

                # ----------------------------------------------
                # Sin detección
                # ----------------------------------------------

                if not detected:

                    last_top3 = None
                    last_variant = None

                    last_message = (
                        "No se detecto una mano"
                    )

                    print(
                        "❌ No se detectó "
                        "ninguna mano."
                    )

                    continue

                # ----------------------------------------------
                # LANDMARKS -> IMG_NORM
                # ----------------------------------------------

                features = (
                    landmarks_to_feature_vector(
                        result,
                        processed
                    )
                )

                if features is None:

                    last_top3 = None
                    last_variant = None

                    last_message = (
                        "Landmarks invalidos"
                    )

                    print(
                        "❌ No se pudieron obtener "
                        "los landmarks."
                    )

                    continue

                # ----------------------------------------------
                # A3
                # ----------------------------------------------

                top3 = predict_top3(
                    model,
                    scaler,
                    features
                )

                # ----------------------------------------------
                # Estado UI
                # ----------------------------------------------

                last_top3 = top3
                last_variant = variant
                last_message = None

                # ----------------------------------------------
                # Consola
                # ----------------------------------------------

                print_prediction(
                    top3,
                    variant
                )

    finally:

        print()
        print(
            "Cerrando recursos..."
        )

        cap.release()

        landmarker.close()

        cv2.destroyAllWindows()

        print(
            "✅ Prueba finalizada."
        )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Smoke test webcam end-to-end "
            "del modelo ASL A3."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "Índice de cámara. "
            "Por defecto: 0."
        )
    )

    parser.add_argument(
        "--hand-model",
        default=str(
            DEFAULT_HAND_MODEL
        )
    )

    parser.add_argument(
        "--asl-model",
        default=str(
            DEFAULT_ASL_MODEL
        )
    )

    parser.add_argument(
        "--scaler",
        default=str(
            DEFAULT_SCALER
        )
    )

    args = parser.parse_args()

    main(args)
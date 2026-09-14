# ============================================================
# Sign2Sign
# LSM Canonical V2 - Captura manual con ENTER
#
# Modelo:
#   V2_M2_mlp_world_norm.keras
#
# Flujo:
#   Webcam -> ENTER -> MediaPipe -> world_norm
#   -> StandardScaler -> MLP -> Top-3
#
# La inferencia SOLO ocurre al presionar ENTER.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PIPELINE COMPARTIDO
# ============================================================

from src.landmarks.pipeline import (
    create_hand_landmarker,
    detect_with_cascade,
    world_landmarks_to_arrays,
    normalize_points,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = (
    ROOT
    / "models"
    / "lsm"
    / "V2_M2_mlp_world_norm.keras"
)

SCALER_PATH = (
    ROOT
    / "models"
    / "lsm"
    / "scaler_world_norm.joblib"
)

MEDIAPIPE_MODEL_PATH = (
    ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

CAMERA_INDEX = 0

CLASSES = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

PROVISIONAL_CLASSES = {
    "E",
    "L",
    "Z",
}


# ============================================================
# COLORES BGR
# ============================================================

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)


# ============================================================
# VALIDAR ARCHIVOS
# ============================================================

def validate_files():

    required = {
        "Modelo LSM": MODEL_PATH,
        "Scaler LSM": SCALER_PATH,
        "MediaPipe": MEDIAPIPE_MODEL_PATH,
    }

    missing = []

    for name, path in required.items():

        if not path.exists():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "\nFaltan archivos:\n\n"
            + "\n".join(missing)
        )


# ============================================================
# BUSCAR ARRAY 21x3
# ============================================================

def find_21x3_array(
    obj,
    prefer_normalized=True
):

    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    if isinstance(
        obj,
        np.ndarray
    ):

        arr = np.asarray(
            obj,
            dtype=np.float32
        )

        if arr.shape == (
            21,
            3
        ):
            return arr


    # --------------------------------------------------------
    # DICT
    # --------------------------------------------------------

    if isinstance(
        obj,
        dict
    ):

        preferred_keys = []

        if prefer_normalized:

            preferred_keys.extend(
                [
                    "world_norm",
                    "normalized",
                    "norm",
                    "normalized_points",
                    "points_norm",
                ]
            )

        preferred_keys.extend(
            [
                "world",
                "points",
                "raw",
                "landmarks",
            ]
        )

        for key in preferred_keys:

            if key not in obj:
                continue

            try:

                arr = np.asarray(
                    obj[key],
                    dtype=np.float32
                )

            except Exception:

                continue

            if arr.shape == (
                21,
                3
            ):
                return arr


        for value in obj.values():

            try:

                result = find_21x3_array(
                    value,
                    prefer_normalized
                )

                if result is not None:
                    return result

            except Exception:

                pass


    # --------------------------------------------------------
    # TUPLE / LIST
    # --------------------------------------------------------

    if isinstance(
        obj,
        (
            tuple,
            list
        )
    ):

        candidates = []

        for value in obj:

            try:

                arr = np.asarray(
                    value,
                    dtype=np.float32
                )

                if arr.shape == (
                    21,
                    3
                ):

                    candidates.append(
                        arr
                    )

            except Exception:

                pass


        if candidates:

            if prefer_normalized:

                return candidates[-1]

            return candidates[0]


    return None


# ============================================================
# EXTRAER WORLD_NORM
# ============================================================

def extract_world_norm(
    detection_result
):

    if detection_result is None:

        return None


    if not hasattr(
        detection_result,
        "hand_world_landmarks"
    ):

        return None


    hands = (
        detection_result
        .hand_world_landmarks
    )


    if not hands:

        return None


    hand_world = hands[0]


    # ========================================================
    # Primero intentamos usar exactamente el pipeline
    # compartido.
    # ========================================================

    try:

        output = world_landmarks_to_arrays(
            hand_world
        )

        candidate = find_21x3_array(
            output,
            prefer_normalized=True
        )

        if candidate is not None:

            features = (
                candidate
                .astype(np.float32)
                .reshape(-1)
            )

            if (
                features.shape == (63,)
                and
                np.isfinite(
                    features
                ).all()
            ):

                return features

    except Exception:

        pass


    # ========================================================
    # FALLBACK
    # ========================================================

    raw_points = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]

            for landmark in hand_world
        ],

        dtype=np.float32,
    )


    if raw_points.shape != (
        21,
        3
    ):

        raise RuntimeError(
            "World landmarks no tienen shape 21x3."
        )


    normalized_output = normalize_points(
        raw_points
    )


    normalized = find_21x3_array(
        normalized_output,
        prefer_normalized=True
    )


    if normalized is None:

        raise RuntimeError(
            "No se pudo generar world_norm."
        )


    features = (
        normalized
        .astype(np.float32)
        .reshape(-1)
    )


    if features.shape != (
        63,
    ):

        raise RuntimeError(
            f"Shape world_norm inesperado: "
            f"{features.shape}"
        )


    if not np.isfinite(
        features
    ).all():

        raise RuntimeError(
            "world_norm contiene NaN o Inf."
        )


    return features


# ============================================================
# REALIZAR UNA PREDICCIÓN
# ============================================================

def predict_frame(
    frame,
    landmarker,
    scaler,
    model
):

    # --------------------------------------------------------
    # PIPELINE MEDIAPIPE
    #
    # Firma real confirmada:
    #
    # detect_with_cascade(landmarker, original)
    # --------------------------------------------------------

    detection = detect_with_cascade(
        landmarker,
        frame
    )


    if not isinstance(
        detection,
        dict
    ):

        raise RuntimeError(
            "detect_with_cascade() no devolvió dict."
        )


    if not detection.get(
        "ok",
        False
    ):

        return {
            "ok": False,
            "reason": "NO_HAND",
            "variant": detection.get(
                "variant",
                "unknown"
            ),
        }


    result = detection.get(
        "result"
    )


    if result is None:

        return {
            "ok": False,
            "reason": "NO_RESULT",
            "variant": detection.get(
                "variant",
                "unknown"
            ),
        }


    # --------------------------------------------------------
    # WORLD_NORM
    # --------------------------------------------------------

    features = extract_world_norm(
        result
    )


    if features is None:

        return {
            "ok": False,
            "reason": "NO_WORLD_LANDMARKS",
            "variant": detection.get(
                "variant",
                "unknown"
            ),
        }


    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    X = scaler.transform(
        features.reshape(
            1,
            63
        )
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0
    )[0]


    if probabilities.shape != (
        26,
    ):

        raise RuntimeError(
            f"Salida inesperada del modelo: "
            f"{probabilities.shape}"
        )


    # --------------------------------------------------------
    # TOP-3
    # --------------------------------------------------------

    indices = np.argsort(
        probabilities
    )[::-1][:3]


    top3 = [
        {
            "class": CLASSES[
                int(index)
            ],

            "probability": float(
                probabilities[
                    int(index)
                ]
            ),
        }

        for index in indices
    ]


    return {
        "ok": True,

        "variant": detection.get(
            "variant",
            "unknown"
        ),

        "top3": top3,

        "features": features,
    }


# ============================================================
# PANEL BASE
# ============================================================

def draw_base_panel(
    frame
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (540, 105),
        BLACK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame
    )


    cv2.putText(
        frame,
        "Sign2Sign - LSM V2-M2",
        (25, 40),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,

        WHITE,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "ENTER: capturar | C: limpiar | Q/ESC: salir",
        (25, 75),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,

        CYAN,
        1,
        cv2.LINE_AA
    )


# ============================================================
# PANEL DE RESULTADO
# ============================================================

def draw_prediction_panel(
    frame,
    prediction
):

    if prediction is None:

        cv2.putText(
            frame,
            "Acomoda la sena y presiona ENTER",
            (25, 100),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,

            WHITE,
            1,
            cv2.LINE_AA
        )

        return


    # --------------------------------------------------------
    # ERROR / NO HAND
    # --------------------------------------------------------

    if not prediction["ok"]:

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (10, 115),
            (540, 190),
            BLACK,
            -1
        )

        cv2.addWeighted(
            overlay,
            0.60,
            frame,
            0.40,
            0,
            frame
        )


        cv2.putText(
            frame,
            "No se pudo reconocer una mano.",
            (25, 150),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,

            RED,
            2,
            cv2.LINE_AA
        )

        return


    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    top3 = prediction[
        "top3"
    ]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 115),
        (540, 275),
        BLACK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame
    )


    top1 = top3[0]

    top1_label = (
        top1["class"]
    )

    top1_prob = (
        top1["probability"]
    )


    top1_color = GREEN

    if (
        top1_label
        in PROVISIONAL_CLASSES
    ):

        top1_color = YELLOW


    cv2.putText(
        frame,
        (
            f"Resultado: {top1_label} "
            f"({top1_prob * 100:.2f}%)"
        ),
        (25, 150),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,

        top1_color,
        2,
        cv2.LINE_AA
    )


    y = 185

    for rank, item in enumerate(
        top3,
        start=1
    ):

        label = item[
            "class"
        ]

        probability = item[
            "probability"
        ]


        color = WHITE

        if label in PROVISIONAL_CLASSES:

            color = YELLOW


        cv2.putText(
            frame,
            (
                f"{rank}. {label}: "
                f"{probability * 100:.2f}%"
            ),
            (25, y),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,

            color,
            1,
            cv2.LINE_AA
        )


        y += 25


    cv2.putText(
        frame,
        (
            "Metodo: "
            f"{prediction['variant']}"
        ),
        (310, 255),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,

        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "=== SIGN2SIGN - LSM CAPTURA CON ENTER ==="
    )

    print("=" * 70)


    # ========================================================
    # VALIDAR
    # ========================================================

    validate_files()


    # ========================================================
    # SCALER
    # ========================================================

    print(
        "\nCargando scaler..."
    )

    scaler = joblib.load(
        SCALER_PATH
    )


    # ========================================================
    # MODELO
    # ========================================================

    print(
        "Cargando modelo..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


    print(
        f"Modelo cargado. "
        f"Input: {model.input_shape} | "
        f"Output: {model.output_shape}"
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    print(
        "Creando Hand Landmarker..."
    )

    landmarker = create_hand_landmarker(
        str(
            MEDIAPIPE_MODEL_PATH
        )
    )


    # ========================================================
    # WEBCAM
    # ========================================================

    print(
        "Abriendo webcam..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


    if not cap.isOpened():

        raise RuntimeError(
            "No se pudo abrir la webcam."
        )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    print()
    print("=" * 70)

    print(
        "✅ Todo cargado."
    )

    print()

    print(
        "ENTER -> capturar y clasificar sena"
    )

    print(
        "C     -> limpiar resultado"
    )

    print(
        "Q/ESC -> salir"
    )

    print()

    print(
        "Mientras no presiones ENTER, "
        "NO se ejecuta inferencia."
    )

    print("=" * 70)


    # ========================================================
    # ÚLTIMA PREDICCIÓN
    # ========================================================

    last_prediction = None


    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            ok, frame = cap.read()


            if not ok:

                continue


            # =================================================
            # FRAME PARA MOSTRAR
            # =================================================

            display_frame = (
                frame.copy()
            )


            # =================================================
            # DIBUJAR UI
            # =================================================

            draw_base_panel(
                display_frame
            )

            draw_prediction_panel(
                display_frame,
                last_prediction
            )


            cv2.imshow(
                "Sign2Sign - LSM Captura",
                display_frame
            )


            # =================================================
            # TECLA
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            # =================================================
            # SALIR
            # =================================================

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                break


            # =================================================
            # LIMPIAR
            # =================================================

            if key in (
                ord("c"),
                ord("C"),
            ):

                last_prediction = None

                print(
                    "\nResultado limpiado."
                )

                continue


            # =================================================
            # ENTER
            #
            # Windows normalmente devuelve 13.
            # 10 se contempla por compatibilidad.
            # =================================================

            if key in (
                13,
                10,
            ):

                print()
                print("-" * 60)

                print(
                    "Capturando sena..."
                )


                try:

                    last_prediction = (
                        predict_frame(
                            frame,
                            landmarker,
                            scaler,
                            model
                        )
                    )


                    if (
                        last_prediction[
                            "ok"
                        ]
                    ):

                        top3 = (
                            last_prediction[
                                "top3"
                            ]
                        )


                        print(
                            "\nTop-1:"
                        )

                        print(
                            f"{top3[0]['class']} "
                            f"= "
                            f"{top3[0]['probability'] * 100:.2f}%"
                        )


                        print(
                            "\nTop-3:"
                        )

                        for rank, item in enumerate(
                            top3,
                            start=1
                        ):

                            print(
                                f"{rank}. "
                                f"{item['class']} "
                                f"= "
                                f"{item['probability'] * 100:.2f}%"
                            )


                        print(
                            "\nMetodo:",
                            last_prediction[
                                "variant"
                            ]
                        )


                    else:

                        print(
                            "\n⚠ No se pudo "
                            "detectar/clasificar la mano."
                        )


                except Exception as exc:

                    print(
                        "\n❌ Error durante la captura:"
                    )

                    print(
                        exc
                    )


    finally:

        # ====================================================
        # CERRAR
        # ====================================================

        cap.release()

        cv2.destroyAllWindows()


        try:

            landmarker.close()

        except Exception:

            pass


        print()
        print(
            "✅ Webcam cerrada."
        )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
# ============================================================
# Sign2Sign
# Smoke test del clasificador LSM Canonical V2 por webcam
#
# Modelo:
#   V2_M2_mlp_world_norm.keras
#
# Representación:
#   world_norm -> 21 landmarks x 3 = 63 features
#
# IMPORTANTE:
#   - NO se hace flip horizontal.
#   - Se reutiliza src/landmarks/pipeline.py.
#   - El scaler SOLO hace transform().
#   - E/L/Z siguen siendo clases provisionales.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os

# Reducir un poco el ruido de TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import inspect
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# ROOT DEL PROYECTO
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


# IMPORTANTE:
# El modelo fue entrenado con sorted(A-Z).
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


# ============================================================
# UTILIDADES
# ============================================================

def print_separator():
    print("=" * 70)


def validate_files():

    files = {
        "Modelo LSM": MODEL_PATH,
        "Scaler LSM": SCALER_PATH,
        "MediaPipe": MEDIAPIPE_MODEL_PATH,
    }

    missing = []

    for name, path in files.items():

        if not path.exists():
            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "\nNo se encontraron los siguientes archivos:\n\n"
            + "\n".join(missing)
        )


# ============================================================
# DETECT_WITH_CASCADE
#
# En versiones anteriores del proyecto esta función ya cambió
# de forma durante la integración.
#
# Por eso usamos introspección para respetar la firma real
# presente actualmente en pipeline.py.
# ============================================================

def call_detect_with_cascade(
    landmarker,
    frame
):
    """
    Llama al pipeline compartido exactamente con la
    firma actual de Sign2Sign:

        detect_with_cascade(landmarker, original)

    donde 'original' es el frame BGR capturado
    directamente desde OpenCV.
    """

    return detect_with_cascade(
        landmarker,
        frame
    )
    signature = inspect.signature(
        detect_with_cascade
    )

    kwargs = {}

    timestamp_ms = int(
        time.time() * 1000
    )

    for name, parameter in signature.parameters.items():

        lname = name.lower()

        # ----------------------------------------------------
        # LANDMARKER
        # ----------------------------------------------------

        if (
            "landmarker" in lname
            or lname in {
                "detector",
                "hand_detector",
            }
        ):

            kwargs[name] = landmarker
            continue


        # ----------------------------------------------------
        # IMAGE / FRAME
        # ----------------------------------------------------

        if (
            "image" in lname
            or "frame" in lname
            or lname in {
                "img",
                "bgr",
                "image_bgr",
            }
        ):

            kwargs[name] = frame
            continue


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        if (
            "timestamp" in lname
            or lname in {
                "time_ms",
                "ts",
            }
        ):

            kwargs[name] = timestamp_ms
            continue


        # ----------------------------------------------------
        # Parámetro desconocido pero OPCIONAL
        # ----------------------------------------------------

        if (
            parameter.default
            is not inspect.Parameter.empty
        ):
            continue


        # ----------------------------------------------------
        # Parámetro desconocido y OBLIGATORIO
        # ----------------------------------------------------

        raise RuntimeError(
            "\nNo pude resolver automáticamente "
            "la firma de detect_with_cascade().\n\n"
            f"Firma detectada:\n{signature}\n\n"
            f"Parámetro desconocido obligatorio: {name}"
        )


    return detect_with_cascade(
        **kwargs
    )


# ============================================================
# EXTRAER ARRAY 21x3 DE UNA SALIDA
#
# world_landmarks_to_arrays() puede devolver estructuras
# ligeramente diferentes según la versión de pipeline.py.
#
# Esta función busca de forma defensiva un array 21x3.
# ============================================================

def find_21x3_array(
    obj,
    prefer_normalized=True
):

    # --------------------------------------------------------
    # NUMPY ARRAY DIRECTO
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


        # Buscar recursivamente.
        for value in obj.values():

            try:

                result = find_21x3_array(
                    value,
                    prefer_normalized=prefer_normalized
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

            # Normalmente:
            #
            #   raw, normalized
            #
            # por eso para world_norm preferimos el último.
            if prefer_normalized:
                return candidates[-1]

            return candidates[0]


    return None


# ============================================================
# WORLD_NORM
# ============================================================

def extract_world_norm(
    detection_result
):

    # --------------------------------------------------------
    # Validar MediaPipe result
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # INTENTO 1:
    # usar directamente la función oficial de pipeline.py
    # --------------------------------------------------------

    try:

        signature = inspect.signature(
            world_landmarks_to_arrays
        )

        kwargs = {}

        for name, parameter in signature.parameters.items():

            lname = name.lower()

            if (
                "landmark" in lname
                or "point" in lname
                or "world" in lname
                or lname in {
                    "hand",
                    "points",
                }
            ):

                kwargs[name] = hand_world
                continue


            if (
                parameter.default
                is not inspect.Parameter.empty
            ):
                continue


            raise RuntimeError(
                f"Parámetro desconocido: {name}"
            )


        output = world_landmarks_to_arrays(
            **kwargs
        )


        candidate = find_21x3_array(
            output,
            prefer_normalized=True
        )


        if candidate is not None:

            flat = (
                candidate
                .astype(np.float32)
                .reshape(-1)
            )

            if flat.shape == (
                63,
            ):

                if np.isfinite(
                    flat
                ).all():

                    return flat


    except Exception as exc:

        # No terminamos el programa todavía porque tenemos
        # fallback usando normalize_points().
        pipeline_error = exc

    else:
        pipeline_error = None


    # --------------------------------------------------------
    # INTENTO 2:
    # extraer xyz y pasar por normalize_points()
    # --------------------------------------------------------

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
            "MediaPipe no produjo 21x3 world landmarks."
        )


    normalized_output = normalize_points(
        raw_points
    )


    normalized = find_21x3_array(
        normalized_output,
        prefer_normalized=True
    )


    if normalized is None:

        message = (
            "\nNo fue posible obtener world_norm.\n"
            f"Firma world_landmarks_to_arrays: "
            f"{inspect.signature(world_landmarks_to_arrays)}\n"
            f"Firma normalize_points: "
            f"{inspect.signature(normalize_points)}\n"
        )

        if pipeline_error is not None:

            message += (
                "\nError del primer intento:\n"
                f"{pipeline_error}\n"
            )

        raise RuntimeError(
            message
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
            f"world_norm produjo shape "
            f"{features.shape}, esperaba (63,)."
        )


    if not np.isfinite(
        features
    ).all():

        raise RuntimeError(
            "world_norm contiene NaN o Inf."
        )


    return features


# ============================================================
# OVERLAY
# ============================================================

def draw_panel(
    frame,
    top3,
    method,
    detected
):

    # Fondo semitransparente.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (450, 215),
        BLACK,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame,
    )


    # --------------------------------------------------------
    # TÍTULO
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Sign2Sign - LSM V2-M2",
        (25, 40),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,

        WHITE,
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------------
    # NO DETECTION
    # --------------------------------------------------------

    if not detected:

        cv2.putText(
            frame,
            "Mano no detectada",
            (25, 85),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,

            RED,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Q / ESC: salir",
            (25, 185),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,

            WHITE,
            1,
            cv2.LINE_AA,
        )

        return


    # --------------------------------------------------------
    # TOP-1
    # --------------------------------------------------------

    top1_class = top3[0][0]
    top1_prob = top3[0][1]


    top1_color = GREEN

    if top1_class in PROVISIONAL_CLASSES:
        top1_color = YELLOW


    cv2.putText(
        frame,
        f"Top-1: {top1_class}  {top1_prob * 100:.2f}%",
        (25, 80),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,

        top1_color,
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------------
    # TOP-3
    # --------------------------------------------------------

    y = 115

    for rank, (
        label,
        probability
    ) in enumerate(
        top3,
        start=1
    ):

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
            cv2.LINE_AA,
        )

        y += 25


    # --------------------------------------------------------
    # CASCADE METHOD
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Metodo: {method}",
        (250, 185),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,

        WHITE,
        1,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_separator()

    print(
        "=== SIGN2SIGN - LSM WEBCAM SMOKE TEST ==="
    )

    print_separator()


    # ========================================================
    # ARCHIVOS
    # ========================================================

    validate_files()


    print()
    print(
        "Modelo:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Scaler:"
    )

    print(
        SCALER_PATH
    )

    print()
    print(
        "MediaPipe:"
    )

    print(
        MEDIAPIPE_MODEL_PATH
    )


    # ========================================================
    # INFO DEL PIPELINE
    # ========================================================

    print()
    print(
        "Firma detect_with_cascade:"
    )

    print(
        inspect.signature(
            detect_with_cascade
        )
    )

    print()
    print(
        "Firma world_landmarks_to_arrays:"
    )

    print(
        inspect.signature(
            world_landmarks_to_arrays
        )
    )

    print()
    print(
        "Firma normalize_points:"
    )

    print(
        inspect.signature(
            normalize_points
        )
    )


    # ========================================================
    # CARGAR SCALER
    # ========================================================

    print()
    print(
        "Cargando scaler..."
    )

    scaler = joblib.load(
        SCALER_PATH
    )


    # ========================================================
    # CARGAR MODELO
    # ========================================================

    print(
        "Cargando modelo TensorFlow..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


    # ========================================================
    # VALIDAR MODELO
    # ========================================================

    output_shape = (
        model.output_shape
    )

    if output_shape[-1] != 26:

        raise RuntimeError(
            f"El modelo tiene "
            f"{output_shape[-1]} salidas. "
            "Se esperaban 26."
        )


    print(
        f"Modelo cargado. "
        f"Input: {model.input_shape} | "
        f"Output: {model.output_shape}"
    )


    # ========================================================
    # MEDIAPIPE
    #
    # IMPORTANTE:
    # pasar str(), no Path.
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
            f"No se pudo abrir la webcam "
            f"{CAMERA_INDEX}."
        )


    # Intentamos resolución razonable.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    print()
    print_separator()

    print(
        "✅ Todo cargado."
    )

    print()
    print(
        "Haz señas LSM frente a la cámara."
    )

    print(
        "NO se aplica flip horizontal."
    )

    print(
        "Presiona Q o ESC para salir."
    )

    print_separator()


    last_error_printed = None


    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            ok, frame = cap.read()


            if not ok:

                print(
                    "⚠ No se pudo leer frame."
                )

                continue


            # =================================================
            # NO HACER FLIP
            # =================================================

            display_frame = frame.copy()


            # =================================================
            # DETECCIÓN CASCADE
            # =================================================

            try:

                detection = (
                    call_detect_with_cascade(
                        landmarker,
                        frame
                    )
                )

            except Exception as exc:

                error_text = str(
                    exc
                )

                if (
                    error_text
                    != last_error_printed
                ):

                    print()
                    print(
                        "❌ Error en "
                        "detect_with_cascade:"
                    )

                    print(
                        error_text
                    )

                    last_error_printed = (
                        error_text
                    )


                draw_panel(
                    display_frame,
                    top3=[],
                    method="ERROR",
                    detected=False,
                )


                cv2.imshow(
                    "Sign2Sign - LSM Webcam",
                    display_frame,
                )


                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break


                continue


            # =================================================
            # VALIDAR RETORNO DE CASCADE
            # =================================================

            if not isinstance(
                detection,
                dict
            ):

                raise RuntimeError(
                    "detect_with_cascade() "
                    "no devolvió dict.\n"
                    f"Tipo recibido: "
                    f"{type(detection)}"
                )


            detected = bool(
                detection.get(
                    "ok",
                    False
                )
            )


            method = str(
                detection.get(
                    "variant",
                    "unknown"
                )
            )


            if not detected:

                draw_panel(
                    display_frame,
                    top3=[],
                    method=method,
                    detected=False,
                )


                cv2.imshow(
                    "Sign2Sign - LSM Webcam",
                    display_frame,
                )


                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break


                continue


            # =================================================
            # RESULTADO MEDIAPIPE
            # =================================================

            result = detection.get(
                "result"
            )


            if result is None:

                continue


            # =================================================
            # WORLD_NORM → 63
            # =================================================

            try:

                features = (
                    extract_world_norm(
                        result
                    )
                )

            except Exception as exc:

                error_text = str(
                    exc
                )

                if (
                    error_text
                    != last_error_printed
                ):

                    print()
                    print(
                        "❌ Error extrayendo "
                        "world_norm:"
                    )

                    print(
                        error_text
                    )

                    last_error_printed = (
                        error_text
                    )

                continue


            if features is None:

                continue


            # =================================================
            # SCALER
            # =================================================

            features_scaled = (
                scaler.transform(
                    features.reshape(
                        1,
                        63
                    )
                )
                .astype(
                    np.float32
                )
            )


            # =================================================
            # PREDICCIÓN
            # =================================================

            probabilities = model.predict(
                features_scaled,
                verbose=0
            )[0]


            if probabilities.shape != (
                26,
            ):

                raise RuntimeError(
                    f"Predicción shape "
                    f"{probabilities.shape}; "
                    "esperaba (26,)."
                )


            # =================================================
            # TOP-3
            # =================================================

            top3_indices = (
                np.argsort(
                    probabilities
                )[::-1][:3]
            )


            top3 = [
                (
                    CLASSES[int(idx)],
                    float(
                        probabilities[
                            int(idx)
                        ]
                    ),
                )
                for idx in top3_indices
            ]


            # =================================================
            # OVERLAY
            # =================================================

            draw_panel(
                display_frame,
                top3=top3,
                method=method,
                detected=True,
            )


            # =================================================
            # MOSTRAR
            # =================================================

            cv2.imshow(
                "Sign2Sign - LSM Webcam",
                display_frame,
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break


    finally:

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
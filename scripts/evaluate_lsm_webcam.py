# ============================================================
# Sign2Sign
# Evaluacion controlada LSM Canonical V2 por webcam
#
# Modelo:
#   V2_M2_mlp_world_norm.keras
#
# Protocolo:
#   - 23 clases LSM genuinas
#   - 5 intentos por clase
#   - 115 intentos totales
#   - predicciones OCULTAS durante la evaluacion
#   - una captura por ENTER
#
# Se guardan:
#   - imagen exacta de cada intento
#   - resultados por intento
#   - resumen global
#   - resumen por clase
#   - pares de confusion
#   - matriz de confusion CSV/PNG
#
# E/L/Z NO forman parte de las clases reales evaluadas,
# pero SI pueden aparecer como predicciones del modelo.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


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
# CONFIGURACION
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


# Datos FUERA del repo.
OUTPUT_ROOT = Path(
    r"D:\Sign2SignData\inventory\webcam_eval\lsm"
)


CAMERA_INDEX = 0

ATTEMPTS_PER_CLASS = 5


# ============================================================
# CLASES
# ============================================================

# Las 26 salidas reales del modelo.
ALL_CLASSES = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


# 23 clases genuinas LSM.
# E / L / Z son provisionales y quedan fuera
# de la evaluacion principal.
GENUINE_CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]


PROVISIONAL_CLASSES = {
    "E",
    "L",
    "Z",
}


CLASS_TO_INDEX = {
    label: i
    for i, label in enumerate(
        ALL_CLASSES
    )
}


# ============================================================
# COLORES
# OpenCV usa BGR.
# ============================================================

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
BLACK = (0, 0, 0)


# ============================================================
# ARCHIVOS
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
            "\nFaltan archivos requeridos:\n\n"
            + "\n".join(missing)
        )


# ============================================================
# LOCALIZAR ARRAY 21 x 3
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

        keys = []

        if prefer_normalized:

            keys.extend(
                [
                    "world_norm",
                    "normalized",
                    "norm",
                    "normalized_points",
                    "points_norm",
                ]
            )


        keys.extend(
            [
                "world",
                "points",
                "raw",
                "landmarks",
            ]
        )


        for key in keys:

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


        # Busqueda recursiva.
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
    # LIST / TUPLE
    # --------------------------------------------------------

    if isinstance(
        obj,
        (
            list,
            tuple
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

            # Normalmente la salida normalizada aparece
            # despues de la cruda.
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
    # MISMO PIPELINE QUE YA PROBAMOS EN EL SMOKE TEST.
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
            f"Shape inesperado: {features.shape}"
        )


    if not np.isfinite(
        features
    ).all():

        raise RuntimeError(
            "world_norm contiene NaN o Inf."
        )


    return features


# ============================================================
# EVALUAR UN FRAME
# ============================================================

def evaluate_frame(
    frame,
    landmarker,
    scaler,
    model
):

    # --------------------------------------------------------
    # DETECCION
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
            "detect_with_cascade() no devolvio dict."
        )


    variant = detection.get(
        "variant",
        "unknown"
    )


    if not detection.get(
        "ok",
        False
    ):

        return {
            "detected": False,
            "variant": variant,
        }


    result = detection.get(
        "result"
    )


    if result is None:

        return {
            "detected": False,
            "variant": variant,
        }


    # --------------------------------------------------------
    # WORLD_NORM
    # --------------------------------------------------------

    features = extract_world_norm(
        result
    )


    if features is None:

        return {
            "detected": False,
            "variant": variant,
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
    # MLP
    # --------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0
    )[0]


    if probabilities.shape != (
        26,
    ):

        raise RuntimeError(
            f"Salida modelo inesperada: "
            f"{probabilities.shape}"
        )


    # --------------------------------------------------------
    # RANKING COMPLETO
    # --------------------------------------------------------

    ranking = np.argsort(
        probabilities
    )[::-1]


    top3_indices = ranking[:3]


    top3 = [
        {
            "class": ALL_CLASSES[
                int(index)
            ],

            "probability": float(
                probabilities[
                    int(index)
                ]
            ),
        }

        for index in top3_indices
    ]


    return {
        "detected": True,
        "variant": variant,
        "probabilities": probabilities,
        "ranking": ranking,
        "top3": top3,
    }


# ============================================================
# CREAR DIRECTORIO DE EJECUCION
# ============================================================

def create_run_directory():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    run_dir = (
        OUTPUT_ROOT
        / f"run_{timestamp}"
    )


    captures_dir = (
        run_dir
        / "captures"
    )


    captures_dir.mkdir(
        parents=True,
        exist_ok=False
    )


    return (
        run_dir,
        captures_dir
    )


# ============================================================
# UI
# ============================================================

def draw_ui(
    frame,
    target_class,
    attempt_number,
    total_attempt,
    total_attempts,
    status_message=None,
):

    overlay = frame.copy()


    # ========================================================
    # PANEL SUPERIOR
    # ========================================================

    cv2.rectangle(
        overlay,
        (10, 10),
        (620, 235),
        BLACK,
        -1
    )


    cv2.addWeighted(
        overlay,
        0.68,
        frame,
        0.32,
        0,
        frame
    )


    # ========================================================
    # TITULO
    # ========================================================

    cv2.putText(
        frame,
        "Sign2Sign - Evaluacion LSM Webcam",
        (30, 45),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,

        WHITE,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # CLASE OBJETIVO
    # ========================================================

    cv2.putText(
        frame,
        "Clase objetivo:",
        (30, 95),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,

        WHITE,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        target_class,
        (225, 100),

        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,

        CYAN,
        3,
        cv2.LINE_AA
    )


    # ========================================================
    # INTENTO
    # ========================================================

    cv2.putText(
        frame,
        (
            f"Intento de clase: "
            f"{attempt_number}/{ATTEMPTS_PER_CLASS}"
        ),
        (30, 140),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,

        WHITE,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Progreso total: "
            f"{total_attempt}/{total_attempts}"
        ),
        (30, 170),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,

        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # CONTROLES
    # ========================================================

    cv2.putText(
        frame,
        "ENTER: capturar   |   Q / ESC: cancelar",
        (30, 205),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,

        YELLOW,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # MENSAJE TEMPORAL
    # ========================================================

    if status_message:

        cv2.putText(
            frame,
            status_message,
            (650, 65),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,

            GREEN,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # IMPORTANTE:
    # NO mostrar prediccion.
    # ========================================================

    cv2.putText(
        frame,
        "Prediccion oculta durante la prueba",
        (650, 100),

        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,

        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# RESULTADOS
# ============================================================

def generate_results(
    rows,
    run_dir
):

    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # CSV PRINCIPAL
    # ========================================================

    results_path = (
        run_dir
        / "webcam_results.csv"
    )


    df.to_csv(
        results_path,
        index=False
    )


    # ========================================================
    # TOTALES
    # ========================================================

    total = len(
        df
    )


    detected = int(
        df["detected"].sum()
    )


    not_detected = (
        total
        - detected
    )


    detection_rate = (
        detected / total
        if total
        else 0.0
    )


    detected_df = df[
        df["detected"]
    ].copy()


    # ========================================================
    # TOP-1 / TOP-3 DADO DETECCION
    # ========================================================

    if detected > 0:

        top1_given_detection = float(
            detected_df[
                "top1_correct"
            ].mean()
        )


        top3_given_detection = float(
            detected_df[
                "true_in_top3"
            ].mean()
        )

    else:

        top1_given_detection = 0.0
        top3_given_detection = 0.0


    # ========================================================
    # END TO END
    #
    # No detection cuenta como fallo.
    # ========================================================

    end_to_end_top1 = float(
        df[
            "top1_correct"
        ].fillna(False).mean()
    )


    end_to_end_top3 = float(
        df[
            "true_in_top3"
        ].fillna(False).mean()
    )


    # ========================================================
    # RESUMEN JSON
    # ========================================================

    summary = {

        "model":
            "V2_M2_MLP_world_norm",

        "representation":
            "world_norm",

        "classes_evaluated":
            GENUINE_CLASSES,

        "provisional_classes_not_evaluated":
            sorted(
                PROVISIONAL_CLASSES
            ),

        "attempts_per_class":
            ATTEMPTS_PER_CLASS,

        "attempts_total":
            total,

        "detected":
            detected,

        "not_detected":
            not_detected,

        "detection_rate":
            detection_rate,

        "top1_accuracy_given_detection":
            top1_given_detection,

        "top3_accuracy_given_detection":
            top3_given_detection,

        "end_to_end_top1_accuracy":
            end_to_end_top1,

        "end_to_end_top3_accuracy":
            end_to_end_top3,
    }


    summary_path = (
        run_dir
        / "summary.json"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # RESUMEN POR CLASE
    # ========================================================

    class_rows = []


    for true_class in GENUINE_CLASSES:

        class_df = df[
            df["true_class"]
            == true_class
        ]


        class_detected = class_df[
            class_df["detected"]
        ]


        n_total = len(
            class_df
        )


        n_detected = int(
            class_df["detected"].sum()
        )


        top1_correct = int(
            class_df[
                "top1_correct"
            ]
            .fillna(False)
            .sum()
        )


        top3_correct = int(
            class_df[
                "true_in_top3"
            ]
            .fillna(False)
            .sum()
        )


        # Prediccion dominante.
        if len(
            class_detected
        ) > 0:

            counts = (
                class_detected[
                    "predicted_class"
                ]
                .value_counts()
            )


            dominant_prediction = (
                counts.index[0]
            )


            dominant_prediction_count = int(
                counts.iloc[0]
            )

        else:

            dominant_prediction = ""
            dominant_prediction_count = 0


        class_rows.append(
            {
                "class":
                    true_class,

                "attempts":
                    n_total,

                "detected":
                    n_detected,

                "detection_rate":
                    (
                        n_detected / n_total
                        if n_total
                        else 0.0
                    ),

                "top1_correct":
                    top1_correct,

                "top3_correct":
                    top3_correct,

                "top1_end_to_end_accuracy":
                    (
                        top1_correct / n_total
                        if n_total
                        else 0.0
                    ),

                "top3_end_to_end_accuracy":
                    (
                        top3_correct / n_total
                        if n_total
                        else 0.0
                    ),

                "dominant_prediction":
                    dominant_prediction,

                "dominant_prediction_count":
                    dominant_prediction_count,
            }
        )


    by_class_df = pd.DataFrame(
        class_rows
    )


    by_class_path = (
        run_dir
        / "by_class.csv"
    )


    by_class_df.to_csv(
        by_class_path,
        index=False
    )


    # ========================================================
    # PARES DE CONFUSION
    # ========================================================

    errors = detected_df[
        detected_df[
            "top1_correct"
        ] == False
    ]


    if len(
        errors
    ) > 0:

        confusion_pairs = (
            errors
            .groupby(
                [
                    "true_class",
                    "predicted_class"
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    else:

        confusion_pairs = pd.DataFrame(
            columns=[
                "true_class",
                "predicted_class",
                "count",
            ]
        )


    confusion_pairs_path = (
        run_dir
        / "confusion_pairs.csv"
    )


    confusion_pairs.to_csv(
        confusion_pairs_path,
        index=False
    )


    # ========================================================
    # MATRIZ DE CONFUSION
    #
    # 23 filas reales
    # 26 columnas posibles.
    # ========================================================

    detected_true = (
        detected_df[
            "true_class"
        ]
    )


    detected_pred = (
        detected_df[
            "predicted_class"
        ]
    )


    cm_df = pd.crosstab(

        pd.Categorical(
            detected_true,
            categories=GENUINE_CLASSES
        ),

        pd.Categorical(
            detected_pred,
            categories=ALL_CLASSES
        ),

        dropna=False
    )


    cm_df.index.name = (
        "true_class"
    )

    cm_df.columns.name = (
        "predicted_class"
    )


    cm_csv_path = (
        run_dir
        / "confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_csv_path
    )


    # ========================================================
    # MATRIZ PNG AZUL + NUMEROS
    # ========================================================

    cm = cm_df.to_numpy()


    fig, ax = plt.subplots(
        figsize=(
            18,
            13
        )
    )


    image = ax.imshow(
        cm,
        cmap="Blues",
        aspect="auto"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )


    colorbar.set_label(
        "Numero de muestras"
    )


    ax.set_xticks(
        np.arange(
            len(
                ALL_CLASSES
            )
        )
    )


    ax.set_yticks(
        np.arange(
            len(
                GENUINE_CLASSES
            )
        )
    )


    ax.set_xticklabels(
        ALL_CLASSES
    )


    ax.set_yticklabels(
        GENUINE_CLASSES
    )


    ax.set_xlabel(
        "Clase predicha"
    )


    ax.set_ylabel(
        "Clase real"
    )


    ax.set_title(
        "LSM Webcam - Matriz de confusion"
    )


    # --------------------------------------------------------
    # LINEAS DE CELDAS
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            len(
                ALL_CLASSES
            ),
            1
        ),
        minor=True
    )


    ax.set_yticks(
        np.arange(
            -0.5,
            len(
                GENUINE_CLASSES
            ),
            1
        ),
        minor=True
    )


    ax.grid(
        which="minor",
        color="gray",
        linewidth=0.7,
        alpha=0.7
    )


    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )


    ax.set_axisbelow(
        False
    )


    # --------------------------------------------------------
    # NUMEROS
    # --------------------------------------------------------

    max_value = (
        cm.max()
        if cm.size
        else 0
    )


    threshold = (
        max_value * 0.50
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = int(
                cm[i, j]
            )


            if value == 0:

                continue


            color = (
                "white"
                if value > threshold
                else "black"
            )


            ax.text(
                j,
                i,
                str(value),

                ha="center",
                va="center",

                color=color,
                fontsize=9,
            )


    plt.tight_layout()


    cm_png_path = (
        run_dir
        / "confusion_matrix.png"
    )


    plt.savefig(
        cm_png_path,
        dpi=250,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    # ========================================================
    # IMPRIMIR RESULTADOS
    # ========================================================

    print()
    print("=" * 75)

    print(
        "=== RESULTADO LSM WEBCAM ==="
    )

    print("=" * 75)


    print(
        "\nClases:",
        ", ".join(
            GENUINE_CLASSES
        )
    )


    print(
        "\nIntentos totales:",
        total
    )


    print(
        "Detectadas:",
        detected
    )


    print(
        "No detectadas:",
        not_detected
    )


    print(
        f"Tasa deteccion: "
        f"{detection_rate * 100:.2f}%"
    )


    print(
        "\nAccuracy Top-1 dado deteccion:",
        f"{top1_given_detection * 100:.2f}%"
    )


    print(
        "Accuracy Top-3 dado deteccion:",
        f"{top3_given_detection * 100:.2f}%"
    )


    print(
        "\nAccuracy end-to-end Top-1:",
        f"{end_to_end_top1 * 100:.2f}%"
    )


    print(
        "Accuracy end-to-end Top-3:",
        f"{end_to_end_top3 * 100:.2f}%"
    )


    print()
    print(
        "=== POR CLASE ==="
    )


    print(
        by_class_df[
            [
                "class",
                "detected",
                "top1_correct",
                "top3_correct",
                "dominant_prediction",
                "dominant_prediction_count",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "=== CONFUSIONES MAS FRECUENTES ==="
    )


    if len(
        confusion_pairs
    ) > 0:

        print(
            confusion_pairs
            .head(30)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "Sin errores Top-1."
        )


    print()
    print(
        "=== ARCHIVOS GENERADOS ==="
    )


    print(
        results_path
    )

    print(
        summary_path
    )

    print(
        by_class_path
    )

    print(
        confusion_pairs_path
    )

    print(
        cm_csv_path
    )

    print(
        cm_png_path
    )

    print(
        run_dir
        / "captures"
    )


    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)

    print(
        "=== SIGN2SIGN - EVALUACION CONTROLADA LSM ==="
    )

    print("=" * 75)


    # ========================================================
    # VALIDAR
    # ========================================================

    validate_files()


    # ========================================================
    # CREAR RUN
    # ========================================================

    run_dir, captures_dir = (
        create_run_directory()
    )


    print(
        "\nSalida:"
    )

    print(
        run_dir
    )


    # ========================================================
    # MODELO
    # ========================================================

    print(
        "\nCargando scaler..."
    )


    scaler = joblib.load(
        SCALER_PATH
    )


    print(
        "Cargando modelo..."
    )


    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


    if model.output_shape[-1] != 26:

        raise RuntimeError(
            f"Modelo tiene "
            f"{model.output_shape[-1]} salidas."
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


    # ========================================================
    # CONFIGURACION DE PRUEBA
    # ========================================================

    total_attempts = (
        len(
            GENUINE_CLASSES
        )
        *
        ATTEMPTS_PER_CLASS
    )


    rows = []


    aborted = False


    # Mensaje visual breve tras cada captura.
    status_message = None

    status_until = 0.0


    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    try:

        for class_index, true_class in enumerate(
            GENUINE_CLASSES
        ):

            for attempt in range(
                1,
                ATTEMPTS_PER_CLASS + 1
            ):

                captured = False


                while not captured:

                    ok, frame = cap.read()


                    if not ok:

                        continue


                    # -----------------------------------------
                    # Frame crudo.
                    #
                    # NO FLIP.
                    # -----------------------------------------

                    display_frame = (
                        frame.copy()
                    )


                    total_attempt_number = (
                        class_index
                        * ATTEMPTS_PER_CLASS
                        + attempt
                    )


                    current_status = None


                    if (
                        status_message
                        and
                        time.time()
                        < status_until
                    ):

                        current_status = (
                            status_message
                        )


                    draw_ui(
                        display_frame,

                        target_class=true_class,

                        attempt_number=attempt,

                        total_attempt=total_attempt_number,

                        total_attempts=total_attempts,

                        status_message=current_status,
                    )


                    cv2.imshow(
                        "Sign2Sign - Evaluacion LSM",
                        display_frame
                    )


                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )


                    # =========================================
                    # CANCELAR
                    # =========================================

                    if key in (
                        ord("q"),
                        ord("Q"),
                        27,
                    ):

                        aborted = True
                        break


                    # =========================================
                    # ENTER
                    # =========================================

                    if key not in (
                        13,
                        10,
                    ):

                        continue


                    # =========================================
                    # GUARDAR IMAGEN EXACTA
                    # ANTES de dibujar overlays.
                    # =========================================

                    capture_name = (
                        f"{true_class}_"
                        f"{attempt:02d}.jpg"
                    )


                    capture_path = (
                        captures_dir
                        / capture_name
                    )


                    cv2.imwrite(
                        str(
                            capture_path
                        ),
                        frame
                    )


                    # =========================================
                    # INFERENCIA
                    # =========================================

                    print(
                        f"\nCaptura "
                        f"{total_attempt_number}"
                        f"/{total_attempts}"
                        f" | Clase {true_class}"
                        f" | Intento {attempt}"
                    )


                    try:

                        result = evaluate_frame(
                            frame,
                            landmarker,
                            scaler,
                            model
                        )


                    except Exception as exc:

                        print(
                            "Error de inferencia:",
                            exc
                        )


                        result = {
                            "detected": False,
                            "variant": "error",
                        }


                    # =========================================
                    # NO DETECTION
                    # =========================================

                    if not result[
                        "detected"
                    ]:

                        rows.append(
                            {
                                "attempt_global":
                                    total_attempt_number,

                                "true_class":
                                    true_class,

                                "attempt_in_class":
                                    attempt,

                                "detected":
                                    False,

                                "variant":
                                    result.get(
                                        "variant",
                                        "unknown"
                                    ),

                                "predicted_class":
                                    None,

                                "top1_prob":
                                    None,

                                "top2_class":
                                    None,

                                "top2_prob":
                                    None,

                                "top3_class":
                                    None,

                                "top3_prob":
                                    None,

                                "true_rank":
                                    None,

                                "top1_correct":
                                    False,

                                "true_in_top3":
                                    False,

                                "provisional_prediction":
                                    False,

                                "capture_path":
                                    str(
                                        capture_path
                                    ),

                                "timestamp":
                                    datetime.now()
                                    .isoformat(),
                            }
                        )


                        print(
                            "No detection."
                        )


                    # =========================================
                    # DETECTADA
                    # =========================================

                    else:

                        top3 = result[
                            "top3"
                        ]


                        ranking = result[
                            "ranking"
                        ]


                        predicted_class = (
                            top3[0][
                                "class"
                            ]
                        )


                        true_index = (
                            CLASS_TO_INDEX[
                                true_class
                            ]
                        )


                        true_position = np.where(
                            ranking
                            == true_index
                        )[0]


                        if len(
                            true_position
                        ):

                            true_rank = int(
                                true_position[0]
                                + 1
                            )

                        else:

                            true_rank = None


                        top1_correct = (
                            predicted_class
                            == true_class
                        )


                        true_in_top3 = (
                            true_rank is not None
                            and
                            true_rank <= 3
                        )


                        rows.append(
                            {
                                "attempt_global":
                                    total_attempt_number,

                                "true_class":
                                    true_class,

                                "attempt_in_class":
                                    attempt,

                                "detected":
                                    True,

                                "variant":
                                    result[
                                        "variant"
                                    ],

                                "predicted_class":
                                    predicted_class,

                                "top1_prob":
                                    top3[0][
                                        "probability"
                                    ],

                                "top2_class":
                                    top3[1][
                                        "class"
                                    ],

                                "top2_prob":
                                    top3[1][
                                        "probability"
                                    ],

                                "top3_class":
                                    top3[2][
                                        "class"
                                    ],

                                "top3_prob":
                                    top3[2][
                                        "probability"
                                    ],

                                "true_rank":
                                    true_rank,

                                "top1_correct":
                                    top1_correct,

                                "true_in_top3":
                                    true_in_top3,

                                "provisional_prediction":
                                    (
                                        predicted_class
                                        in
                                        PROVISIONAL_CLASSES
                                    ),

                                "capture_path":
                                    str(
                                        capture_path
                                    ),

                                "timestamp":
                                    datetime.now()
                                    .isoformat(),
                            }
                        )


                        # -------------------------------------
                        # NO MOSTRAR PREDICCION.
                        # -------------------------------------

                        print(
                            "Captura registrada."
                        )


                    # =========================================
                    # CONFIRMACION VISUAL
                    # SIN MOSTRAR RESULTADO
                    # =========================================

                    status_message = (
                        "Captura registrada"
                    )


                    status_until = (
                        time.time()
                        + 0.70
                    )


                    captured = True


                if aborted:

                    break


            if aborted:

                break


    finally:

        cap.release()

        cv2.destroyAllWindows()


        try:

            landmarker.close()

        except Exception:

            pass


    # ========================================================
    # ABORTADO
    # ========================================================

    if aborted:

        print()
        print(
            "⚠ Evaluacion cancelada."
        )


        if rows:

            partial_path = (
                run_dir
                / "webcam_results_PARTIAL.csv"
            )


            pd.DataFrame(
                rows
            ).to_csv(
                partial_path,
                index=False
            )


            print(
                "Resultados parciales:"
            )

            print(
                partial_path
            )


        return


    # ========================================================
    # GENERAR RESULTADOS
    # ========================================================

    generate_results(
        rows,
        run_dir
    )


    print()
    print(
        "✅ Evaluacion LSM terminada."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    main()
# ============================================================
# Sign2Sign
# Evaluacion focalizada LSM Canonical V2 por webcam
#
# Clases:
#   J, K, P, R, U
#
# Objetivos:
#   J -> verificar convencion estatica / referencia
#   P/K -> estudiar ambiguedad entre ambas
#   R/U -> estudiar confusion geometrica entre ambas
#
# Protocolo:
#   10 intentos por clase
#   50 intentos totales
#   predicciones ocultas durante la prueba
#
# IMPORTANTE:
#   Reutiliza el pipeline ya validado de:
#       evaluate_lsm_webcam.py
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
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================
# REUTILIZAR EVALUADOR LSM YA VALIDADO
# ============================================================

from evaluate_lsm_webcam import (
    MODEL_PATH,
    SCALER_PATH,
    MEDIAPIPE_MODEL_PATH,
    CAMERA_INDEX,
    ALL_CLASSES,
    PROVISIONAL_CLASSES,
    CLASS_TO_INDEX,
    validate_files,
    evaluate_frame,
)

from src.landmarks.pipeline import (
    create_hand_landmarker,
)


# ============================================================
# CONFIGURACION DE ESTA PRUEBA
# ============================================================

FOCUSED_CLASSES = [
    "J",
    "K",
    "P",
    "R",
    "U",
]

ATTEMPTS_PER_CLASS = 10

OUTPUT_ROOT = Path(
    r"D:\Sign2SignData\inventory\webcam_eval\lsm_focused"
)


# ============================================================
# COLORES
# ============================================================

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
BLACK = (0, 0, 0)


# ============================================================
# CREAR RUN
# ============================================================

def create_run_directory():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        OUTPUT_ROOT
        / f"focused_run_{timestamp}"
    )

    captures_dir = (
        run_dir
        / "captures"
    )

    captures_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    return run_dir, captures_dir


# ============================================================
# UI
# ============================================================

def draw_ui(
    frame,
    target_class,
    attempt_number,
    total_attempt_number,
    total_attempts,
    status_message=None,
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (650, 250),
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

    cv2.putText(
        frame,
        "Sign2Sign - LSM Focalizado",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Clase objetivo:",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        target_class,
        (230, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        CYAN,
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        (
            f"Intento: "
            f"{attempt_number}/{ATTEMPTS_PER_CLASS}"
        ),
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        (
            f"Progreso: "
            f"{total_attempt_number}/{total_attempts}"
        ),
        (30, 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "ENTER: capturar | Q / ESC: cancelar",
        (30, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        YELLOW,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Prediccion oculta",
        (390, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1,
        cv2.LINE_AA
    )

    if status_message:

        cv2.putText(
            frame,
            status_message,
            (390, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GREEN,
            2,
            cv2.LINE_AA
        )


# ============================================================
# GENERAR RESULTADOS
# ============================================================

def generate_results(
    rows,
    run_dir
):

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # CSV PRINCIPAL
    # --------------------------------------------------------

    results_path = (
        run_dir
        / "focused_webcam_results.csv"
    )

    df.to_csv(
        results_path,
        index=False
    )


    # --------------------------------------------------------
    # METRICAS GLOBALES
    # --------------------------------------------------------

    total = len(df)

    detected = int(
        df["detected"].sum()
    )

    not_detected = (
        total - detected
    )

    detection_rate = (
        detected / total
        if total
        else 0.0
    )

    detected_df = df[
        df["detected"]
    ].copy()


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


    end_to_end_top1 = float(
        df[
            "top1_correct"
        ]
        .fillna(False)
        .mean()
    )

    end_to_end_top3 = float(
        df[
            "true_in_top3"
        ]
        .fillna(False)
        .mean()
    )


    summary = {

        "model":
            "V2_M2_MLP_world_norm",

        "test_type":
            "focused_webcam",

        "classes":
            FOCUSED_CLASSES,

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
        / "focused_summary.json"
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


    # --------------------------------------------------------
    # POR CLASE
    # --------------------------------------------------------

    class_rows = []


    for true_class in FOCUSED_CLASSES:

        class_df = df[
            df["true_class"]
            == true_class
        ]

        detected_class_df = class_df[
            class_df["detected"]
        ]

        attempts = len(
            class_df
        )

        detected_count = int(
            class_df[
                "detected"
            ].sum()
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


        if len(
            detected_class_df
        ) > 0:

            prediction_counts = (
                detected_class_df[
                    "predicted_class"
                ]
                .value_counts()
            )

            dominant_prediction = (
                prediction_counts
                .index[0]
            )

            dominant_count = int(
                prediction_counts
                .iloc[0]
            )

        else:

            dominant_prediction = ""
            dominant_count = 0


        class_rows.append(
            {
                "class":
                    true_class,

                "attempts":
                    attempts,

                "detected":
                    detected_count,

                "top1_correct":
                    top1_correct,

                "top3_correct":
                    top3_correct,

                "top1_end_to_end_accuracy":
                    (
                        top1_correct
                        / attempts
                        if attempts
                        else 0
                    ),

                "top3_end_to_end_accuracy":
                    (
                        top3_correct
                        / attempts
                        if attempts
                        else 0
                    ),

                "dominant_prediction":
                    dominant_prediction,

                "dominant_prediction_count":
                    dominant_count,
            }
        )


    by_class_df = pd.DataFrame(
        class_rows
    )


    by_class_path = (
        run_dir
        / "focused_by_class.csv"
    )

    by_class_df.to_csv(
        by_class_path,
        index=False
    )


    # --------------------------------------------------------
    # CONFUSIONES
    # --------------------------------------------------------

    errors = detected_df[
        detected_df[
            "top1_correct"
        ] == False
    ]


    if len(errors):

        confusion_pairs = (
            errors
            .groupby(
                [
                    "true_class",
                    "predicted_class",
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


    confusion_path = (
        run_dir
        / "focused_confusion_pairs.csv"
    )

    confusion_pairs.to_csv(
        confusion_path,
        index=False
    )


    # --------------------------------------------------------
    # MATRIZ
    #
    # Filas = las cinco clases focalizadas.
    # Columnas = las 26 posibles salidas del modelo.
    # --------------------------------------------------------

    cm_df = pd.crosstab(

        pd.Categorical(
            detected_df[
                "true_class"
            ],
            categories=FOCUSED_CLASSES
        ),

        pd.Categorical(
            detected_df[
                "predicted_class"
            ],
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
        / "focused_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_csv_path
    )


    # --------------------------------------------------------
    # MATRIZ PNG
    # --------------------------------------------------------

    cm = cm_df.to_numpy()


    fig, ax = plt.subplots(
        figsize=(18, 6)
    )


    image = ax.imshow(
        cm,
        cmap="Blues",
        aspect="auto"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02
    )

    colorbar.set_label(
        "Numero de muestras"
    )


    ax.set_xticks(
        np.arange(
            len(ALL_CLASSES)
        )
    )

    ax.set_yticks(
        np.arange(
            len(FOCUSED_CLASSES)
        )
    )


    ax.set_xticklabels(
        ALL_CLASSES
    )

    ax.set_yticklabels(
        FOCUSED_CLASSES
    )


    ax.set_xlabel(
        "Clase predicha"
    )

    ax.set_ylabel(
        "Clase real"
    )


    ax.set_title(
        "LSM Webcam - Evaluacion focalizada"
    )


    # Lineas de celdas.
    ax.set_xticks(
        np.arange(
            -0.5,
            len(ALL_CLASSES),
            1
        ),
        minor=True
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(FOCUSED_CLASSES),
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


    max_value = (
        cm.max()
        if cm.size
        else 0
    )

    threshold = (
        max_value * 0.5
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


            text_color = (
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
                color=text_color,
                fontsize=10
            )


    plt.tight_layout()


    cm_png_path = (
        run_dir
        / "focused_confusion_matrix.png"
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
    # TERMINAL
    # ========================================================

    print()
    print("=" * 75)

    print(
        "=== RESULTADO LSM WEBCAM - FOCALIZADO ==="
    )

    print("=" * 75)


    print(
        "\nClases:",
        ", ".join(
            FOCUSED_CLASSES
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
        ]
        .to_string(
            index=False
        )
    )


    print()
    print(
        "=== CONFUSIONES ==="
    )


    if len(
        confusion_pairs
    ):

        print(
            confusion_pairs
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
        "=== ARCHIVOS ==="
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
        confusion_path
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)

    print(
        "=== SIGN2SIGN - LSM FOCALIZADO ==="
    )

    print("=" * 75)


    validate_files()


    run_dir, captures_dir = (
        create_run_directory()
    )


    print(
        "\nClases:"
    )

    print(
        ", ".join(
            FOCUSED_CLASSES
        )
    )


    print(
        "\nIntentos por clase:",
        ATTEMPTS_PER_CLASS
    )


    print(
        "Intentos totales:",
        (
            len(FOCUSED_CLASSES)
            * ATTEMPTS_PER_CLASS
        )
    )


    print(
        "\nSalida:"
    )

    print(
        run_dir
    )


    # --------------------------------------------------------
    # CARGAR
    # --------------------------------------------------------

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


    print(
        "Creando Hand Landmarker..."
    )

    landmarker = create_hand_landmarker(
        str(
            MEDIAPIPE_MODEL_PATH
        )
    )


    # --------------------------------------------------------
    # CAMARA
    # --------------------------------------------------------

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


    total_attempts = (
        len(FOCUSED_CLASSES)
        * ATTEMPTS_PER_CLASS
    )


    rows = []

    aborted = False

    status_message = None
    status_until = 0.0


    try:

        for class_index, true_class in enumerate(
            FOCUSED_CLASSES
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

                        total_attempt_number=
                            total_attempt_number,

                        total_attempts=
                            total_attempts,

                        status_message=
                            current_status,
                    )


                    cv2.imshow(
                        "Sign2Sign - LSM Focalizado",
                        display_frame
                    )


                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )


                    # -----------------------------------------
                    # SALIR
                    # -----------------------------------------

                    if key in (
                        ord("q"),
                        ord("Q"),
                        27,
                    ):

                        aborted = True
                        break


                    # -----------------------------------------
                    # SOLO ENTER
                    # -----------------------------------------

                    if key not in (
                        13,
                        10,
                    ):

                        continue


                    # -----------------------------------------
                    # GUARDAR FRAME ORIGINAL
                    # -----------------------------------------

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


                    print(
                        f"\nCaptura "
                        f"{total_attempt_number}"
                        f"/{total_attempts}"
                        f" | {true_class}"
                        f" | intento {attempt}"
                    )


                    # -----------------------------------------
                    # EVALUAR
                    # -----------------------------------------

                    try:

                        result = evaluate_frame(
                            frame,
                            landmarker,
                            scaler,
                            model
                        )

                    except Exception as exc:

                        print(
                            "Error:",
                            exc
                        )

                        result = {
                            "detected": False,
                            "variant": "error",
                        }


                    # -----------------------------------------
                    # NO DETECCION
                    # -----------------------------------------

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


                    # -----------------------------------------
                    # DETECTADA
                    # -----------------------------------------

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


                        position = np.where(
                            ranking
                            == true_index
                        )[0]


                        if len(position):

                            true_rank = int(
                                position[0]
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

                                "capture_path":
                                    str(
                                        capture_path
                                    ),

                                "timestamp":
                                    datetime.now()
                                    .isoformat(),
                            }
                        )


                        # No revelamos la prediccion.
                        print(
                            "Captura registrada."
                        )


                    status_message = (
                        "Captura registrada"
                    )

                    status_until = (
                        time.time()
                        + 0.6
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
    # CANCELADA
    # ========================================================

    if aborted:

        print(
            "\nEvaluacion cancelada."
        )

        if rows:

            partial_path = (
                run_dir
                / "focused_results_PARTIAL.csv"
            )

            pd.DataFrame(
                rows
            ).to_csv(
                partial_path,
                index=False
            )

            print(
                partial_path
            )

        return


    # ========================================================
    # RESULTADOS
    # ========================================================

    generate_results(
        rows,
        run_dir
    )


    print()
    print(
        "✅ Prueba focalizada terminada."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    main()
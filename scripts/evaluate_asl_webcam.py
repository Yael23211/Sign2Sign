import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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

# Las capturas NO se guardarán dentro del repo.
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Sign2SignData\inventory\webcam_eval\asl"
)


# ==========================================================
# LANDMARKS -> IMG_NORM
# ==========================================================

def landmarks_to_feature_vector(
    result,
    processed_image
):
    if result is None:
        return None

    if not hasattr(result, "hand_landmarks"):
        raise RuntimeError(
            "Resultado MediaPipe inválido."
        )

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    height, width = processed_image.shape[:2]

    signature = inspect.signature(
        image_landmarks_to_arrays
    )

    param_count = len(
        signature.parameters
    )

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

    if not isinstance(
        output,
        (tuple, list)
    ):

        raise RuntimeError(
            "image_landmarks_to_arrays() "
            "no devolvió tuple/list."
        )

    if len(output) < 2:

        raise RuntimeError(
            "No se obtuvo img_norm."
        )

    normalized = output[1]

    features = np.asarray(
        normalized,
        dtype=np.float32
    ).reshape(-1)

    if features.shape != (63,):

        raise RuntimeError(
            "Vector img_norm inválido. "
            f"Esperado (63,), "
            f"obtenido {features.shape}"
        )

    if not np.isfinite(
        features
    ).all():

        raise RuntimeError(
            "img_norm contiene NaN "
            "o infinitos."
        )

    return features


# ==========================================================
# CASCADE
# ==========================================================

def unpack_detection(detection):

    if not isinstance(
        detection,
        dict
    ):

        raise RuntimeError(
            "detect_with_cascade() "
            "no devolvió dict."
        )

    if "ok" not in detection:

        raise RuntimeError(
            "Salida del cascade sin clave 'ok'. "
            f"Claves: {list(detection.keys())}"
        )

    detected = bool(
        detection["ok"]
    )

    if not detected:

        return (
            False,
            None,
            None,
            None
        )

    for key in (
        "variant",
        "processed",
        "result"
    ):

        if key not in detection:

            raise RuntimeError(
                f"Falta '{key}' en la "
                "salida del cascade."
            )

    return (
        True,
        detection["variant"],
        detection["processed"],
        detection["result"]
    )


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predict_top3(
    model,
    scaler,
    features
):

    X = features.reshape(
        1,
        -1
    )

    # SOLO transform.
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
                float(
                    probabilities[idx]
                )
            )
        )

    return top3


# ==========================================================
# INTERFAZ
# ==========================================================

def draw_interface(
    frame,
    target_class,
    attempt,
    attempts_per_class,
    completed,
    total,
    message
):

    output = frame.copy()

    width = output.shape[1]

    cv2.rectangle(
        output,
        (0, 0),
        (width, 175),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        "Sign2Sign - Evaluacion ASL Webcam",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        (
            f"Realiza la letra: "
            f"{target_class}"
        ),
        (15, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        (
            f"Intento: "
            f"{attempt}/{attempts_per_class}"
        ),
        (15, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        1
    )

    cv2.putText(
        output,
        (
            f"Progreso: "
            f"{completed}/{total}"
        ),
        (15, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1
    )

    cv2.putText(
        output,
        message,
        (15, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1
    )

    return output


# ==========================================================
# GUARDAR RESULTADOS PARCIALES
# ==========================================================

def save_results_csv(
    results,
    run_dir
):

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        run_dir / "webcam_results.csv",
        index=False
    )

    return df


# ==========================================================
# RESUMEN
# ==========================================================

def generate_summary(
    df,
    run_dir,
    attempts_per_class
):

    total = len(df)

    detected = int(
        df["hand_detected"].sum()
    )

    correct = int(
        df["correct"].sum()
    )

    top3_correct = int(
        df["true_in_top3"].sum()
    )

    detection_rate = (
        detected / total
        if total
        else 0.0
    )

    conditional_accuracy = (
        correct / detected
        if detected
        else 0.0
    )

    conditional_top3 = (
        top3_correct / detected
        if detected
        else 0.0
    )

    end_to_end_accuracy = (
        correct / total
        if total
        else 0.0
    )

    end_to_end_top3 = (
        top3_correct / total
        if total
        else 0.0
    )

    summary = {
        "total_attempts": total,
        "attempts_per_class":
            attempts_per_class,
        "detected": detected,
        "not_detected":
            total - detected,
        "detection_rate":
            detection_rate,
        "top1_correct":
            correct,
        "classifier_accuracy_given_detection":
            conditional_accuracy,
        "top3_correct":
            top3_correct,
        "top3_accuracy_given_detection":
            conditional_top3,
        "end_to_end_top1_accuracy":
            end_to_end_accuracy,
        "end_to_end_top3_accuracy":
            end_to_end_top3
    }

    with open(
        run_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ------------------------------------------------------
    # POR CLASE
    # ------------------------------------------------------

    rows = []

    for cls in CLASSES:

        part = df[
            df["true_class"] == cls
        ]

        if len(part) == 0:
            continue

        n = len(part)

        n_detected = int(
            part[
                "hand_detected"
            ].sum()
        )

        n_correct = int(
            part[
                "correct"
            ].sum()
        )

        n_top3 = int(
            part[
                "true_in_top3"
            ].sum()
        )

        rows.append({
            "class": cls,
            "attempts": n,
            "detected":
                n_detected,
            "detection_rate":
                n_detected / n,
            "top1_correct":
                n_correct,
            "top1_accuracy_given_detection":
                (
                    n_correct / n_detected
                    if n_detected
                    else 0.0
                ),
            "top3_correct":
                n_top3,
            "top3_accuracy_given_detection":
                (
                    n_top3 / n_detected
                    if n_detected
                    else 0.0
                ),
            "end_to_end_top1":
                n_correct / n,
            "end_to_end_top3":
                n_top3 / n
        })

    by_class = pd.DataFrame(
        rows
    )

    by_class.to_csv(
        run_dir / "by_class.csv",
        index=False
    )

    # ------------------------------------------------------
    # CONFUSION MATRIX:
    # solo capturas donde MediaPipe detectó.
    # ------------------------------------------------------

    detected_df = df[
        df["hand_detected"]
    ].copy()

    if len(detected_df) > 0:

        cm = confusion_matrix(
            detected_df["true_class"],
            detected_df["predicted_class"],
            labels=CLASSES
        )

        fig, ax = plt.subplots(
            figsize=(15, 15)
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASSES
        )

        disp.plot(
            ax=ax,
            colorbar=False,
            xticks_rotation=45
        )

        ax.set_title(
            "ASL Webcam - Matriz de confusion"
        )

        plt.tight_layout()

        plt.savefig(
            run_dir
            / "confusion_matrix.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    return (
        summary,
        by_class
    )


# ==========================================================
# MOSTRAR RESUMEN
# ==========================================================

def print_summary(
    summary,
    by_class
):

    print()
    print(
        "========================================"
    )

    print(
        "=== RESULTADO EVALUACIÓN ASL WEBCAM ==="
    )

    print(
        "========================================"
    )

    print()

    print(
        "Intentos totales :",
        summary["total_attempts"]
    )

    print(
        "Detectadas        :",
        summary["detected"]
    )

    print(
        "No detectadas     :",
        summary["not_detected"]
    )

    print(
        "Tasa detección    :",
        f"{summary['detection_rate'] * 100:.2f}%"
    )

    print()

    print(
        "Accuracy Top-1 "
        "dado detección :",
        (
            f"{summary['classifier_accuracy_given_detection'] * 100:.2f}%"
        )
    )

    print(
        "Accuracy Top-3 "
        "dado detección :",
        (
            f"{summary['top3_accuracy_given_detection'] * 100:.2f}%"
        )
    )

    print()

    print(
        "Accuracy end-to-end Top-1 :",
        (
            f"{summary['end_to_end_top1_accuracy'] * 100:.2f}%"
        )
    )

    print(
        "Accuracy end-to-end Top-3 :",
        (
            f"{summary['end_to_end_top3_accuracy'] * 100:.2f}%"
        )
    )

    print()
    print(
        "=== POR CLASE ==="
    )

    display_cols = [
        "class",
        "detected",
        "top1_correct",
        "top3_correct",
        "end_to_end_top1"
    ]

    display_df = by_class[
        display_cols
    ].copy()

    display_df[
        "end_to_end_top1"
    ] = (
        display_df[
            "end_to_end_top1"
        ]
        * 100
    ).round(2)

    print(
        display_df.to_string(
            index=False
        )
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

    output_root = Path(
        args.output_dir
    )

    for path, name in [
        (
            hand_model_path,
            "MediaPipe"
        ),
        (
            asl_model_path,
            "A3"
        ),
        (
            scaler_path,
            "Scaler"
        )
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"{name} no encontrado:\n"
                f"{path}"
            )

    # ------------------------------------------------------
    # DIRECTORIO DE ESTA SESIÓN
    # ------------------------------------------------------

    run_name = (
        "run_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_dir = (
        output_root
        / run_name
    )

    captures_dir = (
        run_dir
        / "captures"
    )

    captures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "=== EVALUACIÓN ASL WEBCAM ==="
    )

    print()
    print(
        "Resultados:"
    )

    print(
        f"  {run_dir}"
    )

    print()

    # ------------------------------------------------------
    # CARGAR A3
    # ------------------------------------------------------

    print(
        "Cargando A3..."
    )

    model = tf.keras.models.load_model(
        asl_model_path,
        compile=False
    )

    scaler = joblib.load(
        scaler_path
    )

    if model.input_shape[-1] != 63:

        raise RuntimeError(
            "A3 no espera 63 features."
        )

    if model.output_shape[-1] != 26:

        raise RuntimeError(
            "A3 no tiene 26 salidas."
        )

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None
    )

    if (
        scaler_features is not None
        and scaler_features != 63
    ):

        raise RuntimeError(
            "Scaler incompatible."
        )

    print(
        "✅ A3 y scaler cargados."
    )

    # ------------------------------------------------------
    # MEDIAPIPE
    # ------------------------------------------------------

    landmarker = create_hand_landmarker(
        str(hand_model_path)
    )

    print(
        "✅ MediaPipe cargado."
    )

    # ------------------------------------------------------
    # WEBCAM
    # ------------------------------------------------------

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():

        landmarker.close()

        raise RuntimeError(
            "No se pudo abrir webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    attempts_per_class = (
        args.attempts
    )

    total_attempts = (
        len(CLASSES)
        * attempts_per_class
    )

    results = []

    current_class_index = 0
    current_attempt = 1

    message = (
        "Haz la sena y presiona ENTER | ESC: terminar"
    )

    print()
    print(
        f"Letras: {len(CLASSES)}"
    )

    print(
        "Intentos por letra:",
        attempts_per_class
    )

    print(
        "Intentos totales:",
        total_attempts
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "- Haz cada seña de forma natural."
    )

    print(
        "- No intentes ajustarte a la "
        "predicción del modelo."
    )

    print(
        "- Las predicciones permanecerán "
        "ocultas hasta el final."
    )

    print(
        "- J y Z se evalúan con la "
        "configuración estática usada "
        "por el proyecto."
    )

    try:

        while (
            current_class_index
            < len(CLASSES)
        ):

            target_class = (
                CLASSES[
                    current_class_index
                ]
            )

            ok, frame = cap.read()

            if not ok:
                break

            display = draw_interface(
                frame,
                target_class,
                current_attempt,
                attempts_per_class,
                len(results),
                total_attempts,
                message
            )

            cv2.imshow(
                "Sign2Sign - Evaluacion ASL Webcam",
                display
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ESC
            if key == 27:
                print(
                    "\nEvaluación interrumpida."
                )
                break

            # ENTER
            if key not in (
                10,
                13
            ):
                continue

            captured = frame.copy()

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )

            filename = (
                f"{target_class}_"
                f"{current_attempt:02d}_"
                f"{timestamp}.jpg"
            )

            image_path = (
                captures_dir
                / filename
            )

            # Guardamos el frame EXACTO evaluado.
            cv2.imwrite(
                str(image_path),
                captured
            )

            detection = (
                detect_with_cascade(
                    landmarker,
                    captured
                )
            )

            (
                detected,
                variant,
                processed,
                result
            ) = unpack_detection(
                detection
            )

            row = {
                "timestamp":
                    timestamp,
                "true_class":
                    target_class,
                "attempt":
                    current_attempt,
                "hand_detected":
                    detected,
                "detection_method":
                    (
                        variant
                        if detected
                        else None
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
                "true_in_top3":
                    False,
                "correct":
                    False,
                "image_path":
                    str(image_path)
            }

            if detected:

                features = (
                    landmarks_to_feature_vector(
                        result,
                        processed
                    )
                )

                if features is not None:

                    top3 = predict_top3(
                        model,
                        scaler,
                        features
                    )

                    (
                        top1_class,
                        top1_prob
                    ) = top3[0]

                    (
                        top2_class,
                        top2_prob
                    ) = top3[1]

                    (
                        top3_class,
                        top3_prob
                    ) = top3[2]

                    top3_classes = [
                        top1_class,
                        top2_class,
                        top3_class
                    ]

                    row.update({
                        "predicted_class":
                            top1_class,
                        "top1_prob":
                            top1_prob,
                        "top2_class":
                            top2_class,
                        "top2_prob":
                            top2_prob,
                        "top3_class":
                            top3_class,
                        "top3_prob":
                            top3_prob,
                        "true_in_top3":
                            (
                                target_class
                                in top3_classes
                            ),
                        "correct":
                            (
                                top1_class
                                == target_class
                            )
                    })

            results.append(
                row
            )

            # Guardado inmediato:
            # si algo falla, no perdemos lo anterior.
            save_results_csv(
                results,
                run_dir
            )

            print(
                f"[{len(results):03d}/"
                f"{total_attempts}] "
                f"{target_class} "
                f"{current_attempt}/"
                f"{attempts_per_class} "
                f"| {'DETECTADA' if detected else 'NO DETECTADA'}"
            )

            # IMPORTANTE:
            # deliberadamente NO imprimimos
            # la predicción.

            current_attempt += 1

            if (
                current_attempt
                > attempts_per_class
            ):

                current_attempt = 1

                current_class_index += 1

    finally:

        cap.release()

        landmarker.close()

        cv2.destroyAllWindows()

    # ------------------------------------------------------
    # RESULTADOS
    # ------------------------------------------------------

    if len(results) == 0:

        print(
            "No se registraron intentos."
        )

        return

    df = save_results_csv(
        results,
        run_dir
    )

    (
        summary,
        by_class
    ) = generate_summary(
        df,
        run_dir,
        attempts_per_class
    )

    print_summary(
        summary,
        by_class
    )

    print()
    print(
        "Archivos generados:"
    )

    print(
        f"  {run_dir / 'webcam_results.csv'}"
    )

    print(
        f"  {run_dir / 'summary.json'}"
    )

    print(
        f"  {run_dir / 'by_class.csv'}"
    )

    print(
        f"  {run_dir / 'confusion_matrix.png'}"
    )

    print(
        f"  {captures_dir}"
    )

    print()
    print(
        "✅ Evaluación terminada."
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=5
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

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_ROOT
        )
    )

    main(
        parser.parse_args()
    )
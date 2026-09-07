import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors


# ==========================================================
# ROOT
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==========================================================
# PIPELINE REAL
# ==========================================================

from src.landmarks.pipeline import (
    create_hand_landmarker,
    detect_with_cascade,
    image_landmarks_to_arrays,
)


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

ALL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Solo las clases cuyo problema se mantuvo de forma
# sistemática después de la prueba focalizada.
FOCUS_CLASSES = list("OPRTX")


DEFAULT_ASL_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks\asl_landmarks_final.csv"
)

DEFAULT_FOCUSED_ROOT = Path(
    r"D:\Sign2SignData\inventory\webcam_eval\asl_focused"
)

DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Sign2SignData\inventory\webcam_eval\asl_domain_diagnosis"
)

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
# UTILIDADES
# ==========================================================

def find_latest_focused_results(root: Path):

    if not root.exists():
        raise FileNotFoundError(
            f"No existe:\n{root}"
        )

    candidates = sorted(
        root.glob(
            "focused_run_*/focused_webcam_results.csv"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No se encontró ningún "
            "focused_webcam_results.csv en:\n"
            f"{root}"
        )

    return candidates[0]


def find_img_norm_columns(df):

    # Mantener el ORDEN ORIGINAL del CSV.
    # No ordenar alfabéticamente.
    columns = [
        c
        for c in df.columns
        if str(c).startswith("img_norm_")
    ]

    if len(columns) != 63:
        raise RuntimeError(
            "Se esperaban exactamente 63 columnas "
            f"img_norm y se encontraron {len(columns)}."
        )

    return columns


# ==========================================================
# CASCADE
# ==========================================================

def unpack_detection(detection):

    if not isinstance(
        detection,
        dict
    ):
        raise RuntimeError(
            "detect_with_cascade() no devolvió dict."
        )

    if "ok" not in detection:
        raise RuntimeError(
            "La salida de detect_with_cascade() "
            "no contiene la clave 'ok'."
        )

    if not bool(
        detection["ok"]
    ):
        return (
            False,
            None,
            None,
            None,
        )

    for key in (
        "variant",
        "processed",
        "result",
    ):
        if key not in detection:
            raise RuntimeError(
                f"Falta '{key}' en la salida "
                "del cascade."
            )

    return (
        True,
        detection["variant"],
        detection["processed"],
        detection["result"],
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

    if not hasattr(
        result,
        "hand_landmarks"
    ):
        raise RuntimeError(
            "Resultado MediaPipe inválido."
        )

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    height, width = (
        processed_image.shape[:2]
    )

    signature = inspect.signature(
        image_landmarks_to_arrays
    )

    param_count = len(
        signature.parameters
    )

    if param_count == 1:

        output = (
            image_landmarks_to_arrays(
                landmarks
            )
        )

    elif param_count >= 3:

        output = (
            image_landmarks_to_arrays(
                landmarks,
                width,
                height
            )
        )

    else:

        raise RuntimeError(
            "Firma inesperada de "
            "image_landmarks_to_arrays(): "
            f"{signature}"
        )

    if (
        not isinstance(
            output,
            (tuple, list)
        )
        or len(output) < 2
    ):
        raise RuntimeError(
            "No se pudo obtener img_norm."
        )

    normalized = output[1]

    features = np.asarray(
        normalized,
        dtype=np.float32
    ).reshape(-1)

    if features.shape != (63,):
        raise RuntimeError(
            "Vector img_norm inválido. "
            f"Shape: {features.shape}"
        )

    if not np.isfinite(
        features
    ).all():
        raise RuntimeError(
            "img_norm contiene NaN o infinitos."
        )

    return features


# ==========================================================
# A3
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

    X_scaled = scaler.transform(
        X
    ).astype(
        np.float32
    )

    probabilities = model.predict(
        X_scaled,
        verbose=0
    )[0]

    indices = np.argsort(
        probabilities
    )[-3:][::-1]

    return [
        (
            ALL_CLASSES[int(i)],
            float(
                probabilities[int(i)]
            )
        )
        for i in indices
    ]


# ==========================================================
# CENTROIDES
# ==========================================================

def build_centroids(
    X_train_scaled,
    y_train
):

    centroids = {}

    for cls in ALL_CLASSES:

        mask = (
            y_train == cls
        )

        if not np.any(mask):
            continue

        centroids[cls] = (
            X_train_scaled[
                mask
            ].mean(
                axis=0
            )
        )

    return centroids


def centroid_analysis(
    query_scaled,
    centroids,
    true_class
):

    distances = []

    for cls, centroid in (
        centroids.items()
    ):

        distance = float(
            np.linalg.norm(
                query_scaled - centroid
            )
        )

        distances.append(
            (
                cls,
                distance
            )
        )

    distances.sort(
        key=lambda x: x[1]
    )

    ordered_classes = [
        cls
        for cls, _
        in distances
    ]

    true_rank = (
        ordered_classes.index(
            true_class
        ) + 1
        if true_class
        in ordered_classes
        else None
    )

    result = {
        "nearest_centroid_class":
            distances[0][0],

        "nearest_centroid_distance":
            distances[0][1],

        "true_centroid_rank":
            true_rank,

        "true_centroid_distance":
            next(
                (
                    dist
                    for cls, dist
                    in distances
                    if cls == true_class
                ),
                None
            ),
    }

    for rank in range(
        min(
            5,
            len(distances)
        )
    ):

        cls, dist = (
            distances[rank]
        )

        result[
            f"centroid_top{rank + 1}_class"
        ] = cls

        result[
            f"centroid_top{rank + 1}_distance"
        ] = dist

    return result


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    dataset_path = Path(
        args.dataset
    )

    focused_results_path = (
        Path(
            args.focused_results
        )
        if args.focused_results
        else find_latest_focused_results(
            Path(
                args.focused_root
            )
        )
    )

    output_root = Path(
        args.output_dir
    )

    model_path = Path(
        args.asl_model
    )

    scaler_path = Path(
        args.scaler
    )

    hand_model_path = Path(
        args.hand_model
    )

    # ------------------------------------------------------
    # VALIDAR ARCHIVOS
    # ------------------------------------------------------

    for path, name in [
        (
            dataset_path,
            "Dataset ASL"
        ),
        (
            focused_results_path,
            "Resultados focalizados"
        ),
        (
            model_path,
            "Modelo A3"
        ),
        (
            scaler_path,
            "Scaler"
        ),
        (
            hand_model_path,
            "MediaPipe"
        ),
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"{name} no encontrado:\n"
                f"{path}"
            )

    print()
    print(
        "============================================"
    )

    print(
        "=== DIAGNÓSTICO ASL: DATASET VS WEBCAM ==="
    )

    print(
        "============================================"
    )

    print()

    print(
        "Dataset:"
    )

    print(
        f"  {dataset_path}"
    )

    print()

    print(
        "Resultados webcam:"
    )

    print(
        f"  {focused_results_path}"
    )

    print()

    print(
        "Clases analizadas:"
    )

    print(
        "  "
        + ", ".join(
            FOCUS_CLASSES
        )
    )

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    run_name = (
        "diagnosis_"
        + focused_results_path.parent.name
    )

    run_dir = (
        output_root
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # CARGAR DATASET ASL
    # ------------------------------------------------------

    print()
    print(
        "Cargando dataset ASL..."
    )

    df_asl = pd.read_csv(
        dataset_path,
        low_memory=False
    )

    if "class" not in df_asl.columns:

        raise RuntimeError(
            "No existe la columna 'class' "
            "en asl_landmarks_final.csv."
        )

    if "split_final" not in df_asl.columns:

        raise RuntimeError(
            "No existe 'split_final' "
            "en asl_landmarks_final.csv."
        )

    feature_columns = (
        find_img_norm_columns(
            df_asl
        )
    )

    train_df = df_asl[
        df_asl[
            "split_final"
        ]
        .astype(str)
        .str.lower()
        .eq("train")
    ].copy()

    train_df = train_df[
        train_df[
            "class"
        ].isin(
            ALL_CLASSES
        )
    ].copy()

    # Garantizar features numéricas.
    X_train = (
        train_df[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    y_train = (
        train_df[
            "class"
        ]
        .astype(str)
        .to_numpy()
    )

    valid_rows = np.isfinite(
        X_train
    ).all(
        axis=1
    )

    X_train = (
        X_train[
            valid_rows
        ]
    )

    y_train = (
        y_train[
            valid_rows
        ]
    )

    train_df = (
        train_df
        .iloc[
            np.where(
                valid_rows
            )[0]
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "Muestras TRAIN:",
        len(
            train_df
        )
    )

    print(
        "Features:",
        X_train.shape[1]
    )

    # ------------------------------------------------------
    # MODELO + SCALER
    # ------------------------------------------------------

    print()
    print(
        "Cargando A3 + scaler..."
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    scaler = joblib.load(
        scaler_path
    )

    if (
        model.input_shape[-1]
        != 63
    ):
        raise RuntimeError(
            "A3 no espera 63 features."
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

    X_train_scaled = (
        scaler.transform(
            X_train
        )
        .astype(
            np.float32
        )
    )

    print(
        "✅ TRAIN transformado "
        "con el scaler original."
    )

    # ------------------------------------------------------
    # KNN
    # ------------------------------------------------------

    neighbor_count = int(
        args.neighbors
    )

    neighbor_count = min(
        neighbor_count,
        len(
            X_train_scaled
        )
    )

    knn = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="euclidean",
        algorithm="auto"
    )

    knn.fit(
        X_train_scaled
    )

    # ------------------------------------------------------
    # CENTROIDES
    # ------------------------------------------------------

    centroids = build_centroids(
        X_train_scaled,
        y_train
    )

    # ------------------------------------------------------
    # WEBCAM CSV
    # ------------------------------------------------------

    df_webcam = pd.read_csv(
        focused_results_path
    )

    if "true_class" not in df_webcam.columns:

        raise RuntimeError(
            "El focused CSV no contiene "
            "'true_class'."
        )

    df_webcam = df_webcam[
        df_webcam[
            "true_class"
        ].isin(
            FOCUS_CLASSES
        )
    ].copy()

    print()
    print(
        "Capturas webcam a analizar:",
        len(
            df_webcam
        )
    )

    # ------------------------------------------------------
    # MEDIAPIPE
    # ------------------------------------------------------

    landmarker = (
        create_hand_landmarker(
            str(
                hand_model_path
            )
        )
    )

    sample_rows = []

    neighbor_rows = []

    # ------------------------------------------------------
    # ANALIZAR CADA CAPTURA
    # ------------------------------------------------------

    try:

        for index, row in (
            df_webcam.iterrows()
        ):

            true_class = str(
                row[
                    "true_class"
                ]
            )

            attempt = int(
                row[
                    "attempt"
                ]
            )

            image_path = Path(
                str(
                    row[
                        "image_path"
                    ]
                )
            )

            print(
                f"[{index + 1:02d}/"
                f"{len(df_webcam):02d}] "
                f"{true_class} "
                f"intento {attempt}"
            )

            if not image_path.exists():

                print(
                    "  ⚠ imagen no encontrada"
                )

                continue

            image = cv2.imread(
                str(
                    image_path
                )
            )

            if image is None:

                print(
                    "  ⚠ OpenCV no pudo leer imagen"
                )

                continue

            # ----------------------------------------------
            # MISMO PIPELINE
            # ----------------------------------------------

            detection = (
                detect_with_cascade(
                    landmarker,
                    image
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

            if not detected:

                print(
                    "  ⚠ MediaPipe no detectó"
                )

                continue

            features = (
                landmarks_to_feature_vector(
                    result,
                    processed
                )
            )

            if features is None:
                continue

            query_scaled = (
                scaler.transform(
                    features.reshape(
                        1,
                        -1
                    )
                )
                .astype(
                    np.float32
                )[0]
            )

            # ----------------------------------------------
            # A3
            # ----------------------------------------------

            top3 = predict_top3(
                model,
                scaler,
                features
            )

            fresh_pred = (
                top3[0][0]
            )

            fresh_prob = (
                top3[0][1]
            )

            saved_pred = (
                str(
                    row[
                        "predicted_class"
                    ]
                )
                if pd.notna(
                    row[
                        "predicted_class"
                    ]
                )
                else None
            )

            # ----------------------------------------------
            # KNN TRAIN
            # ----------------------------------------------

            distances, indices = (
                knn.kneighbors(
                    query_scaled.reshape(
                        1,
                        -1
                    )
                )
            )

            distances = (
                distances[0]
            )

            indices = (
                indices[0]
            )

            neighbor_classes = (
                y_train[
                    indices
                ]
            )

            class_counts = Counter(
                neighbor_classes
            )

            dominant_neighbor_class = (
                class_counts
                .most_common(1)[0][0]
            )

            true_neighbors = int(
                np.sum(
                    neighbor_classes
                    == true_class
                )
            )

            # ----------------------------------------------
            # CENTROIDES
            # ----------------------------------------------

            centroid_info = (
                centroid_analysis(
                    query_scaled,
                    centroids,
                    true_class
                )
            )

            # ----------------------------------------------
            # RESUMEN MUESTRA
            # ----------------------------------------------

            sample_result = {
                "true_class":
                    true_class,

                "attempt":
                    attempt,

                "image_path":
                    str(
                        image_path
                    ),

                "detection_method":
                    variant,

                "saved_prediction":
                    saved_pred,

                "fresh_prediction":
                    fresh_pred,

                "prediction_reproduced":
                    (
                        saved_pred
                        == fresh_pred
                    ),

                "fresh_top1_prob":
                    fresh_prob,

                "a3_top2_class":
                    top3[1][0],

                "a3_top2_prob":
                    top3[1][1],

                "a3_top3_class":
                    top3[2][0],

                "a3_top3_prob":
                    top3[2][1],

                "nearest_neighbor_class":
                    neighbor_classes[0],

                "nearest_neighbor_distance":
                    float(
                        distances[0]
                    ),

                "knn_dominant_class":
                    dominant_neighbor_class,

                "true_class_neighbors_in_top_k":
                    true_neighbors,

                "neighbor_k":
                    neighbor_count,

                "neighbor_class_distribution":
                    json.dumps(
                        dict(
                            class_counts
                        ),
                        ensure_ascii=False
                    ),
            }

            sample_result.update(
                centroid_info
            )

            sample_rows.append(
                sample_result
            )

            # ----------------------------------------------
            # DETALLE DE VECINOS
            # ----------------------------------------------

            for rank, (
                train_index,
                distance
            ) in enumerate(
                zip(
                    indices,
                    distances
                ),
                start=1
            ):

                train_row = (
                    train_df.iloc[
                        int(
                            train_index
                        )
                    ]
                )

                neighbor_row = {
                    "query_true_class":
                        true_class,

                    "query_attempt":
                        attempt,

                    "query_image_path":
                        str(
                            image_path
                        ),

                    "neighbor_rank":
                        rank,

                    "neighbor_class":
                        str(
                            train_row[
                                "class"
                            ]
                        ),

                    "distance":
                        float(
                            distance
                        ),
                }

                # Si existe path en el dataset,
                # conservarlo para abrir luego
                # la imagen original.
                for candidate_column in [
                    "path",
                    "image_path",
                    "source_path",
                    "filepath",
                ]:

                    if (
                        candidate_column
                        in train_df.columns
                    ):

                        neighbor_row[
                            "neighbor_image_path"
                        ] = str(
                            train_row[
                                candidate_column
                            ]
                        )

                        break

                # Información adicional útil.
                for optional_column in [
                    "split_original",
                    "split_final",
                    "group",
                    "source_language",
                    "dataset",
                ]:

                    if (
                        optional_column
                        in train_df.columns
                    ):

                        neighbor_row[
                            optional_column
                        ] = train_row[
                            optional_column
                        ]

                neighbor_rows.append(
                    neighbor_row
                )

    finally:

        landmarker.close()

    # ------------------------------------------------------
    # DATAFRAMES DE SALIDA
    # ------------------------------------------------------

    samples_df = pd.DataFrame(
        sample_rows
    )

    neighbors_df = pd.DataFrame(
        neighbor_rows
    )

    if len(
        samples_df
    ) == 0:

        raise RuntimeError(
            "No se pudo analizar ninguna captura."
        )

    # ------------------------------------------------------
    # RESUMEN POR CLASE
    # ------------------------------------------------------

    class_rows = []

    for cls in FOCUS_CLASSES:

        part = samples_df[
            samples_df[
                "true_class"
            ] == cls
        ]

        if len(part) == 0:
            continue

        fresh_accuracy = float(
            (
                part[
                    "fresh_prediction"
                ]
                == cls
            ).mean()
        )

        centroid_true_top1 = float(
            (
                part[
                    "nearest_centroid_class"
                ]
                == cls
            ).mean()
        )

        centroid_true_top3 = float(
            (
                part[
                    "true_centroid_rank"
                ]
                <= 3
            ).mean()
        )

        avg_true_centroid_rank = float(
            part[
                "true_centroid_rank"
            ].mean()
        )

        avg_true_neighbors = float(
            part[
                "true_class_neighbors_in_top_k"
            ].mean()
        )

        dominant_knn = (
            part[
                "knn_dominant_class"
            ]
            .value_counts()
            .index[0]
        )

        dominant_centroid = (
            part[
                "nearest_centroid_class"
            ]
            .value_counts()
            .index[0]
        )

        dominant_a3 = (
            part[
                "fresh_prediction"
            ]
            .value_counts()
            .index[0]
        )

        class_rows.append({
            "class":
                cls,

            "samples":
                len(
                    part
                ),

            "a3_accuracy":
                fresh_accuracy,

            "dominant_a3_prediction":
                dominant_a3,

            "dominant_knn_class":
                dominant_knn,

            "avg_true_neighbors_in_top_k":
                avg_true_neighbors,

            "neighbor_k":
                neighbor_count,

            "nearest_centroid_is_true_rate":
                centroid_true_top1,

            "true_centroid_in_top3_rate":
                centroid_true_top3,

            "avg_true_centroid_rank":
                avg_true_centroid_rank,

            "dominant_nearest_centroid":
                dominant_centroid,
        })

    by_class_df = pd.DataFrame(
        class_rows
    )

    # ------------------------------------------------------
    # GUARDAR
    # ------------------------------------------------------

    samples_path = (
        run_dir
        / "domain_diagnosis_samples.csv"
    )

    neighbors_path = (
        run_dir
        / "domain_nearest_neighbors.csv"
    )

    class_path = (
        run_dir
        / "domain_diagnosis_by_class.csv"
    )

    samples_df.to_csv(
        samples_path,
        index=False
    )

    neighbors_df.to_csv(
        neighbors_path,
        index=False
    )

    by_class_df.to_csv(
        class_path,
        index=False
    )

    # ------------------------------------------------------
    # CONSOLA
    # ------------------------------------------------------

    print()
    print(
        "========================================="
    )

    print(
        "=== RESUMEN DATASET VS WEBCAM ==="
    )

    print(
        "========================================="
    )

    print()

    display = (
        by_class_df.copy()
    )

    for col in [
        "a3_accuracy",
        "nearest_centroid_is_true_rate",
        "true_centroid_in_top3_rate",
    ]:

        display[col] = (
            display[col]
            * 100
        ).round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "Lectura rápida:"
    )

    print(
        "- Si KNN y centroides también "
        "apuntan a otra clase, la captura "
        "webcam está cayendo en otra región "
        "del dataset."
    )

    print(
        "- Si KNN/centroide apuntan a la "
        "clase correcta pero A3 no, el "
        "problema está más cerca del "
        "clasificador."
    )

    print(
        "- Si la clase verdadera ni siquiera "
        "aparece cerca geométricamente, hay "
        "que revisar ejecución, orientación "
        "o representación del dataset."
    )

    print()
    print(
        "=== ARCHIVOS GENERADOS ==="
    )

    print(
        samples_path
    )

    print(
        neighbors_path
    )

    print(
        class_path
    )

    print()
    print(
        "✅ Diagnóstico terminado."
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Diagnóstico ASL webcam vs "
            "distribución de TRAIN."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_ASL_DATASET
        )
    )

    parser.add_argument(
        "--focused-root",
        default=str(
            DEFAULT_FOCUSED_ROOT
        )
    )

    parser.add_argument(
        "--focused-results",
        default=None,
        help=(
            "Ruta específica a "
            "focused_webcam_results.csv. "
            "Si se omite, usa el run más reciente."
        )
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=10
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
        "--hand-model",
        default=str(
            DEFAULT_HAND_MODEL
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
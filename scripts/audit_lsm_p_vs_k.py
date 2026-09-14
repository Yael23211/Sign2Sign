# ============================================================
# Sign2Sign
# Auditoria focalizada LSM: P webcam vs P/K de TRAIN
#
# Objetivo:
#   Analizar por que las capturas webcam de P pueden confundirse
#   con K (u otras clases), comparando la representacion world_norm
#   con las referencias P y K que realmente vio el modelo en TRAIN.
#
# Para cada captura webcam P:
#   - extrae world_norm con el MISMO pipeline
#   - aplica el MISMO StandardScaler de V2-M2
#   - encuentra:
#       3 P TRAIN mas cercanas
#       3 K TRAIN mas cercanas
#
# Genera:
#   - p_vs_k_visual.png
#   - p_vs_k_distances.csv
#
# NO modifica ningun dataset.
# ============================================================


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import io
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
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
# REUTILIZAR PIPELINE VALIDADO
# ============================================================

from evaluate_lsm_webcam import (
    MEDIAPIPE_MODEL_PATH,
    SCALER_PATH,
    extract_world_norm,
)

from src.landmarks.pipeline import (
    create_hand_landmarker,
    detect_with_cascade,
)


# ============================================================
# RUTAS
# ============================================================

DATA_ROOT = Path(
    r"D:\Sign2SignData"
)

CANONICAL_DATASET = (
    DATA_ROOT
    / "processed"
    / "landmarks"
    / "lsm_landmarks_canonical_v2_splits.csv"
)

FOCUSED_ROOT = (
    DATA_ROOT
    / "inventory"
    / "webcam_eval"
    / "lsm_focused"
)

OUTPUT_ROOT = (
    DATA_ROOT
    / "inventory"
    / "webcam_eval"
    / "lsm_p_vs_k_audit"
)


# ============================================================
# CONFIG
# ============================================================

N_NEIGHBORS = 3

REFERENCE_CLASSES = [
    "P",
    "K",
]


# ============================================================
# UTILIDADES
# ============================================================

def find_latest_focused_run():

    if not FOCUSED_ROOT.exists():
        raise FileNotFoundError(
            f"No existe:\n{FOCUSED_ROOT}"
        )

    runs = sorted(
        [
            p
            for p in FOCUSED_ROOT.iterdir()
            if p.is_dir()
            and p.name.startswith("focused_run_")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError(
            "No se encontro ningun focused_run."
        )

    return runs[0]


def find_lsm_zip():

    candidates = list(
        (DATA_ROOT / "raw").rglob(
            "LSM_original.zip"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "No se encontro LSM_original.zip debajo de "
            f"{DATA_ROOT / 'raw'}"
        )

    return candidates[0]


def find_world_norm_columns(df):

    cols = [
        c
        for c in df.columns
        if "world_norm" in c.lower()
    ]

    if len(cols) != 63:

        print(
            "\nColumnas candidatas world_norm encontradas:",
            len(cols)
        )

        print(cols[:20])

        raise RuntimeError(
            "Se esperaban exactamente 63 columnas world_norm."
        )

    return cols


def normalize_zip_path(path):

    return str(path).replace(
        "\\",
        "/"
    ).lstrip("/")


def build_zip_index(zip_file):

    print(
        "Indexando ZIP..."
    )

    names = zip_file.namelist()

    exact = {}

    basename = {}

    for name in names:

        normalized = normalize_zip_path(
            name
        )

        exact[
            normalized.lower()
        ] = name

        base = Path(
            normalized
        ).name.lower()

        basename.setdefault(
            base,
            []
        ).append(
            name
        )

    print(
        f"Archivos indexados: {len(names)}"
    )

    return exact, basename


def resolve_zip_member(
    row_path,
    exact_index,
    basename_index
):

    normalized = normalize_zip_path(
        row_path
    )

    # Intento exacto.
    key = normalized.lower()

    if key in exact_index:
        return exact_index[key]

    # Algunos manifests pueden incluir prefijos diferentes.
    for indexed_key, original_name in exact_index.items():

        if indexed_key.endswith(
            key
        ):
            return original_name

    # Fallback por nombre de archivo.
    base = Path(
        normalized
    ).name.lower()

    matches = basename_index.get(
        base,
        []
    )

    if len(matches) == 1:
        return matches[0]

    return None


def read_image_from_zip(
    zip_file,
    member
):

    if member is None:
        return None

    try:

        data = zip_file.read(
            member
        )

        arr = np.frombuffer(
            data,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR
        )

        return img

    except Exception:

        return None


def bgr_to_rgb(image):

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# WEBCAM -> WORLD_NORM
# ============================================================

def extract_capture_features(
    image_path,
    landmarker
):

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:

        raise RuntimeError(
            f"No se pudo leer:\n{image_path}"
        )

    detection = detect_with_cascade(
        landmarker,
        frame
    )

    if not isinstance(
        detection,
        dict
    ):

        raise RuntimeError(
            "detect_with_cascade no devolvio dict."
        )

    if not detection.get(
        "ok",
        False
    ):

        return None, detection.get(
            "variant",
            "unknown"
        )

    result = detection.get(
        "result"
    )

    if result is None:

        return None, detection.get(
            "variant",
            "unknown"
        )

    features = extract_world_norm(
        result
    )

    return (
        features,
        detection.get(
            "variant",
            "unknown"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("=== LSM - AUDITORIA P WEBCAM vs P/K TRAIN ===")
    print("=" * 78)

    # ========================================================
    # VALIDACIONES
    # ========================================================

    if not CANONICAL_DATASET.exists():

        raise FileNotFoundError(
            f"No existe dataset Canonical V2:\n"
            f"{CANONICAL_DATASET}"
        )

    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            f"No existe scaler:\n"
            f"{SCALER_PATH}"
        )

    if not MEDIAPIPE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No existe MediaPipe model:\n"
            f"{MEDIAPIPE_MODEL_PATH}"
        )


    # ========================================================
    # ULTIMO FOCUSED RUN
    # ========================================================

    focused_run = find_latest_focused_run()

    captures_dir = (
        focused_run
        / "captures"
    )

    print(
        "\nFocused run:"
    )

    print(
        focused_run
    )


    p_captures = sorted(
        captures_dir.glob(
            "P_*.jpg"
        )
    )


    if not p_captures:

        raise FileNotFoundError(
            "No se encontraron capturas P_*.jpg."
        )


    print(
        f"\nCapturas P encontradas: "
        f"{len(p_captures)}"
    )


    # ========================================================
    # RESULTADOS DEL FOCALIZADO
    # Opcional, solo para anotar la prediccion.
    # ========================================================

    focused_results_path = (
        focused_run
        / "focused_webcam_results.csv"
    )


    focused_results = None

    if focused_results_path.exists():

        focused_results = pd.read_csv(
            focused_results_path
        )


    # ========================================================
    # DATASET
    # ========================================================

    print(
        "\nLeyendo Canonical V2..."
    )


    df = pd.read_csv(
        CANONICAL_DATASET,
        low_memory=False
    )


    print(
        f"Muestras totales: {len(df)}"
    )


    # ========================================================
    # DETECTAR COLUMNA SPLIT
    # ========================================================

    if "split_final" in df.columns:

        split_col = (
            "split_final"
        )

    elif "split_final_v2" in df.columns:

        split_col = (
            "split_final_v2"
        )

    else:

        raise RuntimeError(
            "No encontre split_final ni split_final_v2."
        )


    # ========================================================
    # TRAIN P / K
    # ========================================================

    train_pk = df[
        (
            df[
                split_col
            ].astype(str).str.lower()
            == "train"
        )
        &
        (
            df[
                "class"
            ].isin(
                REFERENCE_CLASSES
            )
        )
    ].copy()


    print(
        "\nReferencias TRAIN:"
    )

    print(
        train_pk[
            "class"
        ].value_counts()
    )


    # ========================================================
    # FEATURES
    # ========================================================

    world_cols = find_world_norm_columns(
        train_pk
    )


    X_train = (
        train_pk[
            world_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    if not np.isfinite(
        X_train
    ).all():

        raise RuntimeError(
            "TRAIN P/K contiene NaN o Inf."
        )


    # ========================================================
    # SCALER
    # ========================================================

    scaler = joblib.load(
        SCALER_PATH
    )


    X_train_scaled = scaler.transform(
        X_train
    ).astype(
        np.float32
    )


    labels_train = (
        train_pk[
            "class"
        ]
        .astype(str)
        .to_numpy()
    )


    train_indices = (
        train_pk.index.to_numpy()
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    print(
        "\nCreando Hand Landmarker..."
    )


    landmarker = create_hand_landmarker(
        str(
            MEDIAPIPE_MODEL_PATH
        )
    )


    # ========================================================
    # ZIP
    # ========================================================

    zip_path = find_lsm_zip()

    print(
        "\nZIP LSM:"
    )

    print(
        zip_path
    )


    zip_file = zipfile.ZipFile(
        zip_path,
        "r"
    )


    exact_index, basename_index = (
        build_zip_index(
            zip_file
        )
    )


    # ========================================================
    # SALIDA
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # PROCESAR CAPTURAS
    # ========================================================

    result_rows = []

    visual_rows = []


    try:

        for capture_path in p_captures:

            print()
            print("-" * 78)

            print(
                f"Procesando {capture_path.name}"
            )


            features, variant = (
                extract_capture_features(
                    capture_path,
                    landmarker
                )
            )


            if features is None:

                print(
                    "No detection."
                )

                result_rows.append(
                    {
                        "capture":
                            capture_path.name,

                        "detected":
                            False,

                        "variant":
                            variant,
                    }
                )

                continue


            X_query = scaler.transform(
                features.reshape(
                    1,
                    -1
                )
            ).astype(
                np.float32
            )[0]


            # =================================================
            # DISTANCIAS
            # =================================================

            distances = np.linalg.norm(
                X_train_scaled
                - X_query,
                axis=1
            )


            # =================================================
            # INDICES P
            # =================================================

            p_mask = (
                labels_train
                == "P"
            )

            k_mask = (
                labels_train
                == "K"
            )


            p_positions = np.where(
                p_mask
            )[0]

            k_positions = np.where(
                k_mask
            )[0]


            p_sorted = p_positions[
                np.argsort(
                    distances[
                        p_positions
                    ]
                )
            ][:N_NEIGHBORS]


            k_sorted = k_positions[
                np.argsort(
                    distances[
                        k_positions
                    ]
                )
            ][:N_NEIGHBORS]


            nearest_p_distance = float(
                distances[
                    p_sorted[0]
                ]
            )


            nearest_k_distance = float(
                distances[
                    k_sorted[0]
                ]
            )


            closer_reference = (
                "P"
                if nearest_p_distance
                <= nearest_k_distance
                else "K"
            )


            # =================================================
            # PREDICCION QUE TUVO EN PRUEBA FOCALIZADA
            # =================================================

            predicted_class = None
            top1_prob = None


            if focused_results is not None:

                stem = (
                    capture_path.stem
                )

                attempt_number = int(
                    stem.split("_")[-1]
                )


                match = focused_results[
                    (
                        focused_results[
                            "true_class"
                        ]
                        == "P"
                    )
                    &
                    (
                        focused_results[
                            "attempt_in_class"
                        ]
                        == attempt_number
                    )
                ]


                if len(match):

                    row = match.iloc[0]

                    predicted_class = (
                        row.get(
                            "predicted_class"
                        )
                    )

                    top1_prob = (
                        row.get(
                            "top1_prob"
                        )
                    )


            print(
                f"Nearest P: "
                f"{nearest_p_distance:.3f}"
            )

            print(
                f"Nearest K: "
                f"{nearest_k_distance:.3f}"
            )

            print(
                f"Mas cercana geometricamente: "
                f"{closer_reference}"
            )


            if predicted_class is not None:

                print(
                    f"Prediccion focalizada: "
                    f"{predicted_class}"
                )


            # =================================================
            # FILAS CSV
            # =================================================

            result_row = {

                "capture":
                    capture_path.name,

                "detected":
                    True,

                "variant":
                    variant,

                "predicted_class":
                    predicted_class,

                "top1_prob":
                    top1_prob,

                "nearest_P_distance":
                    nearest_p_distance,

                "nearest_K_distance":
                    nearest_k_distance,

                "distance_K_minus_P":
                    (
                        nearest_k_distance
                        - nearest_p_distance
                    ),

                "closer_reference":
                    closer_reference,
            }


            for rank, pos in enumerate(
                p_sorted,
                start=1
            ):

                dataset_index = (
                    train_indices[
                        pos
                    ]
                )

                row = df.loc[
                    dataset_index
                ]

                result_row[
                    f"P_{rank}_distance"
                ] = float(
                    distances[
                        pos
                    ]
                )

                result_row[
                    f"P_{rank}_group"
                ] = row.get(
                    "group",
                    ""
                )

                result_row[
                    f"P_{rank}_path"
                ] = row.get(
                    "path",
                    ""
                )


            for rank, pos in enumerate(
                k_sorted,
                start=1
            ):

                dataset_index = (
                    train_indices[
                        pos
                    ]
                )

                row = df.loc[
                    dataset_index
                ]

                result_row[
                    f"K_{rank}_distance"
                ] = float(
                    distances[
                        pos
                    ]
                )

                result_row[
                    f"K_{rank}_group"
                ] = row.get(
                    "group",
                    ""
                )

                result_row[
                    f"K_{rank}_path"
                ] = row.get(
                    "path",
                    ""
                )


            result_rows.append(
                result_row
            )


            # =================================================
            # PREPARAR IMAGENES PARA FIGURA
            # =================================================

            webcam_img = cv2.imread(
                str(
                    capture_path
                )
            )


            p_images = []

            k_images = []


            for pos in p_sorted:

                dataset_index = (
                    train_indices[
                        pos
                    ]
                )

                row = df.loc[
                    dataset_index
                ]

                member = resolve_zip_member(
                    row.get(
                        "path",
                        ""
                    ),
                    exact_index,
                    basename_index
                )

                image = read_image_from_zip(
                    zip_file,
                    member
                )

                p_images.append(
                    (
                        image,
                        float(
                            distances[
                                pos
                            ]
                        ),
                        row.get(
                            "group",
                            ""
                        )
                    )
                )


            for pos in k_sorted:

                dataset_index = (
                    train_indices[
                        pos
                    ]
                )

                row = df.loc[
                    dataset_index
                ]

                member = resolve_zip_member(
                    row.get(
                        "path",
                        ""
                    ),
                    exact_index,
                    basename_index
                )

                image = read_image_from_zip(
                    zip_file,
                    member
                )

                k_images.append(
                    (
                        image,
                        float(
                            distances[
                                pos
                            ]
                        ),
                        row.get(
                            "group",
                            ""
                        )
                    )
                )


            visual_rows.append(
                {
                    "capture":
                        capture_path.name,

                    "webcam":
                        webcam_img,

                    "predicted_class":
                        predicted_class,

                    "nearest_p":
                        p_images,

                    "nearest_k":
                        k_images,

                    "nearest_p_distance":
                        nearest_p_distance,

                    "nearest_k_distance":
                        nearest_k_distance,

                    "closer_reference":
                        closer_reference,
                }
            )


    finally:

        try:

            landmarker.close()

        except Exception:

            pass

        zip_file.close()


    # ========================================================
    # CSV
    # ========================================================

    result_df = pd.DataFrame(
        result_rows
    )


    csv_path = (
        OUTPUT_ROOT
        / "p_vs_k_distances.csv"
    )


    result_df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # FIGURA
    #
    # Columnas:
    #   0 webcam
    #   1-3 P train
    #   4-6 K train
    # ========================================================

    if visual_rows:

        n_rows = len(
            visual_rows
        )

        n_cols = (
            1
            + N_NEIGHBORS
            + N_NEIGHBORS
        )


        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                21,
                3.7 * n_rows
            )
        )


        if n_rows == 1:

            axes = np.expand_dims(
                axes,
                axis=0
            )


        fig.suptitle(
            (
                "LSM P — Webcam vs TRAIN P/K\n"
                "Izquierda: captura webcam | "
                "Centro: P mas cercanas | "
                "Derecha: K mas cercanas"
            ),
            fontsize=16
        )


        for row_idx, item in enumerate(
            visual_rows
        ):

            # ===============================================
            # WEBCAM
            # ===============================================

            ax = axes[
                row_idx,
                0
            ]

            webcam_rgb = bgr_to_rgb(
                item[
                    "webcam"
                ]
            )


            if webcam_rgb is not None:

                ax.imshow(
                    webcam_rgb
                )


            prediction_text = (
                (
                    f"\nPred: "
                    f"{item['predicted_class']}"
                )
                if item[
                    "predicted_class"
                ] is not None
                else ""
            )


            ax.set_title(
                (
                    f"WEBCAM P\n"
                    f"{item['capture']}"
                    f"{prediction_text}\n"
                    f"dP={item['nearest_p_distance']:.2f} | "
                    f"dK={item['nearest_k_distance']:.2f}\n"
                    f"Mas cercana: "
                    f"{item['closer_reference']}"
                ),
                fontsize=9
            )


            ax.axis(
                "off"
            )


            # ===============================================
            # P TRAIN
            # ===============================================

            for rank, (
                image,
                distance,
                group
            ) in enumerate(
                item[
                    "nearest_p"
                ],
                start=1
            ):

                ax = axes[
                    row_idx,
                    rank
                ]


                rgb = bgr_to_rgb(
                    image
                )


                if rgb is not None:

                    ax.imshow(
                        rgb
                    )

                else:

                    ax.text(
                        0.5,
                        0.5,
                        "Imagen no encontrada",
                        ha="center",
                        va="center"
                    )


                ax.set_title(
                    (
                        f"P TRAIN #{rank}\n"
                        f"group={group}\n"
                        f"d={distance:.2f}"
                    ),
                    fontsize=9
                )


                ax.axis(
                    "off"
                )


            # ===============================================
            # K TRAIN
            # ===============================================

            for rank, (
                image,
                distance,
                group
            ) in enumerate(
                item[
                    "nearest_k"
                ],
                start=1
            ):

                col = (
                    N_NEIGHBORS
                    + rank
                )


                ax = axes[
                    row_idx,
                    col
                ]


                rgb = bgr_to_rgb(
                    image
                )


                if rgb is not None:

                    ax.imshow(
                        rgb
                    )

                else:

                    ax.text(
                        0.5,
                        0.5,
                        "Imagen no encontrada",
                        ha="center",
                        va="center"
                    )


                ax.set_title(
                    (
                        f"K TRAIN #{rank}\n"
                        f"group={group}\n"
                        f"d={distance:.2f}"
                    ),
                    fontsize=9
                )


                ax.axis(
                    "off"
                )


        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.98
            ]
        )


        png_path = (
            OUTPUT_ROOT
            / "p_vs_k_visual.png"
        )


        plt.savefig(
            png_path,
            dpi=180,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


    else:

        png_path = None


    # ========================================================
    # RESUMEN
    # ========================================================

    valid_df = result_df[
        result_df[
            "detected"
        ] == True
    ]


    print()
    print("=" * 78)
    print("=== AUDITORIA TERMINADA ===")
    print("=" * 78)


    if len(
        valid_df
    ):

        print(
            "\nCapturas detectadas:",
            len(
                valid_df
            )
        )


        print(
            "\nReferencia geometrica mas cercana:"
        )


        print(
            valid_df[
                "closer_reference"
            ]
            .value_counts()
            .to_string()
        )


        print(
            "\nDistancia media a P:"
        )

        print(
            f"{valid_df['nearest_P_distance'].mean():.4f}"
        )


        print(
            "\nDistancia media a K:"
        )

        print(
            f"{valid_df['nearest_K_distance'].mean():.4f}"
        )


        print()
        print(
            "Detalle:"
        )


        print(
            valid_df[
                [
                    "capture",
                    "predicted_class",
                    "nearest_P_distance",
                    "nearest_K_distance",
                    "distance_K_minus_P",
                    "closer_reference",
                ]
            ]
            .to_string(
                index=False
            )
        )


    print()
    print(
        "CSV:"
    )

    print(
        csv_path
    )


    if png_path:

        print(
            "\nVisual:"
        )

        print(
            png_path
        )


    print()
    print(
        "✅ No se modifico ningun dataset."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()
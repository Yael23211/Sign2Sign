import argparse
import inspect
import io
import sys
import zipfile
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

FOCUS_CLASSES = list("OPRTX")


DEFAULT_ASL_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks\asl_landmarks_final.csv"
)


DEFAULT_ASL_ZIP = Path(
    r"D:\Sign2SignData\raw\asl\ASL_original.zip"
)


DEFAULT_DIAGNOSIS_ROOT = Path(
    r"D:\Sign2SignData\inventory\webcam_eval\asl_domain_diagnosis"
)


DEFAULT_HAND_MODEL = (
    ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)


DEFAULT_SCALER = (
    ROOT
    / "models"
    / "asl"
    / "scaler_img_norm.joblib"
)


# ==========================================================
# ÚLTIMO DIAGNÓSTICO
# ==========================================================

def find_latest_diagnosis(root):

    if not root.exists():
        raise FileNotFoundError(
            f"No existe:\n{root}"
        )

    candidates = []

    for directory in root.glob(
        "diagnosis_*"
    ):

        samples = (
            directory
            / "domain_diagnosis_samples.csv"
        )

        neighbors = (
            directory
            / "domain_nearest_neighbors.csv"
        )

        if (
            samples.exists()
            and neighbors.exists()
        ):
            candidates.append(
                directory
            )

    if not candidates:

        raise FileNotFoundError(
            "No se encontró ningún diagnóstico "
            f"válido dentro de:\n{root}"
        )

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return candidates[0]


# ==========================================================
# FEATURES
# ==========================================================

def find_img_norm_columns(df):

    columns = [
        c
        for c in df.columns
        if str(c).startswith(
            "img_norm_"
        )
    ]

    if len(columns) != 63:

        raise RuntimeError(
            "Se esperaban 63 columnas img_norm "
            f"y se encontraron {len(columns)}."
        )

    return columns


def find_path_column(df):

    candidates = [
        "path",
        "image_path",
        "source_path",
        "filepath",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise RuntimeError(
        "No se encontró ninguna columna de "
        "ruta de imagen en el dataset."
    )


# ==========================================================
# ZIP ASL
# ==========================================================

def normalize_zip_path(value):

    if pd.isna(value):
        return None

    return (
        str(value)
        .replace("\\", "/")
        .lstrip("/")
        .lstrip("./")
    )


def build_zip_index(zip_file):

    print()
    print(
        "Indexando imágenes dentro del ZIP..."
    )

    members = [
        name
        for name in zip_file.namelist()
        if not name.endswith("/")
    ]

    exact_index = {}

    basename_index = {}

    for member in members:

        normalized = normalize_zip_path(
            member
        )

        lower = normalized.lower()

        exact_index[
            lower
        ] = member

        basename = Path(
            normalized
        ).name.lower()

        basename_index.setdefault(
            basename,
            []
        ).append(
            member
        )

    print(
        "Archivos encontrados en ZIP:",
        len(members)
    )

    # Mostrar una muestra para entender
    # automáticamente la estructura real.
    print()
    print(
        "Ejemplos de rutas internas:"
    )

    for member in members[:5]:

        print(
            f"  {member}"
        )

    return (
        exact_index,
        basename_index
    )


def resolve_zip_member(
    stored_path,
    exact_index,
    basename_index
):

    normalized = normalize_zip_path(
        stored_path
    )

    if not normalized:
        return None

    lower = normalized.lower()

    # ------------------------------------------------------
    # 1. Coincidencia exacta
    # ------------------------------------------------------

    if lower in exact_index:

        return exact_index[
            lower
        ]

    # ------------------------------------------------------
    # 2. Puede haber una carpeta extra antes:
    #
    # algo/Train_Alphabet/P/imagen.png
    # ------------------------------------------------------

    suffix = (
        "/"
        + lower
    )

    matches = [
        member
        for key, member
        in exact_index.items()
        if key.endswith(
            suffix
        )
    ]

    if len(matches) == 1:

        return matches[0]

    # ------------------------------------------------------
    # 3. Buscar usando:
    # Train_Alphabet/P/archivo.png
    #
    # aunque el ZIP tenga otras carpetas antes.
    # ------------------------------------------------------

    parts = normalized.split(
        "/"
    )

    if "Train_Alphabet" in parts:

        idx = parts.index(
            "Train_Alphabet"
        )

        relative = "/".join(
            parts[idx:]
        ).lower()

        suffix = (
            "/"
            + relative
        )

        matches = [
            member
            for key, member
            in exact_index.items()
            if (
                key == relative
                or key.endswith(
                    suffix
                )
            )
        ]

        if len(matches) == 1:

            return matches[0]

    # ------------------------------------------------------
    # 4. Último recurso:
    # basename.
    #
    # Los nombres UUID deberían ser únicos.
    # ------------------------------------------------------

    basename = Path(
        normalized
    ).name.lower()

    matches = basename_index.get(
        basename,
        []
    )

    if len(matches) == 1:

        return matches[0]

    return None


def load_zip_image_rgb(
    zip_file,
    member
):

    if member is None:
        return None

    try:

        raw = zip_file.read(
            member
        )

    except KeyError:

        return None

    array = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# CASCADE
# ==========================================================

def unpack_detection(
    detection
):

    if not isinstance(
        detection,
        dict
    ):

        raise RuntimeError(
            "detect_with_cascade() "
            "no devolvió dict."
        )

    if not detection.get(
        "ok",
        False
    ):

        return (
            False,
            None,
            None,
            None
        )

    return (
        True,
        detection["variant"],
        detection["processed"],
        detection["result"]
    )


# ==========================================================
# WEBCAM -> IMG_NORM
# ==========================================================

def landmarks_to_feature_vector(
    result,
    processed_image
):

    if (
        result is None
        or not result.hand_landmarks
    ):

        return None

    landmarks = (
        result.hand_landmarks[0]
    )

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

    features = np.asarray(
        output[1],
        dtype=np.float32
    ).reshape(-1)

    if features.shape != (63,):

        raise RuntimeError(
            "img_norm no contiene "
            "63 características."
        )

    return features


# ==========================================================
# IMÁGENES WEBCAM
# ==========================================================

def load_webcam_rgb(path):

    path = Path(
        path
    )

    if not path.exists():
        return None

    image = cv2.imread(
        str(path)
    )

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# DIBUJO
# ==========================================================

def draw_image(
    ax,
    image,
    title
):

    ax.axis(
        "off"
    )

    ax.set_title(
        title,
        fontsize=9
    )

    if image is None:

        ax.text(
            0.5,
            0.5,
            "Imagen no encontrada",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9
        )

        return

    ax.imshow(
        image
    )


# ==========================================================
# CAPTURAS REPRESENTATIVAS
# ==========================================================

def choose_representative_samples(
    class_samples
):

    class_samples = (
        class_samples
        .sort_values(
            "nearest_neighbor_distance"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        class_samples
    ) <= 3:

        return class_samples

    # Más cercana
    # intermedia
    # más alejada
    indices = [
        0,
        len(class_samples) // 2,
        len(class_samples) - 1
    ]

    return (
        class_samples
        .iloc[
            indices
        ]
        .copy()
    )


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    dataset_path = Path(
        args.dataset
    )

    asl_zip_path = Path(
        args.asl_zip
    )

    diagnosis_root = Path(
        args.diagnosis_root
    )

    scaler_path = Path(
        args.scaler
    )

    hand_model_path = Path(
        args.hand_model
    )

    # ------------------------------------------------------
    # VALIDAR
    # ------------------------------------------------------

    for path, name in [
        (
            dataset_path,
            "Dataset ASL"
        ),
        (
            asl_zip_path,
            "ZIP ASL"
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

    # ------------------------------------------------------
    # DIAGNÓSTICO
    # ------------------------------------------------------

    if args.diagnosis_dir:

        diagnosis_dir = Path(
            args.diagnosis_dir
        )

    else:

        diagnosis_dir = (
            find_latest_diagnosis(
                diagnosis_root
            )
        )

    samples_path = (
        diagnosis_dir
        / "domain_diagnosis_samples.csv"
    )

    neighbors_path = (
        diagnosis_dir
        / "domain_nearest_neighbors.csv"
    )

    output_dir = (
        diagnosis_dir
        / "visual_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== VISUALIZACIÓN DATASET VS WEBCAM ==="
    )

    print(
        "============================================"
    )

    print()
    print(
        "Diagnóstico:"
    )

    print(
        f"  {diagnosis_dir}"
    )

    print()
    print(
        "ZIP ASL:"
    )

    print(
        f"  {asl_zip_path}"
    )

    print()
    print(
        "Salida:"
    )

    print(
        f"  {output_dir}"
    )

    # ------------------------------------------------------
    # DATASET ASL
    # ------------------------------------------------------

    print()
    print(
        "Cargando TRAIN ASL..."
    )

    df = pd.read_csv(
        dataset_path,
        low_memory=False
    )

    if "class" not in df.columns:

        raise RuntimeError(
            "No existe columna 'class'."
        )

    if "split_final" not in df.columns:

        raise RuntimeError(
            "No existe columna 'split_final'."
        )

    feature_columns = (
        find_img_norm_columns(
            df
        )
    )

    path_column = (
        find_path_column(
            df
        )
    )

    print(
        "Columna de ruta detectada:",
        path_column
    )

    train_df = df[
        df[
            "split_final"
        ]
        .astype(str)
        .str.lower()
        .eq("train")
    ].copy()

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

    valid = np.isfinite(
        X_train
    ).all(
        axis=1
    )

    valid_indices = np.where(
        valid
    )[0]

    train_df = (
        train_df
        .iloc[
            valid_indices
        ]
        .reset_index(
            drop=True
        )
    )

    X_train = (
        X_train[
            valid
        ]
    )

    print(
        "TRAIN válido:",
        len(
            train_df
        )
    )

    # ------------------------------------------------------
    # SCALER
    # ------------------------------------------------------

    scaler = joblib.load(
        scaler_path
    )

    X_train_scaled = (
        scaler.transform(
            X_train
        )
        .astype(
            np.float32
        )
    )

    # ------------------------------------------------------
    # RESULTADOS DIAGNÓSTICO
    # ------------------------------------------------------

    samples_df = pd.read_csv(
        samples_path
    )

    neighbors_df = pd.read_csv(
        neighbors_path
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

    visual_rows = []

    missing_zip_images = []

    # ------------------------------------------------------
    # ABRIR ZIP UNA SOLA VEZ
    # ------------------------------------------------------

    with zipfile.ZipFile(
        asl_zip_path,
        "r"
    ) as zip_file:

        (
            zip_exact_index,
            zip_basename_index
        ) = build_zip_index(
            zip_file
        )

        try:

            # ==================================================
            # CADA CLASE
            # ==================================================

            for cls in FOCUS_CLASSES:

                print()
                print(
                    f"Procesando {cls}..."
                )

                class_samples = (
                    samples_df[
                        samples_df[
                            "true_class"
                        ].astype(str)
                        == cls
                    ]
                    .copy()
                )

                if len(
                    class_samples
                ) == 0:

                    print(
                        "  ⚠ Sin muestras."
                    )

                    continue

                selected = (
                    choose_representative_samples(
                        class_samples
                    )
                )

                # --------------------------------------------------
                # 7 columnas:
                #
                # 0 webcam
                # 1-3 vecinos globales
                # 4-6 vecinos misma clase
                # --------------------------------------------------

                fig, axes = plt.subplots(
                    nrows=len(selected),
                    ncols=7,
                    figsize=(
                        21,
                        4.5 * len(selected)
                    )
                )

                if len(
                    selected
                ) == 1:

                    axes = np.expand_dims(
                        axes,
                        axis=0
                    )

                # ==================================================
                # CADA CAPTURA
                # ==================================================

                for row_index, (
                    _,
                    sample
                ) in enumerate(
                    selected.iterrows()
                ):

                    attempt = int(
                        sample[
                            "attempt"
                        ]
                    )

                    query_path = Path(
                        str(
                            sample[
                                "image_path"
                            ]
                        )
                    )

                    query_image = (
                        load_webcam_rgb(
                            query_path
                        )
                    )

                    bgr = cv2.imread(
                        str(
                            query_path
                        )
                    )

                    if bgr is None:

                        print(
                            f"  ⚠ No se pudo leer "
                            f"{query_path}"
                        )

                        continue

                    # ------------------------------------------
                    # Recalcular img_norm webcam
                    # ------------------------------------------

                    detection = (
                        detect_with_cascade(
                            landmarker,
                            bgr
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
                            f"  ⚠ Sin landmarks "
                            f"en {cls} #{attempt}"
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

                    # ==========================================
                    # COLUMNA 0
                    # WEBCAM
                    # ==========================================

                    pred = (
                        sample[
                            "fresh_prediction"
                        ]
                    )

                    distance_nn = float(
                        sample[
                            "nearest_neighbor_distance"
                        ]
                    )

                    title = (
                        f"WEBCAM {cls} #{attempt}\n"
                        f"A3 -> {pred}\n"
                        f"dNN={distance_nn:.2f}"
                    )

                    draw_image(
                        axes[
                            row_index,
                            0
                        ],
                        query_image,
                        title
                    )

                    # ==========================================
                    # 3 VECINOS GLOBALES
                    # ==========================================

                    overall = (
                        neighbors_df[
                            (
                                neighbors_df[
                                    "query_true_class"
                                ].astype(str)
                                == cls
                            )
                            &
                            (
                                neighbors_df[
                                    "query_attempt"
                                ].astype(int)
                                == attempt
                            )
                        ]
                        .sort_values(
                            "neighbor_rank"
                        )
                        .head(3)
                    )

                    for (
                        col_offset,
                        (_, neighbor)
                    ) in enumerate(
                        overall.iterrows(),
                        start=1
                    ):

                        stored_path = (
                            neighbor[
                                "neighbor_image_path"
                            ]
                        )

                        member = (
                            resolve_zip_member(
                                stored_path,
                                zip_exact_index,
                                zip_basename_index
                            )
                        )

                        image = (
                            load_zip_image_rgb(
                                zip_file,
                                member
                            )
                        )

                        neighbor_class = str(
                            neighbor[
                                "neighbor_class"
                            ]
                        )

                        distance = float(
                            neighbor[
                                "distance"
                            ]
                        )

                        title = (
                            f"GLOBAL #{int(neighbor['neighbor_rank'])}\n"
                            f"{neighbor_class}\n"
                            f"d={distance:.2f}"
                        )

                        draw_image(
                            axes[
                                row_index,
                                col_offset
                            ],
                            image,
                            title
                        )

                        if member is None:

                            missing_zip_images.append(
                                str(
                                    stored_path
                                )
                            )

                        visual_rows.append({
                            "true_class":
                                cls,

                            "query_attempt":
                                attempt,

                            "comparison_type":
                                "global",

                            "rank":
                                int(
                                    neighbor[
                                        "neighbor_rank"
                                    ]
                                ),

                            "neighbor_class":
                                neighbor_class,

                            "distance":
                                distance,

                            "stored_path":
                                str(
                                    stored_path
                                ),

                            "zip_member":
                                member,
                        })

                    # ==========================================
                    # 3 MÁS CERCANOS DE LA CLASE CORRECTA
                    # ==========================================

                    true_mask = (
                        train_df[
                            "class"
                        ]
                        .astype(str)
                        .eq(cls)
                        .to_numpy()
                    )

                    true_indices = np.where(
                        true_mask
                    )[0]

                    X_true = (
                        X_train_scaled[
                            true_indices
                        ]
                    )

                    differences = (
                        X_true
                        -
                        query_scaled.reshape(
                            1,
                            -1
                        )
                    )

                    distances = np.linalg.norm(
                        differences,
                        axis=1
                    )

                    nearest_local = np.argsort(
                        distances
                    )[:3]

                    for rank, local_index in enumerate(
                        nearest_local,
                        start=1
                    ):

                        global_index = int(
                            true_indices[
                                local_index
                            ]
                        )

                        train_row = (
                            train_df.iloc[
                                global_index
                            ]
                        )

                        distance = float(
                            distances[
                                local_index
                            ]
                        )

                        stored_path = (
                            train_row[
                                path_column
                            ]
                        )

                        member = (
                            resolve_zip_member(
                                stored_path,
                                zip_exact_index,
                                zip_basename_index
                            )
                        )

                        image = (
                            load_zip_image_rgb(
                                zip_file,
                                member
                            )
                        )

                        title = (
                            f"MISMA CLASE #{rank}\n"
                            f"{cls}\n"
                            f"d={distance:.2f}"
                        )

                        draw_image(
                            axes[
                                row_index,
                                3 + rank
                            ],
                            image,
                            title
                        )

                        if member is None:

                            missing_zip_images.append(
                                str(
                                    stored_path
                                )
                            )

                        visual_rows.append({
                            "true_class":
                                cls,

                            "query_attempt":
                                attempt,

                            "comparison_type":
                                "same_class",

                            "rank":
                                rank,

                            "neighbor_class":
                                cls,

                            "distance":
                                distance,

                            "stored_path":
                                str(
                                    stored_path
                                ),

                            "zip_member":
                                member,
                        })

                # --------------------------------------------------
                # TÍTULO GENERAL
                # --------------------------------------------------

                fig.suptitle(
                    (
                        f"ASL {cls} — Webcam vs TRAIN\n"
                        "Izquierda: webcam | "
                        "Centro: vecinos globales | "
                        "Derecha: muestras más cercanas "
                        "de la clase correcta"
                    ),
                    fontsize=15
                )

                plt.tight_layout(
                    rect=[
                        0,
                        0,
                        1,
                        0.94
                    ]
                )

                output_path = (
                    output_dir
                    / f"{cls}_visual_comparison.png"
                )

                plt.savefig(
                    output_path,
                    dpi=180,
                    bbox_inches="tight"
                )

                plt.close()

                print(
                    f"  ✅ {output_path.name}"
                )

        finally:

            landmarker.close()

    # ======================================================
    # GUARDAR DETALLE
    # ======================================================

    visual_df = pd.DataFrame(
        visual_rows
    )

    visual_csv = (
        output_dir
        / "visual_comparison_neighbors.csv"
    )

    visual_df.to_csv(
        visual_csv,
        index=False
    )

    # ------------------------------------------------------
    # RUTAS NO ENCONTRADAS
    # ------------------------------------------------------

    missing_zip_images = sorted(
        set(
            missing_zip_images
        )
    )

    if missing_zip_images:

        missing_path = (
            output_dir
            / "missing_zip_images.txt"
        )

        with open(
            missing_path,
            "w",
            encoding="utf-8"
        ) as f:

            for path in missing_zip_images:

                f.write(
                    path
                    + "\n"
                )

        print()
        print(
            "⚠ Algunas imágenes no pudieron "
            "resolverse dentro del ZIP:"
        )

        print(
            len(
                missing_zip_images
            )
        )

        print(
            f"Consulta: {missing_path}"
        )

    # ======================================================
    # FIN
    # ======================================================

    print()
    print(
        "============================================"
    )

    print(
        "=== TERMINADO ==="
    )

    print(
        "============================================"
    )

    print()

    print(
        "Abre:"
    )

    print(
        f"  {output_dir}"
    )

    print()

    print(
        "Debes encontrar:"
    )

    for cls in FOCUS_CLASSES:

        print(
            f"  {cls}_visual_comparison.png"
        )

    print()

    print(
        "Además:"
    )

    print(
        f"  {visual_csv.name}"
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Comparación visual entre capturas "
            "webcam y muestras ASL almacenadas "
            "directamente dentro del ZIP original."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_ASL_DATASET
        )
    )

    parser.add_argument(
        "--asl-zip",
        default=str(
            DEFAULT_ASL_ZIP
        )
    )

    parser.add_argument(
        "--diagnosis-root",
        default=str(
            DEFAULT_DIAGNOSIS_ROOT
        )
    )

    parser.add_argument(
        "--diagnosis-dir",
        default=None
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

    main(
        parser.parse_args()
    )
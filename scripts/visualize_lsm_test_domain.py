import argparse
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

FOCUS_CLASSES = [
    "G",
    "H",
    "Y",
    "S",
    "X",
    "T",
]


DEFAULT_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv"
)

DEFAULT_ZIP = Path(
    r"D:\Sign2SignData\raw\lsm\LSM_original.zip"
)

DEFAULT_OUTPUT = Path(
    r"D:\Sign2SignData\inventory\lsm_test_domain"
)


# ==========================================================
# COLUMNAS
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
            "Se esperaban exactamente "
            f"63 columnas img_norm, "
            f"pero se encontraron {len(columns)}."
        )

    return columns


# ==========================================================
# PATHS DENTRO DEL ZIP
# ==========================================================

def normalize_path(value):

    if pd.isna(value):
        return None

    value = str(value)

    value = value.replace(
        "\\",
        "/"
    )

    while value.startswith(
        "./"
    ):
        value = value[2:]

    return value.lstrip(
        "/"
    )


def build_zip_index(
    zip_file
):

    print()
    print(
        "Indexando contenido de LSM_original.zip..."
    )

    exact_index = {}
    basename_index = {}

    members = []

    for info in zip_file.infolist():

        if info.is_dir():
            continue

        member = info.filename

        members.append(
            member
        )

        normalized = normalize_path(
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
        "Archivos encontrados:",
        len(members)
    )

    print()
    print(
        "Ejemplos de rutas internas:"
    )

    for member in members[:10]:

        print(
            " ",
            member
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

    normalized = normalize_path(
        stored_path
    )

    if not normalized:
        return None

    lower = normalized.lower()

    # ------------------------------------------------------
    # 1. Exactamente igual
    # ------------------------------------------------------

    if lower in exact_index:

        return exact_index[
            lower
        ]

    # ------------------------------------------------------
    # 2. El ZIP puede tener carpetas adicionales antes
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
    # 3. Intentar desde TT/dataset/...
    # ------------------------------------------------------

    parts = normalized.split(
        "/"
    )

    lower_parts = [
        p.lower()
        for p in parts
    ]

    try:

        tt_index = (
            lower_parts.index(
                "tt"
            )
        )

        relative = "/".join(
            parts[
                tt_index:
            ]
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

    except ValueError:

        pass

    # ------------------------------------------------------
    # 4. Solo nombre de archivo
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


# ==========================================================
# LEER IMAGEN DEL ZIP
# ==========================================================

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
# DIBUJAR
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
            ha="center",
            va="center",
            fontsize=9
        )

        return

    ax.imshow(
        image
    )


# ==========================================================
# SELECCIONAR 3 EJEMPLOS REPRESENTATIVOS
# ==========================================================

def choose_representative_samples(
    test_class_df,
    distances
):

    working = (
        test_class_df.copy()
        .reset_index(
            drop=True
        )
    )

    working[
        "_nearest_train_distance"
    ] = distances

    working = (
        working
        .sort_values(
            "_nearest_train_distance"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        working
    ) <= 3:

        return working

    positions = [
        0,
        len(working) // 2,
        len(working) - 1
    ]

    return (
        working.iloc[
            positions
        ]
        .copy()
    )


# ==========================================================
# CENTROIDES
# ==========================================================

def build_centroids(
    X_train_scaled,
    y_train
):

    centroids = {}

    for cls in sorted(
        set(y_train)
    ):

        mask = (
            y_train
            == cls
        )

        centroids[
            cls
        ] = (
            X_train_scaled[
                mask
            ].mean(
                axis=0
            )
        )

    return centroids


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    dataset_path = Path(
        args.dataset
    )

    zip_path = Path(
        args.zip
    )

    output_root = Path(
        args.output
    )

    # ------------------------------------------------------
    # VALIDAR
    # ------------------------------------------------------

    if not dataset_path.exists():

        raise FileNotFoundError(
            "No existe dataset:\n"
            f"{dataset_path}"
        )

    if not zip_path.exists():

        raise FileNotFoundError(
            "No existe ZIP:\n"
            f"{zip_path}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== LSM TEST DOMAIN VISUAL AUDIT ==="
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
        "ZIP:"
    )

    print(
        f"  {zip_path}"
    )

    print()

    print(
        "Salida:"
    )

    print(
        f"  {output_root}"
    )

    # ------------------------------------------------------
    # CARGAR CSV
    # ------------------------------------------------------

    print()
    print(
        "Cargando landmarks..."
    )

    df = pd.read_csv(
        dataset_path,
        low_memory=False
    )

    required_columns = [
        "class",
        "path",
        "split_final",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise RuntimeError(
                f"Falta columna requerida: "
                f"{column}"
            )

    img_cols = (
        find_img_norm_columns(
            df
        )
    )

    # ------------------------------------------------------
    # TRAIN / TEST
    # ------------------------------------------------------

    train_df = (
        df[
            df[
                "split_final"
            ]
            .astype(str)
            .str.lower()
            .eq("train")
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_df = (
        df[
            df[
                "split_final"
            ]
            .astype(str)
            .str.lower()
            .eq("test")
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "TRAIN:",
        len(
            train_df
        )
    )

    print(
        "TEST:",
        len(
            test_df
        )
    )

    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    X_train = (
        train_df[
            img_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    X_test = (
        test_df[
            img_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        X_train
    ).all():

        raise RuntimeError(
            "TRAIN contiene NaN o Inf."
        )

    if not np.isfinite(
        X_test
    ).all():

        raise RuntimeError(
            "TEST contiene NaN o Inf."
        )

    y_train = (
        train_df[
            "class"
        ]
        .astype(str)
        .to_numpy()
    )

    y_test = (
        test_df[
            "class"
        ]
        .astype(str)
        .to_numpy()
    )

    # ------------------------------------------------------
    # STANDARD SCALER
    #
    # MISMO CRITERIO QUE COLAB:
    # FIT SOLO TRAIN
    # ------------------------------------------------------

    print()
    print(
        "Ajustando StandardScaler "
        "ÚNICAMENTE sobre TRAIN..."
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
        .astype(
            np.float32
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
        .astype(
            np.float32
        )
    )

    print(
        "✅ Scaler listo."
    )

    # ------------------------------------------------------
    # KNN GLOBAL
    # ------------------------------------------------------

    print()
    print(
        "Construyendo índice KNN de TRAIN..."
    )

    knn_global = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean",
        n_jobs=-1
    )

    knn_global.fit(
        X_train_scaled
    )

    print(
        "✅ KNN listo."
    )

    # ------------------------------------------------------
    # CENTROIDES
    # ------------------------------------------------------

    centroids = (
        build_centroids(
            X_train_scaled,
            y_train
        )
    )

    centroid_classes = sorted(
        centroids.keys()
    )

    centroid_matrix = np.vstack([
        centroids[
            cls
        ]
        for cls in centroid_classes
    ])

    # ------------------------------------------------------
    # ABRIR ZIP
    # ------------------------------------------------------

    visual_rows = []
    summary_rows = []
    missing_paths = []

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_file:

        (
            exact_index,
            basename_index
        ) = build_zip_index(
            zip_file
        )

        # ==================================================
        # POR CLASE
        # ==================================================

        for cls in FOCUS_CLASSES:

            print()
            print(
                "--------------------------------------------"
            )

            print(
                f"Procesando clase {cls}"
            )

            print(
                "--------------------------------------------"
            )

            test_mask = (
                y_test
                == cls
            )

            train_mask = (
                y_train
                == cls
            )

            test_indices = np.where(
                test_mask
            )[0]

            train_indices_same = np.where(
                train_mask
            )[0]

            if len(
                test_indices
            ) == 0:

                print(
                    "⚠ No existen muestras "
                    f"TEST para {cls}"
                )

                continue

            if len(
                train_indices_same
            ) == 0:

                print(
                    "⚠ No existen muestras "
                    f"TRAIN para {cls}"
                )

                continue

            X_test_cls = (
                X_test_scaled[
                    test_indices
                ]
            )

            # ----------------------------------------------
            # NN global para TODA la clase
            # ----------------------------------------------

            distances_all, indices_all = (
                knn_global.kneighbors(
                    X_test_cls,
                    n_neighbors=5
                )
            )

            nearest_classes_all = (
                y_train[
                    indices_all[
                        :,
                        0
                    ]
                ]
            )

            self_nn_rate = float(
                np.mean(
                    nearest_classes_all
                    == cls
                )
            )

            dominant_nn = (
                pd.Series(
                    nearest_classes_all
                )
                .value_counts()
                .index[0]
            )

            dominant_nn_count = int(
                pd.Series(
                    nearest_classes_all
                )
                .value_counts()
                .iloc[0]
            )

            # ----------------------------------------------
            # Centroide más cercano
            # ----------------------------------------------

            centroid_diff = (
                X_test_cls[
                    :,
                    None,
                    :
                ]
                -
                centroid_matrix[
                    None,
                    :,
                    :
                ]
            )

            centroid_distances = (
                np.linalg.norm(
                    centroid_diff,
                    axis=2
                )
            )

            nearest_centroid_indices = (
                np.argmin(
                    centroid_distances,
                    axis=1
                )
            )

            nearest_centroid_classes = np.array([
                centroid_classes[i]
                for i
                in nearest_centroid_indices
            ])

            dominant_centroid = (
                pd.Series(
                    nearest_centroid_classes
                )
                .value_counts()
                .index[0]
            )

            true_centroid_idx = (
                centroid_classes.index(
                    cls
                )
            )

            mean_true_centroid_distance = float(
                centroid_distances[
                    :,
                    true_centroid_idx
                ].mean()
            )

            # ----------------------------------------------
            # Resumen
            # ----------------------------------------------

            summary_rows.append({
                "class":
                    cls,

                "test_samples":
                    len(
                        test_indices
                    ),

                "nearest_neighbor_self_rate":
                    self_nn_rate,

                "dominant_nearest_neighbor":
                    dominant_nn,

                "dominant_nearest_neighbor_count":
                    dominant_nn_count,

                "mean_nearest_neighbor_distance":
                    float(
                        distances_all[
                            :,
                            0
                        ].mean()
                    ),

                "dominant_nearest_centroid":
                    dominant_centroid,

                "mean_true_centroid_distance":
                    mean_true_centroid_distance,
            })

            print(
                "TEST samples:",
                len(
                    test_indices
                )
            )

            print(
                "Nearest-neighbor "
                "same-class rate:",
                (
                    f"{self_nn_rate * 100:.2f}%"
                )
            )

            print(
                "Dominant NN:",
                dominant_nn,
                dominant_nn_count
            )

            print(
                "Dominant centroid:",
                dominant_centroid
            )

            # ----------------------------------------------
            # Elegir 3 capturas:
            # cercana / intermedia / lejana
            # ----------------------------------------------

            class_test_df = (
                test_df.iloc[
                    test_indices
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            selected = (
                choose_representative_samples(
                    class_test_df,
                    distances_all[
                        :,
                        0
                    ]
                )
            )

            # ----------------------------------------------
            # Figura
            # ----------------------------------------------

            fig, axes = plt.subplots(
                nrows=len(selected),
                ncols=7,
                figsize=(
                    21,
                    4.7
                    * len(selected)
                )
            )

            if len(
                selected
            ) == 1:

                axes = np.expand_dims(
                    axes,
                    axis=0
                )

            # ==============================================
            # CADA QUERY
            # ==============================================

            for row_idx, (
                _,
                query_row
            ) in enumerate(
                selected.iterrows()
            ):

                query_path = (
                    query_row[
                        "path"
                    ]
                )

                query_member = (
                    resolve_zip_member(
                        query_path,
                        exact_index,
                        basename_index
                    )
                )

                query_image = (
                    load_zip_image_rgb(
                        zip_file,
                        query_member
                    )
                )

                if query_member is None:

                    missing_paths.append(
                        str(
                            query_path
                        )
                    )

                # ------------------------------------------
                # Necesitamos localizar índice original
                # de esta fila dentro de TEST
                # ------------------------------------------

                candidate_matches = (
                    test_df[
                        "path"
                    ]
                    .astype(str)
                    .eq(
                        str(
                            query_path
                        )
                    )
                )

                query_global_test_index = (
                    np.where(
                        candidate_matches
                    )[0][0]
                )

                query_vector = (
                    X_test_scaled[
                        query_global_test_index
                    ]
                )

                # ------------------------------------------
                # KNN global
                # ------------------------------------------

                global_distances, global_indices = (
                    knn_global.kneighbors(
                        query_vector.reshape(
                            1,
                            -1
                        ),
                        n_neighbors=3
                    )
                )

                global_distances = (
                    global_distances[0]
                )

                global_indices = (
                    global_indices[0]
                )

                # ------------------------------------------
                # NN misma clase
                # ------------------------------------------

                X_train_same = (
                    X_train_scaled[
                        train_indices_same
                    ]
                )

                same_diff = (
                    X_train_same
                    -
                    query_vector.reshape(
                        1,
                        -1
                    )
                )

                same_distances = (
                    np.linalg.norm(
                        same_diff,
                        axis=1
                    )
                )

                same_local_indices = (
                    np.argsort(
                        same_distances
                    )[:3]
                )

                # ==========================================
                # COL 0: TEST
                # ==========================================

                group = (
                    query_row[
                        "group"
                    ]
                    if "group"
                    in query_row.index
                    else ""
                )

                session = (
                    query_row[
                        "session_hint"
                    ]
                    if "session_hint"
                    in query_row.index
                    else ""
                )

                test_title = (
                    f"TEST {cls}\n"
                    f"group={group}\n"
                    f"session={session}"
                )

                draw_image(
                    axes[
                        row_idx,
                        0
                    ],
                    query_image,
                    test_title
                )

                # ==========================================
                # COL 1-3: GLOBAL
                # ==========================================

                for rank, (
                    train_index,
                    distance
                ) in enumerate(
                    zip(
                        global_indices,
                        global_distances
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

                    train_path = (
                        train_row[
                            "path"
                        ]
                    )

                    member = (
                        resolve_zip_member(
                            train_path,
                            exact_index,
                            basename_index
                        )
                    )

                    image = (
                        load_zip_image_rgb(
                            zip_file,
                            member
                        )
                    )

                    if member is None:

                        missing_paths.append(
                            str(
                                train_path
                            )
                        )

                    neighbor_class = str(
                        train_row[
                            "class"
                        ]
                    )

                    title = (
                        f"GLOBAL #{rank}\n"
                        f"{neighbor_class}\n"
                        f"d={distance:.2f}"
                    )

                    draw_image(
                        axes[
                            row_idx,
                            rank
                        ],
                        image,
                        title
                    )

                    visual_rows.append({
                        "query_class":
                            cls,

                        "query_path":
                            str(
                                query_path
                            ),

                        "comparison_type":
                            "global",

                        "rank":
                            rank,

                        "neighbor_class":
                            neighbor_class,

                        "distance":
                            float(
                                distance
                            ),

                        "neighbor_path":
                            str(
                                train_path
                            ),

                        "query_group":
                            group,

                        "query_session":
                            session,
                    })

                # ==========================================
                # COL 4-6: MISMA CLASE
                # ==========================================

                for rank, local_idx in enumerate(
                    same_local_indices,
                    start=1
                ):

                    global_train_index = int(
                        train_indices_same[
                            local_idx
                        ]
                    )

                    train_row = (
                        train_df.iloc[
                            global_train_index
                        ]
                    )

                    distance = float(
                        same_distances[
                            local_idx
                        ]
                    )

                    train_path = (
                        train_row[
                            "path"
                        ]
                    )

                    member = (
                        resolve_zip_member(
                            train_path,
                            exact_index,
                            basename_index
                        )
                    )

                    image = (
                        load_zip_image_rgb(
                            zip_file,
                            member
                        )
                    )

                    if member is None:

                        missing_paths.append(
                            str(
                                train_path
                            )
                        )

                    train_group = (
                        train_row[
                            "group"
                        ]
                        if "group"
                        in train_row.index
                        else ""
                    )

                    title = (
                        f"MISMA CLASE #{rank}\n"
                        f"{cls}\n"
                        f"d={distance:.2f}"
                    )

                    draw_image(
                        axes[
                            row_idx,
                            3 + rank
                        ],
                        image,
                        title
                    )

                    visual_rows.append({
                        "query_class":
                            cls,

                        "query_path":
                            str(
                                query_path
                            ),

                        "comparison_type":
                            "same_class",

                        "rank":
                            rank,

                        "neighbor_class":
                            cls,

                        "distance":
                            distance,

                        "neighbor_path":
                            str(
                                train_path
                            ),

                        "query_group":
                            group,

                        "query_session":
                            session,

                        "neighbor_group":
                            train_group,
                    })

            # ----------------------------------------------
            # FIGURA
            # ----------------------------------------------

            fig.suptitle(
                (
                    f"LSM {cls} — TEST session 6 vs TRAIN\n"
                    "Izquierda: TEST | "
                    "Centro: vecinos globales | "
                    "Derecha: vecinos de la clase correcta"
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
                output_root
                / f"{cls}_test_vs_train.png"
            )

            plt.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            print(
                "✅",
                output_path.name
            )

    # ======================================================
    # GUARDAR CSV
    # ======================================================

    visual_df = pd.DataFrame(
        visual_rows
    )

    visual_csv = (
        output_root
        / "visual_neighbors.csv"
    )

    visual_df.to_csv(
        visual_csv,
        index=False
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        output_root
        / "domain_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    # ------------------------------------------------------
    # MISSING PATHS
    # ------------------------------------------------------

    missing_paths = sorted(
        set(
            missing_paths
        )
    )

    if missing_paths:

        missing_file = (
            output_root
            / "missing_images.txt"
        )

        with open(
            missing_file,
            "w",
            encoding="utf-8"
        ) as f:

            for path in missing_paths:

                f.write(
                    path
                    + "\n"
                )

        print()
        print(
            "⚠ Imágenes no resueltas:",
            len(
                missing_paths
            )
        )

        print(
            missing_file
        )

    # ======================================================
    # FIN
    # ======================================================

    print()
    print(
        "============================================"
    )

    print(
        "=== AUDITORÍA TERMINADA ==="
    )

    print(
        "============================================"
    )

    print()
    print(
        "Carpeta de resultados:"
    )

    print(
        output_root
    )

    print()
    print(
        "Archivos esperados:"
    )

    for cls in FOCUS_CLASSES:

        print(
            f"  {cls}_test_vs_train.png"
        )

    print()

    print(
        "  domain_summary.csv"
    )

    print(
        "  visual_neighbors.csv"
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Auditoría visual del domain shift "
            "LSM entre TRAIN y TEST."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_DATASET
        )
    )

    parser.add_argument(
        "--zip",
        default=str(
            DEFAULT_ZIP
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        )
    )

    main(
        parser.parse_args()
    )
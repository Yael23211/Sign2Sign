from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

DATASET_PATH = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv"
)

OUTPUT_DIR = Path(
    r"D:\Sign2SignData\inventory\lsm_group_audit"
)

GENUINE_CLASSES = [
    c
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if c not in {"E", "L", "Z"}
]


# ==========================================================
# UTILIDADES
# ==========================================================

def find_img_norm_columns(df):

    cols = [
        c
        for c in df.columns
        if str(c).startswith("img_norm_")
    ]

    if len(cols) != 63:
        raise RuntimeError(
            "Se esperaban 63 columnas img_norm, "
            f"se encontraron {len(cols)}."
        )

    return cols


def one_or_mixed(series):

    values = (
        series
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(values) == 0:
        return None

    if len(values) == 1:
        return values[0]

    return "MIXED:" + "|".join(
        sorted(values)
    )


def severity(
    nearest_any_class,
    true_class,
    same_class_rank,
    distance_ratio
):

    if pd.isna(
        same_class_rank
    ):
        return "NO_REFERENCE"

    if (
        nearest_any_class
        == true_class
    ):
        return "OK"

    # Heurística únicamente para priorizar revisión.
    # NO determina si una seña es lingüísticamente válida.

    if (
        same_class_rank > 3
        or (
            pd.notna(distance_ratio)
            and distance_ratio >= 1.5
        )
    ):
        return "STRONG_REVIEW"

    return "REVIEW"


# ==========================================================
# MAIN
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== AUDITORÍA GLOBAL DE GRUPOS LSM ==="
    )

    print(
        "============================================"
    )

    print()
    print(
        "Dataset:"
    )

    print(
        DATASET_PATH
    )

    # ------------------------------------------------------
    # DATASET
    # ------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    required = [
        "class",
        "group",
        "split_final"
    ]

    for col in required:

        if col not in df.columns:
            raise RuntimeError(
                f"Falta columna requerida: {col}"
            )

    img_cols = (
        find_img_norm_columns(
            df
        )
    )

    # ------------------------------------------------------
    # SOLO LSM GENUINA
    # ------------------------------------------------------

    audit_df = (
        df.copy()
    )

    if (
        "source_language"
        in audit_df.columns
    ):

        audit_df = audit_df[
            audit_df[
                "source_language"
            ]
            .astype(str)
            .str.upper()
            .eq("LSM")
        ].copy()

    if (
        "provisional_source"
        in audit_df.columns
    ):

        audit_df = audit_df[
            pd.to_numeric(
                audit_df[
                    "provisional_source"
                ],
                errors="coerce"
            )
            .fillna(0)
            .eq(0)
        ].copy()

    if (
        "use_letter_classifier"
        in audit_df.columns
    ):

        audit_df = audit_df[
            pd.to_numeric(
                audit_df[
                    "use_letter_classifier"
                ],
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
        ].copy()

    audit_df = audit_df[
        audit_df[
            "class"
        ].isin(
            GENUINE_CLASSES
        )
    ].copy()

    audit_df = (
        audit_df
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Muestras genuinas usadas:",
        len(audit_df)
    )

    print(
        "Clases:",
        audit_df[
            "class"
        ].nunique()
    )

    print(
        "Grupos:",
        audit_df[
            "group"
        ].nunique()
    )

    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    X_all = (
        audit_df[
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

    finite_mask = np.isfinite(
        X_all
    ).all(
        axis=1
    )

    if not finite_mask.all():

        print(
            "Eliminando filas no finitas:",
            int(
                (~finite_mask).sum()
            )
        )

        audit_df = (
            audit_df.iloc[
                np.where(
                    finite_mask
                )[0]
            ]
            .reset_index(
                drop=True
            )
        )

        X_all = (
            X_all[
                finite_mask
            ]
        )

    # ------------------------------------------------------
    # SCALER
    #
    # Para mantener el mismo espacio usado por L3:
    # FIT ÚNICAMENTE SOBRE EL TRAIN OFICIAL.
    # ------------------------------------------------------

    train_mask = (
        audit_df[
            "split_final"
        ]
        .astype(str)
        .str.lower()
        .eq("train")
        .to_numpy()
    )

    scaler = StandardScaler()

    scaler.fit(
        X_all[
            train_mask
        ]
    )

    X_scaled = (
        scaler.transform(
            X_all
        )
        .astype(
            np.float32
        )
    )

    print()
    print(
        "✅ StandardScaler ajustado "
        "sobre TRAIN oficial."
    )

    # ------------------------------------------------------
    # ID DE GRUPO ÚNICO
    # ------------------------------------------------------

    audit_df[
        "_group_id"
    ] = (
        audit_df[
            "class"
        ].astype(str)
        + "||"
        + audit_df[
            "group"
        ].astype(str)
    )

    group_ids = (
        audit_df[
            "_group_id"
        ]
        .unique()
        .tolist()
    )

    # ------------------------------------------------------
    # CENTROIDES DE GRUPO
    # ------------------------------------------------------

    centroid_rows = []
    centroid_vectors = []

    print()
    print(
        "Calculando centroides de grupos..."
    )

    for group_id in group_ids:

        mask = (
            audit_df[
                "_group_id"
            ]
            .eq(
                group_id
            )
            .to_numpy()
        )

        positions = np.where(
            mask
        )[0]

        part = (
            audit_df.iloc[
                positions
            ]
        )

        X_group = (
            X_scaled[
                positions
            ]
        )

        centroid = (
            X_group.mean(
                axis=0
            )
        )

        spread = float(
            np.linalg.norm(
                X_group
                - centroid.reshape(
                    1,
                    -1
                ),
                axis=1
            ).mean()
        )

        true_class = str(
            part[
                "class"
            ].iloc[0]
        )

        group_name = str(
            part[
                "group"
            ].iloc[0]
        )

        split = one_or_mixed(
            part[
                "split_final"
            ]
        )

        session = (
            one_or_mixed(
                part[
                    "session_hint"
                ]
            )
            if "session_hint"
            in part.columns
            else None
        )

        quality = (
            one_or_mixed(
                part[
                    "group_quality"
                ]
            )
            if "group_quality"
            in part.columns
            else None
        )

        centroid_rows.append({
            "group_id":
                group_id,

            "class":
                true_class,

            "group":
                group_name,

            "samples":
                len(
                    part
                ),

            "split_final":
                split,

            "session_hint":
                session,

            "group_quality":
                quality,

            "within_group_spread":
                spread,
        })

        centroid_vectors.append(
            centroid
        )

    centroids_df = pd.DataFrame(
        centroid_rows
    )

    centroid_matrix = np.vstack(
        centroid_vectors
    )

    # ------------------------------------------------------
    # COMPARAR GRUPO CONTRA TODOS LOS OTROS GRUPOS
    # ------------------------------------------------------

    audit_rows = []

    print(
        "Comparando grupos..."
    )

    for i in range(
        len(
            centroids_df
        )
    ):

        query = (
            centroid_matrix[i]
        )

        true_class = str(
            centroids_df.iloc[i][
                "class"
            ]
        )

        # Distancia a todos los centroides.
        distances = np.linalg.norm(
            centroid_matrix
            - query.reshape(
                1,
                -1
            ),
            axis=1
        )

        # Excluirse a sí mismo.
        distances[i] = np.inf

        order = np.argsort(
            distances
        )

        nearest_any_index = int(
            order[0]
        )

        nearest_any = (
            centroids_df.iloc[
                nearest_any_index
            ]
        )

        nearest_any_distance = float(
            distances[
                nearest_any_index
            ]
        )

        # ----------------------------------------------
        # BUSCAR PRIMER GRUPO DE LA MISMA CLASE
        # ----------------------------------------------

        same_class_index = None
        same_class_rank = None

        for rank, candidate_index in enumerate(
            order,
            start=1
        ):

            candidate_class = str(
                centroids_df.iloc[
                    int(
                        candidate_index
                    )
                ][
                    "class"
                ]
            )

            if (
                candidate_class
                == true_class
            ):

                same_class_index = int(
                    candidate_index
                )

                same_class_rank = int(
                    rank
                )

                break

        if (
            same_class_index
            is not None
        ):

            nearest_same = (
                centroids_df.iloc[
                    same_class_index
                ]
            )

            nearest_same_distance = float(
                distances[
                    same_class_index
                ]
            )

            if (
                nearest_any_distance
                > 0
            ):

                distance_ratio = (
                    nearest_same_distance
                    /
                    nearest_any_distance
                )

            else:

                distance_ratio = (
                    np.inf
                )

        else:

            nearest_same = None
            nearest_same_distance = np.nan
            distance_ratio = np.nan

        current = (
            centroids_df.iloc[i]
        )

        nearest_any_class = str(
            nearest_any[
                "class"
            ]
        )

        flag = severity(
            nearest_any_class,
            true_class,
            same_class_rank,
            distance_ratio
        )

        result = {
            "class":
                true_class,

            "group":
                current[
                    "group"
                ],

            "samples":
                int(
                    current[
                        "samples"
                    ]
                ),

            "split_final":
                current[
                    "split_final"
                ],

            "session_hint":
                current[
                    "session_hint"
                ],

            "group_quality":
                current[
                    "group_quality"
                ],

            "within_group_spread":
                float(
                    current[
                        "within_group_spread"
                    ]
                ),

            # Vecino global.
            "nearest_group":
                nearest_any[
                    "group"
                ],

            "nearest_group_class":
                nearest_any_class,

            "nearest_group_session":
                nearest_any[
                    "session_hint"
                ],

            "nearest_group_split":
                nearest_any[
                    "split_final"
                ],

            "nearest_group_distance":
                nearest_any_distance,

            # Vecino misma clase.
            "nearest_same_class_group":
                (
                    nearest_same[
                        "group"
                    ]
                    if nearest_same
                    is not None
                    else None
                ),

            "nearest_same_class_session":
                (
                    nearest_same[
                        "session_hint"
                    ]
                    if nearest_same
                    is not None
                    else None
                ),

            "nearest_same_class_split":
                (
                    nearest_same[
                        "split_final"
                    ]
                    if nearest_same
                    is not None
                    else None
                ),

            "nearest_same_class_distance":
                nearest_same_distance,

            "same_class_rank":
                same_class_rank,

            "same_class_to_global_ratio":
                distance_ratio,

            "nearest_group_is_same_class":
                (
                    nearest_any_class
                    == true_class
                ),

            "geometric_flag":
                flag,
        }

        audit_rows.append(
            result
        )

    result_df = pd.DataFrame(
        audit_rows
    )

    # ------------------------------------------------------
    # ORDEN PARA REVISIÓN
    # ------------------------------------------------------

    severity_order = {
        "STRONG_REVIEW": 0,
        "REVIEW": 1,
        "OK": 2,
        "NO_REFERENCE": 3,
    }

    result_df[
        "_severity_order"
    ] = (
        result_df[
            "geometric_flag"
        ].map(
            severity_order
        )
    )

    result_df = (
        result_df
        .sort_values(
            [
                "_severity_order",
                "same_class_rank",
                "same_class_to_global_ratio"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
        .drop(
            columns=[
                "_severity_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------
    # RESUMEN POR SESIÓN
    # ------------------------------------------------------

    session_rows = []

    for session, part in (
        result_df.groupby(
            "session_hint",
            dropna=False
        )
    ):

        strong = (
            part[
                part[
                    "geometric_flag"
                ]
                == "STRONG_REVIEW"
            ]
        )

        review = (
            part[
                part[
                    "geometric_flag"
                ]
                == "REVIEW"
            ]
        )

        ok = (
            part[
                part[
                    "geometric_flag"
                ]
                == "OK"
            ]
        )

        session_rows.append({
            "session_hint":
                session,

            "groups":
                len(
                    part
                ),

            "ok_groups":
                len(
                    ok
                ),

            "review_groups":
                len(
                    review
                ),

            "strong_review_groups":
                len(
                    strong
                ),

            "strong_review_classes":
                ",".join(
                    sorted(
                        strong[
                            "class"
                        ]
                        .astype(str)
                        .unique()
                    )
                ),

            "median_same_class_rank":
                part[
                    "same_class_rank"
                ].median(),

            "median_distance_ratio":
                part[
                    "same_class_to_global_ratio"
                ].replace(
                    [np.inf, -np.inf],
                    np.nan
                ).median(),
        })

    session_df = pd.DataFrame(
        session_rows
    )

    session_df = (
        session_df
        .sort_values(
            [
                "strong_review_groups",
                "review_groups"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------
    # RESUMEN POR CLASE
    # ------------------------------------------------------

    class_rows = []

    for cls, part in (
        result_df.groupby(
            "class"
        )
    ):

        strong = part[
            part[
                "geometric_flag"
            ]
            == "STRONG_REVIEW"
        ]

        class_rows.append({
            "class":
                cls,

            "groups":
                len(
                    part
                ),

            "sessions":
                ",".join(
                    sorted(
                        part[
                            "session_hint"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                ),

            "strong_review_groups":
                len(
                    strong
                ),

            "strong_review_group_names":
                ",".join(
                    strong[
                        "group"
                    ].astype(str)
                ),

            "median_same_class_rank":
                part[
                    "same_class_rank"
                ].median(),

            "median_distance_ratio":
                part[
                    "same_class_to_global_ratio"
                ].replace(
                    [np.inf, -np.inf],
                    np.nan
                ).median(),
        })

    class_df = pd.DataFrame(
        class_rows
    )

    class_df = (
        class_df
        .sort_values(
            "strong_review_groups",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------
    # MATRICES SESIÓN × CLASE
    # ------------------------------------------------------

    ratio_matrix = (
        result_df
        .pivot_table(
            index="session_hint",
            columns="class",
            values=(
                "same_class_to_global_ratio"
            ),
            aggfunc="median"
        )
    )

    rank_matrix = (
        result_df
        .pivot_table(
            index="session_hint",
            columns="class",
            values="same_class_rank",
            aggfunc="median"
        )
    )

    # ------------------------------------------------------
    # GUARDAR
    # ------------------------------------------------------

    all_path = (
        OUTPUT_DIR
        / "lsm_group_audit.csv"
    )

    flagged_path = (
        OUTPUT_DIR
        / "lsm_group_audit_flagged.csv"
    )

    session_path = (
        OUTPUT_DIR
        / "lsm_group_audit_by_session.csv"
    )

    class_path = (
        OUTPUT_DIR
        / "lsm_group_audit_by_class.csv"
    )

    ratio_path = (
        OUTPUT_DIR
        / "lsm_session_class_ratio.csv"
    )

    rank_path = (
        OUTPUT_DIR
        / "lsm_session_class_rank.csv"
    )

    result_df.to_csv(
        all_path,
        index=False
    )

    result_df[
        result_df[
            "geometric_flag"
        ]
        .isin(
            [
                "STRONG_REVIEW",
                "REVIEW"
            ]
        )
    ].to_csv(
        flagged_path,
        index=False
    )

    session_df.to_csv(
        session_path,
        index=False
    )

    class_df.to_csv(
        class_path,
        index=False
    )

    ratio_matrix.to_csv(
        ratio_path
    )

    rank_matrix.to_csv(
        rank_path
    )

    # ------------------------------------------------------
    # CONSOLA
    # ------------------------------------------------------

    print()
    print(
        "============================================"
    )

    print(
        "=== GRUPOS MÁS SOSPECHOSOS ==="
    )

    print(
        "============================================"
    )

    display_cols = [
        "class",
        "group",
        "session_hint",
        "split_final",
        "nearest_group_class",
        "nearest_group",
        "same_class_rank",
        "same_class_to_global_ratio",
        "geometric_flag",
    ]

    print(
        result_df[
            display_cols
        ]
        .head(40)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== RESUMEN POR SESIÓN ==="
    )

    print(
        "============================================"
    )

    print(
        session_df.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== RESUMEN POR CLASE ==="
    )

    print(
        "============================================"
    )

    print(
        class_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Archivos generados:"
    )

    for path in [
        all_path,
        flagged_path,
        session_path,
        class_path,
        ratio_path,
        rank_path,
    ]:

        print(
            " ",
            path
        )

    print()
    print(
        "✅ Auditoría global terminada."
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Los flags indican anomalías geométricas, "
        "NO determinan si una variante es "
        "lingüísticamente correcta o incorrecta."
    )


if __name__ == "__main__":

    main()
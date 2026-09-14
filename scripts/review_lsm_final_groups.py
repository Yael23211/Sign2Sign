import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

DATASET_PATH = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv"
)

ZIP_PATH = Path(
    r"D:\Sign2SignData\raw\lsm\LSM_original.zip"
)

OUTPUT_DIR = Path(
    r"D:\Sign2SignData\inventory\lsm_canonical_v2\final_group_review"
)


# ==========================================================
# GRUPOS QUE QUEREMOS REVISAR
#
# SIEMPRE:
#
#     (class, group)
#
# ==========================================================

TARGETS = [
    ("A", "PHONE_BURST"),
    ("P", "K7_FRAME"),
    ("Q", "Q_NORMAL"),
    ("K", "P_NORMAL"),
]


# ==========================================================
# FEATURES
# ==========================================================

def find_img_norm_columns(df):

    cols = [
        c
        for c in df.columns
        if str(c).startswith("img_norm_")
    ]

    if len(cols) != 63:

        raise RuntimeError(
            f"Se esperaban 63 columnas img_norm, "
            f"pero se encontraron {len(cols)}."
        )

    return cols


# ==========================================================
# PATHS ZIP
# ==========================================================

def normalize_path(value):

    if pd.isna(value):
        return None

    value = str(value).replace(
        "\\",
        "/"
    )

    while value.startswith("./"):
        value = value[2:]

    return value.lstrip("/")


def build_zip_index(zip_file):

    print()
    print("Indexando ZIP...")

    exact_index = {}
    basename_index = {}

    total = 0

    for info in zip_file.infolist():

        if info.is_dir():
            continue

        member = info.filename

        normalized = normalize_path(
            member
        )

        exact_index[
            normalized.lower()
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

        total += 1

    print(
        f"✅ Archivos indexados: {total}"
    )

    return (
        exact_index,
        basename_index
    )


def resolve_member(
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
    # 1. Coincidencia exacta
    # ------------------------------------------------------

    if lower in exact_index:

        return exact_index[
            lower
        ]

    # ------------------------------------------------------
    # 2. Coincidencia por sufijo
    # ------------------------------------------------------

    suffix = "/" + lower

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
    # 3. Intentar desde TT/
    # ------------------------------------------------------

    parts = normalized.split("/")

    lower_parts = [
        p.lower()
        for p in parts
    ]

    if "tt" in lower_parts:

        idx = lower_parts.index(
            "tt"
        )

        relative = "/".join(
            parts[idx:]
        ).lower()

        matches = [
            member
            for key, member
            in exact_index.items()
            if (
                key == relative
                or key.endswith(
                    "/" + relative
                )
            )
        ]

        if len(matches) == 1:

            return matches[0]

    # ------------------------------------------------------
    # 4. Solo filename
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
# CARGAR IMAGEN
# ==========================================================

def load_zip_image(
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
# MÁSCARA CLASS + GROUP
# ==========================================================

def group_mask(
    df,
    class_name,
    group_name
):

    return (
        df[
            "class"
        ]
        .astype(str)
        .eq(
            str(
                class_name
            )
        )
        &
        df[
            "group"
        ]
        .astype(str)
        .eq(
            str(
                group_name
            )
        )
    ).to_numpy()


# ==========================================================
# REPRESENTANTE / CENTROIDE
# ==========================================================

def representative_and_centroid(
    X_scaled,
    positions
):

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

    distances = np.linalg.norm(
        X_group
        - centroid.reshape(
            1,
            -1
        ),
        axis=1
    )

    local_idx = int(
        np.argmin(
            distances
        )
    )

    global_idx = int(
        positions[
            local_idx
        ]
    )

    return (
        global_idx,
        centroid
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
        fontsize=8
    )

    if image is None:

        ax.text(
            0.5,
            0.5,
            "Imagen no encontrada",
            ha="center",
            va="center"
        )

        return

    ax.imshow(
        image
    )


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
        "=== LSM FINAL GROUP REVIEW ==="
    )

    print(
        "============================================"
    )

    # ======================================================
    # CARGAR DATASET COMPLETO
    # ======================================================

    df_all = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    img_cols = (
        find_img_norm_columns(
            df_all
        )
    )

    print()
    print(
        "Filas originales:",
        len(
            df_all
        )
    )

    # ------------------------------------------------------
    # Solo filas utilizables
    # ------------------------------------------------------

    if "use_letter_classifier" in df_all.columns:

        classifier_mask = (
            pd.to_numeric(
                df_all[
                    "use_letter_classifier"
                ],
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
            .to_numpy()
        )

    else:

        classifier_mask = np.ones(
            len(
                df_all
            ),
            dtype=bool
        )

    X_all = (
        df_all[
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

    finite_mask = (
        np.isfinite(
            X_all
        )
        .all(
            axis=1
        )
    )

    usable_mask = (
        classifier_mask
        &
        finite_mask
    )

    df_model = (
        df_all.iloc[
            np.where(
                usable_mask
            )[0]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    X_model = (
        X_all[
            usable_mask
        ]
    )

    # ======================================================
    # SCALER
    #
    # MISMO CRITERIO QUE L3:
    # FIT SOBRE TODO TRAIN.
    # ======================================================

    train_mask = (
        df_model[
            "split_final"
        ]
        .astype(str)
        .str.lower()
        .eq("train")
        .to_numpy()
    )

    scaler = (
        StandardScaler()
    )

    scaler.fit(
        X_model[
            train_mask
        ]
    )

    X_model_scaled = (
        scaler.transform(
            X_model
        )
        .astype(
            np.float32
        )
    )

    print(
        "TRAIN usado para scaler:",
        int(
            train_mask.sum()
        )
    )

    # ======================================================
    # SOLO LSM GENUINA PARA AUDITORÍA
    # ======================================================

    audit_mask = np.ones(
        len(
            df_model
        ),
        dtype=bool
    )

    if "source_language" in df_model.columns:

        audit_mask &= (
            df_model[
                "source_language"
            ]
            .astype(str)
            .str.upper()
            .eq("LSM")
            .to_numpy()
        )

    if "provisional_source" in df_model.columns:

        audit_mask &= (
            pd.to_numeric(
                df_model[
                    "provisional_source"
                ],
                errors="coerce"
            )
            .fillna(0)
            .eq(0)
            .to_numpy()
        )

    audit_positions = np.where(
        audit_mask
    )[0]

    audit_df = (
        df_model.iloc[
            audit_positions
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    X_audit_scaled = (
        X_model_scaled[
            audit_positions
        ]
    )

    print(
        "LSM genuina:",
        len(
            audit_df
        )
    )

    # ======================================================
    # CREAR CATÁLOGO DE CENTROIDES
    # ======================================================

    group_rows = []
    group_centroids = []

    grouped = (
        audit_df.groupby(
            [
                "class",
                "group"
            ],
            sort=True
        )
    )

    print()
    print(
        "Calculando centroides..."
    )

    for (
        class_name,
        group_name
    ), part in grouped:

        mask = (
            group_mask(
                audit_df,
                class_name,
                group_name
            )
        )

        positions = np.where(
            mask
        )[0]

        (
            representative_idx,
            centroid
        ) = (
            representative_and_centroid(
                X_audit_scaled,
                positions
            )
        )

        representative_row = (
            audit_df.iloc[
                representative_idx
            ]
        )

        session = (
            str(
                representative_row[
                    "session_hint"
                ]
            )
            if "session_hint"
            in representative_row.index
            else ""
        )

        group_rows.append({
            "class":
                str(
                    class_name
                ),

            "group":
                str(
                    group_name
                ),

            "samples":
                len(
                    positions
                ),

            "session":
                session,

            "representative_index":
                representative_idx,
        })

        group_centroids.append(
            centroid
        )

    groups_df = pd.DataFrame(
        group_rows
    )

    centroid_matrix = np.vstack(
        group_centroids
    )

    print(
        "Grupos:",
        len(
            groups_df
        )
    )

    # ======================================================
    # ZIP
    # ======================================================

    summary_rows = []
    missing_rows = []

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_file:

        (
            exact_index,
            basename_index
        ) = build_zip_index(
            zip_file
        )

        # ==================================================
        # FIGURA
        # ==================================================

        fig, axes = plt.subplots(
            nrows=len(TARGETS),
            ncols=7,
            figsize=(
                22,
                4.2 * len(TARGETS)
            )
        )

        # ==================================================
        # CADA TARGET
        # ==================================================

        for row_idx, (
            true_class,
            target_group
        ) in enumerate(
            TARGETS
        ):

            print()
            print(
                "--------------------------------------------"
            )

            print(
                f"Revisando: "
                f"{true_class} | {target_group}"
            )

            # ----------------------------------------------
            # Encontrar grupo target
            # ----------------------------------------------

            target_matches = (
                groups_df[
                    groups_df[
                        "class"
                    ]
                    .eq(
                        true_class
                    )
                    &
                    groups_df[
                        "group"
                    ]
                    .eq(
                        target_group
                    )
                ]
            )

            if len(
                target_matches
            ) != 1:

                print(
                    "⚠ Target no encontrado de forma única."
                )

                continue

            target_group_index = int(
                target_matches.index[0]
            )

            target_meta = (
                groups_df.loc[
                    target_group_index
                ]
            )

            target_centroid = (
                centroid_matrix[
                    target_group_index
                ]
            )

            target_rep_idx = int(
                target_meta[
                    "representative_index"
                ]
            )

            target_row = (
                audit_df.iloc[
                    target_rep_idx
                ]
            )

            target_member = (
                resolve_member(
                    target_row[
                        "path"
                    ],
                    exact_index,
                    basename_index
                )
            )

            target_image = (
                load_zip_image(
                    zip_file,
                    target_member
                )
            )

            if target_member is None:

                missing_rows.append({
                    "class":
                        true_class,

                    "group":
                        target_group,

                    "type":
                        "target",

                    "path":
                        str(
                            target_row[
                                "path"
                            ]
                        ),
                })

            # ==============================================
            # COL 0 — TARGET
            # ==============================================

            draw_image(
                axes[
                    row_idx,
                    0
                ],
                target_image,
                (
                    "OBJETIVO\n"
                    f"class={true_class}\n"
                    f"group={target_group}\n"
                    f"session={target_meta['session']}"
                )
            )

            # ==============================================
            # DISTANCIA A TODOS LOS GRUPOS
            # ==============================================

            distances = np.linalg.norm(
                centroid_matrix
                - target_centroid.reshape(
                    1,
                    -1
                ),
                axis=1
            )

            distances[
                target_group_index
            ] = np.inf

            order = np.argsort(
                distances
            )

            # ==============================================
            # 3 GLOBALES
            # ==============================================

            global_indices = (
                order[:3]
            )

            # ==============================================
            # 3 MISMA CLASE
            # ==============================================

            same_class_indices = [
                int(idx)
                for idx in order
                if (
                    groups_df.iloc[
                        int(idx)
                    ][
                        "class"
                    ]
                    == true_class
                )
            ][:3]

            # ==============================================
            # DIBUJAR GLOBAL 1–3
            # ==============================================

            for local_rank, group_idx in enumerate(
                global_indices,
                start=1
            ):

                group_idx = int(
                    group_idx
                )

                meta = (
                    groups_df.iloc[
                        group_idx
                    ]
                )

                rep_idx = int(
                    meta[
                        "representative_index"
                    ]
                )

                row = (
                    audit_df.iloc[
                        rep_idx
                    ]
                )

                member = (
                    resolve_member(
                        row[
                            "path"
                        ],
                        exact_index,
                        basename_index
                    )
                )

                image = (
                    load_zip_image(
                        zip_file,
                        member
                    )
                )

                distance = float(
                    distances[
                        group_idx
                    ]
                )

                draw_image(
                    axes[
                        row_idx,
                        local_rank
                    ],
                    image,
                    (
                        f"GLOBAL #{local_rank}\n"
                        f"class={meta['class']}\n"
                        f"group={meta['group']}\n"
                        f"d={distance:.2f}"
                    )
                )

                summary_rows.append({
                    "target_class":
                        true_class,

                    "target_group":
                        target_group,

                    "comparison_type":
                        "GLOBAL",

                    "rank":
                        local_rank,

                    "neighbor_class":
                        meta[
                            "class"
                        ],

                    "neighbor_group":
                        meta[
                            "group"
                        ],

                    "neighbor_session":
                        meta[
                            "session"
                        ],

                    "distance":
                        distance,
                })

            # ==============================================
            # DIBUJAR MISMA CLASE 1–3
            # ==============================================

            for local_rank, group_idx in enumerate(
                same_class_indices,
                start=1
            ):

                group_idx = int(
                    group_idx
                )

                meta = (
                    groups_df.iloc[
                        group_idx
                    ]
                )

                rep_idx = int(
                    meta[
                        "representative_index"
                    ]
                )

                row = (
                    audit_df.iloc[
                        rep_idx
                    ]
                )

                member = (
                    resolve_member(
                        row[
                            "path"
                        ],
                        exact_index,
                        basename_index
                    )
                )

                image = (
                    load_zip_image(
                        zip_file,
                        member
                    )
                )

                distance = float(
                    distances[
                        group_idx
                    ]
                )

                draw_image(
                    axes[
                        row_idx,
                        3 + local_rank
                    ],
                    image,
                    (
                        f"MISMA CLASE #{local_rank}\n"
                        f"class={meta['class']}\n"
                        f"group={meta['group']}\n"
                        f"d={distance:.2f}"
                    )
                )

                summary_rows.append({
                    "target_class":
                        true_class,

                    "target_group":
                        target_group,

                    "comparison_type":
                        "SAME_CLASS",

                    "rank":
                        local_rank,

                    "neighbor_class":
                        meta[
                            "class"
                        ],

                    "neighbor_group":
                        meta[
                            "group"
                        ],

                    "neighbor_session":
                        meta[
                            "session"
                        ],

                    "distance":
                        distance,
                })

            # ==============================================
            # CONSOLA
            # ==============================================

            print()
            print(
                "Vecinos globales:"
            )

            for rank, idx in enumerate(
                global_indices,
                start=1
            ):

                meta = groups_df.iloc[
                    int(idx)
                ]

                print(
                    f"  #{rank} "
                    f"{meta['class']} | "
                    f"{meta['group']} | "
                    f"d={distances[int(idx)]:.3f}"
                )

            print()
            print(
                "Vecinos de la misma clase:"
            )

            for rank, idx in enumerate(
                same_class_indices,
                start=1
            ):

                meta = groups_df.iloc[
                    int(idx)
                ]

                print(
                    f"  #{rank} "
                    f"{meta['class']} | "
                    f"{meta['group']} | "
                    f"d={distances[int(idx)]:.3f}"
                )

        # ==================================================
        # GUARDAR FIGURA
        # ==================================================

        fig.suptitle(
            (
                "LSM Canonical V2 — Revisión final de grupos\n"
                "Izquierda: objetivo | "
                "Centro: vecinos globales | "
                "Derecha: vecinos de la misma clase"
            ),
            fontsize=15
        )

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96
            ]
        )

        figure_path = (
            OUTPUT_DIR
            / "final_groups_review.png"
        )

        plt.savefig(
            figure_path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # CSV
    # ======================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "final_groups_review.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================================
    # MISSING
    # ======================================================

    if missing_rows:

        missing_df = pd.DataFrame(
            missing_rows
        )

        missing_path = (
            OUTPUT_DIR
            / "missing_images.csv"
        )

        missing_df.to_csv(
            missing_path,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "⚠ Imágenes faltantes:",
            len(
                missing_df
            )
        )

    # ======================================================
    # FIN
    # ======================================================

    print()
    print(
        "============================================"
    )

    print(
        "=== REVISIÓN TERMINADA ==="
    )

    print(
        "============================================"
    )

    print()
    print(
        "Abre:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Archivos:"
    )

    print(
        " final_groups_review.png"
    )

    print(
        " final_groups_review.csv"
    )


if __name__ == "__main__":

    main()
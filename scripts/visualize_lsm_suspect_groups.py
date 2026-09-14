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
    r"D:\Sign2SignData\inventory\lsm_group_audit\visual_review_fixed"
)


# ==========================================================
# OBJETIVOS
#
# IMPORTANTE:
# Ahora cada objetivo se identifica por:
#
#     (class, group)
#
# y NO solamente por "group".
#
# Esto evita mezclar:
#
# G || GENERIC_FRAME
# H || GENERIC_FRAME
# Y || GENERIC_FRAME
#
# etc.
# ==========================================================

SESSION6_TARGETS = [
    ("G", "FRAM_G6"),
    ("H", "FRAM_H6"),
    ("Y", "FRAM_Y6"),
]


NORMAL_TARGETS = [
    ("O", "O_NORMAL"),
    ("Q", "Q_NORMAL"),
    ("R", "R_NORMAL"),
    ("S", "S_NORMAL"),
    ("T", "T_NORMAL"),
    ("U", "U_NORMAL"),
    ("V", "V_NORMAL"),
    ("W", "W_NORMAL"),

    # OJO:
    # En el CSV actual este grupo aparece
    # asociado a class=K.
    #
    # Esto conserva exactamente lo que
    # tenemos en el manifiesto actual.
    ("K", "P_NORMAL"),
]


# ==========================================================
# COLUMNAS IMG_NORM
# ==========================================================

def find_img_norm_columns(df):

    cols = [
        c
        for c in df.columns
        if str(c).startswith("img_norm_")
    ]

    if len(cols) != 63:

        raise RuntimeError(
            "Se esperaban 63 columnas img_norm "
            f"y se encontraron {len(cols)}."
        )

    return cols


# ==========================================================
# PATHS DEL ZIP
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
    # 1. Exacto
    # ------------------------------------------------------

    if lower in exact_index:

        return exact_index[
            lower
        ]

    # ------------------------------------------------------
    # 2. Por sufijo
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
    # 4. Filename
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
# LEER IMAGEN
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

    arr = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if image is None:

        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ==========================================================
# REPRESENTANTE DEL GRUPO
# ==========================================================

def representative_index(
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
# DIBUJAR
# ==========================================================

def draw_image(
    ax,
    image,
    title
):

    ax.axis("off")

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
            va="center"
        )

        return

    ax.imshow(
        image
    )


# ==========================================================
# MÁSCARA EXACTA:
#
# class + group
#
# ÉSTA ES LA CORRECCIÓN PRINCIPAL.
# ==========================================================

def exact_group_mask(
    df,
    class_name,
    group_name
):

    return (
        df["class"]
        .astype(str)
        .eq(
            str(
                class_name
            )
        )
        &
        df["group"]
        .astype(str)
        .eq(
            str(
                group_name
            )
        )
    ).to_numpy()


# ==========================================================
# CREAR COMPARACIÓN
# ==========================================================

def create_comparison(
    targets,
    output_name,
    audit_df,
    X_audit_scaled,
    zip_file,
    exact_index,
    basename_index,
    summary_rows,
    missing_rows
):

    valid_targets = []

    # ------------------------------------------------------
    # VALIDAR OBJETIVOS
    # ------------------------------------------------------

    for true_class, target_group in targets:

        mask = exact_group_mask(
            audit_df,
            true_class,
            target_group
        )

        count = int(
            mask.sum()
        )

        if count == 0:

            print(
                "⚠ No encontrado:"
            )

            print(
                f"   class={true_class}, "
                f"group={target_group}"
            )

            continue

        valid_targets.append(
            (
                true_class,
                target_group
            )
        )

    if not valid_targets:

        print(
            "No hay objetivos válidos para "
            f"{output_name}"
        )

        return

    # ------------------------------------------------------
    # FIGURA
    # ------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=len(valid_targets),
        ncols=4,
        figsize=(
            16,
            3.8 * len(valid_targets)
        )
    )

    if len(valid_targets) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    # ======================================================
    # CADA OBJETIVO
    # ======================================================

    for row_idx, (
        true_class,
        target_group
    ) in enumerate(
        valid_targets
    ):

        print()
        print(
            "----------------------------------------"
        )

        print(
            f"Objetivo: {true_class} | {target_group}"
        )

        # --------------------------------------------------
        # TARGET:
        #
        # class == true_class
        # AND
        # group == target_group
        # --------------------------------------------------

        target_mask = exact_group_mask(
            audit_df,
            true_class,
            target_group
        )

        target_positions = np.where(
            target_mask
        )[0]

        target_part = (
            audit_df.iloc[
                target_positions
            ]
        )

        target_idx, target_centroid = (
            representative_index(
                X_audit_scaled,
                target_positions
            )
        )

        target_row = (
            audit_df.iloc[
                target_idx
            ]
        )

        target_session = (
            str(
                target_row[
                    "session_hint"
                ]
            )
            if "session_hint"
            in target_row.index
            else ""
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

        # ==================================================
        # COL 0
        # ==================================================

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
                f"session={target_session}"
            )
        )

        # --------------------------------------------------
        # TODOS LOS GRUPOS DE ESA MISMA CLASE
        # --------------------------------------------------

        same_class_groups = (
            audit_df[
                audit_df[
                    "class"
                ]
                .astype(str)
                .eq(
                    true_class
                )
            ][
                "group"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        candidates = []

        for candidate_group in (
            same_class_groups
        ):

            # No comparar consigo mismo
            if (
                candidate_group
                == target_group
            ):

                continue

            # ==============================================
            # CORRECCIÓN:
            #
            # antes:
            #     group == candidate_group
            #
            # ahora:
            #
            #     class == true_class
            #     AND
            #     group == candidate_group
            #
            # ==============================================

            candidate_mask = (
                exact_group_mask(
                    audit_df,
                    true_class,
                    candidate_group
                )
            )

            candidate_positions = np.where(
                candidate_mask
            )[0]

            if len(
                candidate_positions
            ) == 0:

                continue

            (
                candidate_idx,
                candidate_centroid
            ) = representative_index(
                X_audit_scaled,
                candidate_positions
            )

            distance = float(
                np.linalg.norm(
                    target_centroid
                    - candidate_centroid
                )
            )

            candidate_row = (
                audit_df.iloc[
                    candidate_idx
                ]
            )

            candidate_session = (
                str(
                    candidate_row[
                        "session_hint"
                    ]
                )
                if "session_hint"
                in candidate_row.index
                else ""
            )

            candidates.append({
                "group":
                    candidate_group,

                "class":
                    true_class,

                "distance":
                    distance,

                "representative_index":
                    candidate_idx,

                "session":
                    candidate_session,
            })

        # --------------------------------------------------
        # ORDENAR
        # --------------------------------------------------

        candidates.sort(
            key=lambda item:
            item[
                "distance"
            ]
        )

        selected = (
            candidates[:3]
        )

        # --------------------------------------------------
        # CONSOLA
        # --------------------------------------------------

        for i, candidate in enumerate(
            selected,
            start=1
        ):

            print(
                f"  #{i}: "
                f"{candidate['class']} | "
                f"{candidate['group']} | "
                f"session={candidate['session']} | "
                f"d={candidate['distance']:.3f}"
            )

        # ==================================================
        # COL 1-3
        # ==================================================

        for col_idx in range(3):

            ax = axes[
                row_idx,
                col_idx + 1
            ]

            if col_idx >= len(
                selected
            ):

                draw_image(
                    ax,
                    None,
                    "Sin comparación"
                )

                continue

            candidate = (
                selected[
                    col_idx
                ]
            )

            candidate_idx = int(
                candidate[
                    "representative_index"
                ]
            )

            candidate_row = (
                audit_df.iloc[
                    candidate_idx
                ]
            )

            member = (
                resolve_member(
                    candidate_row[
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

            if member is None:

                missing_rows.append({
                    "class":
                        true_class,

                    "group":
                        candidate[
                            "group"
                        ],

                    "type":
                        "reference",

                    "path":
                        str(
                            candidate_row[
                                "path"
                            ]
                        ),
                })

            draw_image(
                ax,
                image,
                (
                    f"MISMA CLASE #{col_idx + 1}\n"
                    f"class={true_class}\n"
                    f"group={candidate['group']}\n"
                    f"session={candidate['session']}\n"
                    f"d={candidate['distance']:.2f}"
                )
            )

            summary_rows.append({
                "target_class":
                    true_class,

                "target_group":
                    target_group,

                "target_session":
                    target_session,

                "reference_rank":
                    col_idx + 1,

                "reference_class":
                    true_class,

                "reference_group":
                    candidate[
                        "group"
                    ],

                "reference_session":
                    candidate[
                        "session"
                    ],

                "centroid_distance":
                    candidate[
                        "distance"
                    ],
            })

    # ------------------------------------------------------
    # GUARDAR FIGURA
    # ------------------------------------------------------

    fig.suptitle(
        (
            "LSM — Auditoría visual corregida\n"
            "Izquierda: grupo objetivo | "
            "Derecha: grupos más cercanos "
            "de LA MISMA CLASE"
        ),
        fontsize=15
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        f"✅ {output_path}"
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
        "=== LSM VISUAL GROUP REVIEW - FIXED ==="
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
    # SOLO FILAS UTILIZABLES POR EL CLASIFICADOR
    # ------------------------------------------------------

    if (
        "use_letter_classifier"
        in df_all.columns
    ):

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

    # ------------------------------------------------------
    # FEATURES COMPLETAS
    # ------------------------------------------------------

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

    finite_mask = np.isfinite(
        X_all
    ).all(
        axis=1
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

    print(
        "Filas utilizables:",
        len(
            df_model
        )
    )

    # ======================================================
    # SCALER
    #
    # EXACTAMENTE EL CRITERIO DE L3:
    #
    # FIT SOBRE TODO TRAIN,
    # INCLUYENDO E/L/Z PROVISIONALES.
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

    print(
        "TRAIN usado para scaler:",
        int(
            train_mask.sum()
        )
    )

    scaler = StandardScaler()

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
        "✅ StandardScaler ajustado "
        "sobre TRAIN completo."
    )

    # ======================================================
    # AUDITORÍA:
    #
    # SOLO LSM GENUINA
    # SIN E/L/Z PROVISIONALES.
    # ======================================================

    audit_mask = np.ones(
        len(
            df_model
        ),
        dtype=bool
    )

    if (
        "source_language"
        in df_model.columns
    ):

        audit_mask &= (
            df_model[
                "source_language"
            ]
            .astype(str)
            .str.upper()
            .eq("LSM")
            .to_numpy()
        )

    if (
        "provisional_source"
        in df_model.columns
    ):

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
        "LSM genuina para auditoría:",
        len(
            audit_df
        )
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

        # --------------------------------------------------
        # G6/H6/Y6
        # --------------------------------------------------

        create_comparison(
            SESSION6_TARGETS,
            "session6_extreme_comparison_FIXED.png",
            audit_df,
            X_audit_scaled,
            zip_file,
            exact_index,
            basename_index,
            summary_rows,
            missing_rows
        )

        # --------------------------------------------------
        # NORMAL
        # --------------------------------------------------

        create_comparison(
            NORMAL_TARGETS,
            "normal_groups_comparison_FIXED.png",
            audit_df,
            X_audit_scaled,
            zip_file,
            exact_index,
            basename_index,
            summary_rows,
            missing_rows
        )

    # ======================================================
    # CSV RESUMEN
    # ======================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "visual_group_review_summary_FIXED.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
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
            / "missing_images_FIXED.csv"
        )

        missing_df.to_csv(
            missing_path,
            index=False
        )

        print()
        print(
            "⚠ Imágenes no resueltas:",
            len(
                missing_df
            )
        )

        print(
            missing_path
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
        OUTPUT_DIR
    )

    print()

    print(
        "Archivos principales:"
    )

    print(
        " session6_extreme_comparison_FIXED.png"
    )

    print(
        " normal_groups_comparison_FIXED.png"
    )

    print(
        " visual_group_review_summary_FIXED.csv"
    )


if __name__ == "__main__":

    main()
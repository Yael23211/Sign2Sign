from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

SOURCE_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks"
    r"\lsm_landmarks_canonical_v2.csv"
)

OUTPUT_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks"
    r"\lsm_landmarks_canonical_v2_splits.csv"
)

OUTPUT_DIR = Path(
    r"D:\Sign2SignData\inventory\lsm_canonical_v2\splits"
)

GROUP_ASSIGNMENTS_PATH = (
    OUTPUT_DIR
    / "lsm_canonical_v2_group_assignments.csv"
)

CLASS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "lsm_canonical_v2_split_by_class.csv"
)

SPLIT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "lsm_canonical_v2_split_summary.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "lsm_canonical_v2_split_audit.csv"
)


# ==========================================================
# POLÍTICA DE SPLIT V2
# ==========================================================

VALIDATION_SESSION = "3"
TEST_SESSION = "5"

PROVISIONAL_CLASSES = {
    "E",
    "L",
    "Z",
}

GENUINE_CLASSES = {
    letter
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if letter not in PROVISIONAL_CLASSES
}


# ==========================================================
# UTILIDADES
# ==========================================================

def separator():
    print("=" * 72)


def normalize_session(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    if text.lower() == "nan":
        return ""

    # Por si pandas leyó:
    #
    # 3.0
    # 5.0
    #
    try:
        number = float(text)

        if number.is_integer():
            return str(
                int(number)
            )

    except ValueError:
        pass

    return text.upper()


def unique_nonempty_sessions(series):

    values = {
        normalize_session(value)
        for value in series
    }

    values.discard("")

    return sorted(values)


def provisional_mask(df):

    return (
        df["class"]
        .astype(str)
        .str.upper()
        .isin(PROVISIONAL_CLASSES)
    )


# ==========================================================
# CARGAR
# ==========================================================

def load_dataset():

    if not SOURCE_DATASET.exists():

        raise FileNotFoundError(
            "No se encontró:\n"
            f"{SOURCE_DATASET}"
        )

    df = pd.read_csv(
        SOURCE_DATASET,
        low_memory=False
    )

    required = [
        "class",
        "group",
        "session_hint",
        "split_final",
        "split_constraint",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Faltan columnas:\n"
            + "\n".join(
                missing
            )
        )

    return df


# ==========================================================
# CONSTRUIR ASIGNACIÓN POR GRUPO
#
# La identidad del grupo es:
#
#     class + group
#
# Nunca usamos group por sí solo.
# ==========================================================

def build_group_assignments(df):

    rows = []

    grouped = df.groupby(
        [
            "class",
            "group"
        ],
        sort=True,
        dropna=False
    )

    for (
        class_name,
        group_name
    ), part in grouped:

        class_name = str(
            class_name
        ).upper()

        group_name = str(
            group_name
        )

        sessions = (
            unique_nonempty_sessions(
                part["session_hint"]
            )
        )

        # --------------------------------------------------
        # Las provisionales SIEMPRE van a train.
        # --------------------------------------------------

        if (
            class_name
            in PROVISIONAL_CLASSES
        ):

            split = "train"

            reason = (
                "Clase provisional E/L/Z: "
                "TRAIN_ONLY por diseño."
            )

        else:

            # ----------------------------------------------
            # Validar que un mismo grupo no mezcle
            # simultáneamente session 3 y session 5.
            # ----------------------------------------------

            has_val = (
                VALIDATION_SESSION
                in sessions
            )

            has_test = (
                TEST_SESSION
                in sessions
            )

            if (
                has_val
                and has_test
            ):

                raise RuntimeError(
                    "Un mismo grupo aparece en "
                    "validation y test potencialmente:\n"
                    f"class={class_name}\n"
                    f"group={group_name}\n"
                    f"sessions={sessions}"
                )

            # ----------------------------------------------
            # SESIÓN 3 → VALIDATION
            # ----------------------------------------------

            if has_val:

                split = "validation"

                reason = (
                    "Grupo perteneciente a "
                    "sesión 3."
                )

            # ----------------------------------------------
            # SESIÓN 5 → TEST
            # ----------------------------------------------

            elif has_test:

                split = "test"

                reason = (
                    "Grupo perteneciente a "
                    "sesión 5."
                )

            # ----------------------------------------------
            # RESTO → TRAIN
            # ----------------------------------------------

            else:

                split = "train"

                reason = (
                    "Grupo fuera de sesiones "
                    "3 y 5."
                )

        rows.append({
            "class":
                class_name,

            "group":
                group_name,

            "samples":
                int(
                    len(part)
                ),

            "sessions":
                "|".join(
                    sessions
                ),

            "split_final_v2":
                split,

            "assignment_reason":
                reason,
        })

    assignments = pd.DataFrame(
        rows
    )

    return assignments


# ==========================================================
# APLICAR ASIGNACIONES
# ==========================================================

def apply_assignments(
    df,
    assignments
):

    result = df.copy()

    # Normalizar class antes del merge.
    result["class"] = (
        result["class"]
        .astype(str)
        .str.upper()
    )

    result["group"] = (
        result["group"]
        .astype(str)
    )

    result = result.merge(
        assignments[
            [
                "class",
                "group",
                "split_final_v2"
            ]
        ],
        on=[
            "class",
            "group"
        ],
        how="left",
        validate="many_to_one"
    )

    missing = (
        result[
            "split_final_v2"
        ]
        .isna()
    )

    if missing.any():

        missing_groups = (
            result.loc[
                missing,
                [
                    "class",
                    "group"
                ]
            ]
            .drop_duplicates()
        )

        raise RuntimeError(
            "Hay muestras sin asignación:\n"
            + missing_groups.to_string(
                index=False
            )
        )

    # ------------------------------------------------------
    # Reemplazamos el "unassigned"
    # por el split definitivo V2.
    # ------------------------------------------------------

    result[
        "split_final"
    ] = (
        result[
            "split_final_v2"
        ]
    )

    result = result.drop(
        columns=[
            "split_final_v2"
        ]
    )

    return result


# ==========================================================
# AUDITORÍA
# ==========================================================

def audit_splits(
    df,
    assignments
):

    print()
    separator()
    print("AUDITORÍA SPLITS CANONICAL V2")
    separator()

    # ------------------------------------------------------
    # Conteos generales
    # ------------------------------------------------------

    split_counts = (
        df[
            "split_final"
        ]
        .value_counts()
        .reindex(
            [
                "train",
                "validation",
                "test"
            ],
            fill_value=0
        )
    )

    print()
    print(
        "Conteos:"
    )

    print(
        split_counts.to_string()
    )

    print()
    print(
        "Porcentajes:"
    )

    percentages = (
        split_counts
        / len(df)
        * 100
    )

    print(
        percentages
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    # ------------------------------------------------------
    # No deben quedar unassigned.
    # ------------------------------------------------------

    allowed = {
        "train",
        "validation",
        "test",
    }

    found = set(
        df[
            "split_final"
        ]
        .astype(str)
        .unique()
    )

    if found != allowed:

        raise RuntimeError(
            "Splits inesperados:\n"
            f"{found}"
        )

    # ------------------------------------------------------
    # E/L/Z exclusivamente TRAIN.
    # ------------------------------------------------------

    provisional = df[
        provisional_mask(
            df
        )
    ]

    provisional_split_counts = (
        provisional[
            "split_final"
        ]
        .value_counts()
    )

    print()
    print(
        "E/L/Z por split:"
    )

    print(
        provisional_split_counts
        .to_string()
    )

    if set(
        provisional[
            "split_final"
        ].unique()
    ) != {
        "train"
    }:

        raise RuntimeError(
            "E/L/Z aparecieron fuera de TRAIN."
        )

    # ------------------------------------------------------
    # Las 23 genuinas deben existir
    # en TRAIN, VALIDATION y TEST.
    # ------------------------------------------------------

    genuine = df[
        ~provisional_mask(
            df
        )
    ].copy()

    expected_genuine = (
        GENUINE_CLASSES
    )

    print()
    print(
        "Cobertura de clases genuinas:"
    )

    coverage_rows = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        part = genuine[
            genuine[
                "split_final"
            ].eq(split)
        ]

        classes = set(
            part[
                "class"
            ]
            .astype(str)
            .str.upper()
            .unique()
        )

        missing_classes = (
            expected_genuine
            - classes
        )

        extra_classes = (
            classes
            - expected_genuine
        )

        coverage_rows.append({
            "split":
                split,

            "classes":
                len(classes),

            "missing":
                ",".join(
                    sorted(
                        missing_classes
                    )
                ),

            "extra":
                ",".join(
                    sorted(
                        extra_classes
                    )
                ),
        })

        print(
            f"{split:<10}: "
            f"{len(classes)} clases"
        )

        if missing_classes:

            print(
                "   ❌ faltan:",
                sorted(
                    missing_classes
                )
            )

        if extra_classes:

            print(
                "   ⚠ extras:",
                sorted(
                    extra_classes
                )
            )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    # ------------------------------------------------------
    # Queremos 23 genuinas en los tres.
    # ------------------------------------------------------

    for row in coverage_rows:

        if (
            row[
                "classes"
            ]
            != 23
            or row[
                "missing"
            ]
        ):

            raise RuntimeError(
                "No hay cobertura completa "
                "de 23 clases genuinas en "
                f"{row['split']}."
            )

    # ------------------------------------------------------
    # No puede haber class+group
    # en más de un split.
    # ------------------------------------------------------

    leakage = (
        df
        .groupby(
            [
                "class",
                "group"
            ]
        )[
            "split_final"
        ]
        .nunique()
    )

    leaking_groups = (
        leakage[
            leakage > 1
        ]
    )

    print()
    print(
        "Grupos con leakage entre splits:",
        len(
            leaking_groups
        )
    )

    if len(
        leaking_groups
    ) > 0:

        raise RuntimeError(
            "Hay class+group presentes "
            "en múltiples splits."
        )

    # ------------------------------------------------------
    # Verificar G6/H6/Y6 ausentes.
    # ------------------------------------------------------

    forbidden = {
        ("G", "FRAM_G6"),
        ("H", "FRAM_H6"),
        ("Y", "FRAM_Y6"),
    }

    present_groups = set(
        zip(
            df[
                "class"
            ].astype(str),

            df[
                "group"
            ].astype(str)
        )
    )

    intersection = (
        forbidden
        & present_groups
    )

    if intersection:

        raise RuntimeError(
            "Reaparecieron grupos excluidos:\n"
            f"{intersection}"
        )

    print(
        "Grupos excluidos G6/H6/Y6 "
        "siguen ausentes: ✅"
    )

    # ------------------------------------------------------
    # Features no finitas.
    # ------------------------------------------------------

    img_cols = [
        c
        for c in df.columns
        if str(c).startswith(
            "img_norm_"
        )
    ]

    world_cols = [
        c
        for c in df.columns
        if str(c).startswith(
            "world_norm_"
        )
    ]

    X_img = (
        df[
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

    X_world = (
        df[
            world_cols
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
        X_img
    ).all():

        raise RuntimeError(
            "img_norm contiene NaN/Inf."
        )

    if not np.isfinite(
        X_world
    ).all():

        raise RuntimeError(
            "world_norm contiene NaN/Inf."
        )

    print(
        "Landmarks finitos: ✅"
    )

    # ------------------------------------------------------
    # Auditoría textual
    # ------------------------------------------------------

    audit_rows = [
        {
            "check":
                "total_samples",
            "value":
                len(df),
            "status":
                "OK",
        },

        {
            "check":
                "train_samples",
            "value":
                int(
                    split_counts[
                        "train"
                    ]
                ),
            "status":
                "OK",
        },

        {
            "check":
                "validation_samples",
            "value":
                int(
                    split_counts[
                        "validation"
                    ]
                ),
            "status":
                "OK",
        },

        {
            "check":
                "test_samples",
            "value":
                int(
                    split_counts[
                        "test"
                    ]
                ),
            "status":
                "OK",
        },

        {
            "check":
                "group_leakage",
            "value":
                len(
                    leaking_groups
                ),
            "status":
                "OK",
        },

        {
            "check":
                "provisional_train_only",
            "value":
                len(
                    provisional
                ),
            "status":
                "OK",
        },

        {
            "check":
                "genuine_classes_train",
            "value":
                23,
            "status":
                "OK",
        },

        {
            "check":
                "genuine_classes_validation",
            "value":
                23,
            "status":
                "OK",
        },

        {
            "check":
                "genuine_classes_test",
            "value":
                23,
            "status":
                "OK",
        },
    ]

    audit_df = pd.DataFrame(
        audit_rows
    )

    print()
    print(
        "✅ Auditoría de splits completada."
    )

    return (
        split_counts,
        coverage_df,
        audit_df
    )


# ==========================================================
# RESÚMENES
# ==========================================================

def create_summaries(
    df,
    assignments,
    split_counts,
    audit_df
):

    # ------------------------------------------------------
    # POR CLASE Y SPLIT
    # ------------------------------------------------------

    by_class = (
        df
        .groupby(
            [
                "class",
                "split_final"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=[
                "train",
                "validation",
                "test"
            ],
            fill_value=0
        )
        .reset_index()
        .sort_values(
            "class"
        )
    )

    by_class[
        "total"
    ] = (
        by_class[
            [
                "train",
                "validation",
                "test"
            ]
        ].sum(
            axis=1
        )
    )

    # ------------------------------------------------------
    # GENERAL
    # ------------------------------------------------------

    summary = pd.DataFrame({
        "split": [
            "train",
            "validation",
            "test",
            "total",
        ],

        "samples": [
            int(
                split_counts[
                    "train"
                ]
            ),

            int(
                split_counts[
                    "validation"
                ]
            ),

            int(
                split_counts[
                    "test"
                ]
            ),

            len(df),
        ]
    })

    # ------------------------------------------------------
    # GUARDAR
    # ------------------------------------------------------

    assignments.to_csv(
        GROUP_ASSIGNMENTS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    by_class.to_csv(
        CLASS_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    audit_df.to_csv(
        AUDIT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return (
        by_class,
        summary
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
    separator()
    print("=== LSM CANONICAL V2 — BUILD SPLITS ===")
    separator()

    # ------------------------------------------------------
    # Cargar
    # ------------------------------------------------------

    df = load_dataset()

    print()
    print(
        "Muestras:",
        len(df)
    )

    # ------------------------------------------------------
    # Debemos partir del dataset sin asignar.
    # ------------------------------------------------------

    current_splits = set(
        df[
            "split_final"
        ]
        .astype(str)
        .unique()
    )

    if current_splits != {
        "unassigned"
    }:

        raise RuntimeError(
            "El dataset fuente no está "
            "completamente unassigned.\n"
            f"Valores encontrados: "
            f"{current_splits}"
        )

    # ------------------------------------------------------
    # Asignaciones de grupo
    # ------------------------------------------------------

    assignments = (
        build_group_assignments(
            df
        )
    )

    print()
    print(
        "Grupos class+group:",
        len(
            assignments
        )
    )

    print()
    print(
        "Grupos por split:"
    )

    print(
        assignments[
            "split_final_v2"
        ]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------
    # Aplicar
    # ------------------------------------------------------

    result = (
        apply_assignments(
            df,
            assignments
        )
    )

    # ------------------------------------------------------
    # Auditar
    # ------------------------------------------------------

    (
        split_counts,
        coverage_df,
        audit_df
    ) = audit_splits(
        result,
        assignments
    )

    # ------------------------------------------------------
    # Resúmenes
    # ------------------------------------------------------

    (
        by_class,
        summary
    ) = create_summaries(
        result,
        assignments,
        split_counts,
        audit_df
    )

    print()
    separator()
    print("CONTEO POR CLASE")
    separator()

    print()
    print(
        by_class.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Guardar dataset DESPUÉS de pasar auditoría.
    # ------------------------------------------------------

    result.to_csv(
        OUTPUT_DATASET,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    separator()
    print("✅ SPLITS CANONICAL V2 GENERADOS")
    separator()

    print()
    print(
        "Dataset:"
    )

    print(
        OUTPUT_DATASET
    )

    print()
    print(
        "Resumen:"
    )

    print(
        SPLIT_SUMMARY_PATH
    )

    print()
    print(
        "Por clase:"
    )

    print(
        CLASS_SUMMARY_PATH
    )

    print()
    print(
        "Asignación por grupo:"
    )

    print(
        GROUP_ASSIGNMENTS_PATH
    )

    print()
    print(
        "Auditoría:"
    )

    print(
        AUDIT_PATH
    )

    print()
    print(
        "Política V2:"
    )

    print(
        "  TRAIN      = resto de sesiones/grupos"
    )

    print(
        "  VALIDATION = sesión 3"
    )

    print(
        "  TEST       = sesión 5"
    )

    print(
        "  E/L/Z      = TRAIN ONLY"
    )


if __name__ == "__main__":
    main()
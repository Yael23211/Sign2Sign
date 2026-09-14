import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# RUTAS
# ==========================================================

DATASET_PATH = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv"
)

GROUP_AUDIT_PATH = Path(
    r"D:\Sign2SignData\inventory\lsm_group_audit\lsm_group_audit.csv"
)

OUTPUT_DIR = Path(
    r"D:\Sign2SignData\inventory\lsm_canonical_v2"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "lsm_canonical_group_manifest.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "lsm_canonical_group_manifest_summary.csv"
)


# ==========================================================
# DECISIONES MANUALES YA JUSTIFICADAS
#
# IMPORTANTE:
# Estas decisiones NO se basan únicamente en que L3 falló.
# Provienen de la revisión visual y del criterio de referencia
# adoptado para el proyecto.
#
# Las claves SIEMPRE son:
#
#     (class, group)
#
# ==========================================================

MANUAL_OVERRIDES = {

    # ------------------------------------------------------
    # SESIÓN 6:
    # orientación claramente distinta de la referencia
    # observada en el resto de grupos.
    # ------------------------------------------------------

    ("G", "FRAM_G6"): {
        "decision":
            "EXCLUDE",

        "orientation_status":
            "OUTSIDE_REFERENCE",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La realización observada presenta una "
                "orientación de mano/palma distinta de la "
                "referencia adoptada para Sign2Sign y del "
                "resto de grupos G revisados."
            ),

        "notes":
            (
                "Outlier extremo confirmado mediante "
                "auditoría geométrica y comparación visual."
            ),
    },

    ("H", "FRAM_H6"): {
        "decision":
            "EXCLUDE",

        "orientation_status":
            "OUTSIDE_REFERENCE",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La realización observada presenta una "
                "orientación de mano/palma distinta de la "
                "referencia adoptada para Sign2Sign y del "
                "resto de grupos H revisados."
            ),

        "notes":
            (
                "Outlier extremo confirmado mediante "
                "auditoría geométrica y comparación visual."
            ),
    },

    ("Y", "FRAM_Y6"): {
        "decision":
            "EXCLUDE",

        "orientation_status":
            "OUTSIDE_REFERENCE",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La realización observada presenta una "
                "orientación de mano/palma distinta de la "
                "referencia adoptada para Sign2Sign y del "
                "resto de grupos Y revisados."
            ),

        "notes":
            (
                "Outlier extremo confirmado mediante "
                "auditoría geométrica y comparación visual."
            ),
    },


    # ------------------------------------------------------
    # GRUPOS NORMAL:
    #
    # La auditoría visual mostró variabilidad razonable.
    # NO se excluyen por diferencias geométricas.
    # ------------------------------------------------------

    ("O", "O_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La comparación visual conserva la "
                "configuración correspondiente a O; "
                "las diferencias observadas se consideran "
                "variabilidad aceptable."
            ),

        "notes":
            "Conservar para robustez.",
    },

    ("Q", "Q_NORMAL"): {
        "decision":
            "REVIEW",

        "orientation_status":
            "UNKNOWN",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            False,

        "decision_reason":
            (
                "La muestra parece pertenecer a la familia "
                "visual de Q, pero se decidió contrastarla "
                "formalmente con la referencia adoptada "
                "antes de cerrar Canonical V2."
            ),

        "notes":
            "Revisión lingüística pendiente.",
    },

    ("R", "R_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La configuración visual es coherente con "
                "otros grupos de R; las diferencias se "
                "consideran variabilidad válida."
            ),

        "notes":
            "Conservar para robustez.",
    },

    ("S", "S_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La configuración visual es coherente con "
                "otros grupos de S."
            ),

        "notes":
            "Conservar para robustez.",
    },

    ("T", "T_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La configuración visual es coherente con "
                "otros grupos de T."
            ),

        "notes":
            "Conservar para robustez.",
    },

    ("U", "U_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La configuración visual es coherente con "
                "otros grupos de U; las diferencias "
                "geométricas no justifican exclusión."
            ),

        "notes":
            "Conservar para robustez.",
    },

    ("V", "V_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La configuración visual es coherente con "
                "otros grupos de V; se conserva la "
                "variabilidad observada."
            ),

        "notes":
            "Conservar para robustez.",
    },

    ("W", "W_NORMAL"): {
        "decision":
            "KEEP",

        "orientation_status":
            "ACCEPTABLE_VARIATION",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            True,

        "decision_reason":
            (
                "La configuración visual es coherente con "
                "otros grupos de W."
            ),

        "notes":
            "Conservar para robustez.",
    },


    # ------------------------------------------------------
    # PARTICULARIDAD HISTÓRICA K / P
    # ------------------------------------------------------

    ("K", "P_NORMAL"): {
        "decision":
            "REVIEW",

        "orientation_status":
            "UNKNOWN",

        "laterality":
            "UNKNOWN",

        "reference_checked":
            False,

        "decision_reason":
            (
                "El contenido visual corresponde a la clase "
                "K según el manifiesto actual, pero el nombre "
                "del grupo P_NORMAL mantiene una "
                "inconsistencia nomenclatural histórica."
            ),

        "notes":
            (
                "No eliminar. Revisar nomenclatura K/P "
                "antes de cerrar Canonical V2."
            ),
    },
}


# ==========================================================
# UTILIDADES
# ==========================================================

def unique_join(series):

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = [
        value
        for value in values.unique()
        if value
        and value.lower() != "nan"
    ]

    if not values:
        return ""

    return "|".join(
        sorted(values)
    )


def first_non_null(series):

    values = (
        series
        .dropna()
    )

    if len(values) == 0:
        return ""

    return values.iloc[0]


def bool_to_text(value):

    return (
        "YES"
        if bool(value)
        else "NO"
    )


# ==========================================================
# CARGAR DATASET GENUINO LSM
# ==========================================================

def load_genuine_lsm():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "No se encontró:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    required = [
        "class",
        "group",
        "split_final",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                "Falta columna obligatoria: "
                f"{column}"
            )

    # ------------------------------------------------------
    # Solo filas destinadas al clasificador.
    # ------------------------------------------------------

    if "use_letter_classifier" in df.columns:

        df = df[
            pd.to_numeric(
                df[
                    "use_letter_classifier"
                ],
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
        ].copy()

    # ------------------------------------------------------
    # Solo fuente LSM.
    #
    # Esto deja FUERA del manifiesto de curación a
    # E/L/Z provisionales provenientes de ASL.
    #
    # Esas clases se conservarán aparte cuando construyamos
    # el CSV V2.
    # ------------------------------------------------------

    if "source_language" in df.columns:

        df = df[
            df[
                "source_language"
            ]
            .astype(str)
            .str.upper()
            .eq("LSM")
        ].copy()

    # ------------------------------------------------------
    # Quitar provisionales.
    # ------------------------------------------------------

    if "provisional_source" in df.columns:

        df = df[
            pd.to_numeric(
                df[
                    "provisional_source"
                ],
                errors="coerce"
            )
            .fillna(0)
            .eq(0)
        ].copy()

    df = (
        df
        .reset_index(
            drop=True
        )
    )

    return df


# ==========================================================
# CONSTRUIR RESUMEN POR (CLASS, GROUP)
# ==========================================================

def build_group_manifest(df):

    rows = []

    grouped = df.groupby(
        [
            "class",
            "group"
        ],
        dropna=False,
        sort=True
    )

    for (
        class_name,
        group_name
    ), part in grouped:

        row = {
            "class":
                str(
                    class_name
                ),

            "group":
                str(
                    group_name
                ),

            "samples":
                int(
                    len(part)
                ),

            "session_hint":
                (
                    unique_join(
                        part[
                            "session_hint"
                        ]
                    )
                    if "session_hint"
                    in part.columns
                    else ""
                ),

            # V1 únicamente como trazabilidad.
            # V2 reconstruirá sus splits.
            "v1_split_final":
                unique_join(
                    part[
                        "split_final"
                    ]
                ),

            "group_quality":
                (
                    unique_join(
                        part[
                            "group_quality"
                        ]
                    )
                    if "group_quality"
                    in part.columns
                    else ""
                ),

            "source_dataset":
                (
                    unique_join(
                        part[
                            "source_dataset"
                        ]
                    )
                    if "source_dataset"
                    in part.columns
                    else ""
                ),

            "source_archive":
                (
                    unique_join(
                        part[
                            "source_archive"
                        ]
                    )
                    if "source_archive"
                    in part.columns
                    else ""
                ),

            "source_language":
                (
                    unique_join(
                        part[
                            "source_language"
                        ]
                    )
                    if "source_language"
                    in part.columns
                    else "LSM"
                ),

            "filename_folder_mismatch_count":
                (
                    int(
                        pd.to_numeric(
                            part[
                                "filename_folder_mismatch"
                            ],
                            errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    )
                    if (
                        "filename_folder_mismatch"
                        in part.columns
                    )
                    else 0
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ==========================================================
# MERGE CON AUDITORÍA GEOMÉTRICA
# ==========================================================

def merge_geometric_audit(
    manifest
):

    # Valores por defecto.
    geometric_columns = {
        "geometric_flag":
            "",

        "nearest_group_class":
            "",

        "nearest_group":
            "",

        "nearest_group_distance":
            np.nan,

        "nearest_same_class_group":
            "",

        "nearest_same_class_distance":
            np.nan,

        "same_class_rank":
            np.nan,

        "same_class_to_global_ratio":
            np.nan,

        "within_group_spread":
            np.nan,
    }

    for (
        column,
        default
    ) in geometric_columns.items():

        manifest[
            column
        ] = default

    if not GROUP_AUDIT_PATH.exists():

        print()
        print(
            "⚠ No se encontró auditoría geométrica:"
        )

        print(
            GROUP_AUDIT_PATH
        )

        print(
            "El manifiesto se generará igualmente."
        )

        return manifest

    audit = pd.read_csv(
        GROUP_AUDIT_PATH,
        low_memory=False
    )

    required = [
        "class",
        "group"
    ]

    if not all(
        column in audit.columns
        for column in required
    ):

        print()
        print(
            "⚠ lsm_group_audit.csv no contiene "
            "class/group. Se omite merge."
        )

        return manifest

    available_columns = [
        column
        for column in geometric_columns
        if column in audit.columns
    ]

    audit_subset = (
        audit[
            [
                "class",
                "group",
                *available_columns
            ]
        ]
        .drop_duplicates(
            subset=[
                "class",
                "group"
            ]
        )
        .copy()
    )

    # Quitar columnas placeholder antes de merge.
    manifest = manifest.drop(
        columns=available_columns,
        errors="ignore"
    )

    manifest = manifest.merge(
        audit_subset,
        on=[
            "class",
            "group"
        ],
        how="left"
    )

    # Reponer cualquier columna ausente.
    for (
        column,
        default
    ) in geometric_columns.items():

        if column not in manifest.columns:

            manifest[
                column
            ] = default

    return manifest


# ==========================================================
# COLUMNAS DE CURACIÓN
# ==========================================================

def initialize_curation_columns(
    manifest
):

    # ------------------------------------------------------
    # DEFAULT:
    #
    # KEEP.
    #
    # La filosofía es:
    #
    # No eliminamos un grupo hasta tener evidencia
    # lingüística/visual suficiente.
    # ------------------------------------------------------

    manifest[
        "decision"
    ] = "KEEP"

    manifest[
        "laterality"
    ] = "UNKNOWN"

    manifest[
        "orientation_status"
    ] = "UNKNOWN"

    manifest[
        "reference_checked"
    ] = "NO"

    manifest[
        "decision_reason"
    ] = (
        "Sin evidencia actual que justifique exclusión. "
        "Se conserva provisionalmente."
    )

    manifest[
        "notes"
    ] = ""

    # ------------------------------------------------------
    # Campo auxiliar.
    #
    # OJO:
    # NO altera decision.
    # ------------------------------------------------------

    manifest[
        "geometric_review_recommended"
    ] = np.where(
        manifest[
            "geometric_flag"
        ]
        .astype(str)
        .isin(
            [
                "REVIEW",
                "STRONG_REVIEW"
            ]
        ),
        "YES",
        "NO"
    )

    return manifest


# ==========================================================
# APLICAR DECISIONES MANUALES
# ==========================================================

def apply_manual_overrides(
    manifest
):

    applied = 0

    for (
        class_name,
        group_name
    ), values in (
        MANUAL_OVERRIDES.items()
    ):

        mask = (
            manifest[
                "class"
            ]
            .astype(str)
            .eq(
                class_name
            )
            &
            manifest[
                "group"
            ]
            .astype(str)
            .eq(
                group_name
            )
        )

        count = int(
            mask.sum()
        )

        if count == 0:

            print()
            print(
                "⚠ Override no encontrado:"
            )

            print(
                f"  class={class_name}, "
                f"group={group_name}"
            )

            continue

        manifest.loc[
            mask,
            "decision"
        ] = values[
            "decision"
        ]

        manifest.loc[
            mask,
            "laterality"
        ] = values[
            "laterality"
        ]

        manifest.loc[
            mask,
            "orientation_status"
        ] = values[
            "orientation_status"
        ]

        manifest.loc[
            mask,
            "reference_checked"
        ] = bool_to_text(
            values[
                "reference_checked"
            ]
        )

        manifest.loc[
            mask,
            "decision_reason"
        ] = values[
            "decision_reason"
        ]

        manifest.loc[
            mask,
            "notes"
        ] = values[
            "notes"
        ]

        applied += count

    print()
    print(
        "Overrides manuales aplicados:",
        applied
    )

    return manifest


# ==========================================================
# PRIORIDAD DE REVISIÓN
# ==========================================================

def add_review_priority(
    manifest
):

    def calculate(row):

        # EXCLUDE ya decidido.
        if (
            row[
                "decision"
            ]
            == "EXCLUDE"
        ):

            return 100

        # Revisión manual explícita.
        if (
            row[
                "decision"
            ]
            == "REVIEW"
        ):

            return 90

        rank = pd.to_numeric(
            pd.Series([
                row.get(
                    "same_class_rank",
                    np.nan
                )
            ]),
            errors="coerce"
        ).iloc[0]

        ratio = pd.to_numeric(
            pd.Series([
                row.get(
                    "same_class_to_global_ratio",
                    np.nan
                )
            ]),
            errors="coerce"
        ).iloc[0]

        score = 0

        # Solo sirve para PRIORIZAR inspección.
        # Nunca cambia KEEP automáticamente.

        if pd.notna(
            rank
        ):

            if rank > 50:
                score += 50

            elif rank > 15:
                score += 30

            elif rank > 5:
                score += 10

        if pd.notna(
            ratio
        ):

            if ratio >= 3:
                score += 30

            elif ratio >= 1.5:
                score += 15

        if (
            str(
                row.get(
                    "geometric_flag",
                    ""
                )
            )
            == "STRONG_REVIEW"
        ):

            score += 10

        return min(
            score,
            80
        )

    manifest[
        "review_priority"
    ] = manifest.apply(
        calculate,
        axis=1
    )

    return manifest


# ==========================================================
# RESUMEN
# ==========================================================

def build_summary(
    manifest
):

    summary = (
        manifest
        .groupby(
            "decision",
            dropna=False
        )
        .agg(
            groups=(
                "group",
                "count"
            ),

            samples=(
                "samples",
                "sum"
            ),
        )
        .reset_index()
    )

    return summary


# ==========================================================
# MAIN
# ==========================================================

def main(
    overwrite=False
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Proteger edición manual futura.
    # ------------------------------------------------------

    if (
        OUTPUT_MANIFEST.exists()
        and not overwrite
    ):

        raise FileExistsError(
            "\nEl manifiesto ya existe:\n"
            f"{OUTPUT_MANIFEST}\n\n"
            "No se sobrescribió para proteger "
            "decisiones manuales.\n\n"
            "Si realmente quieres regenerarlo usa:\n"
            "python scripts\\create_lsm_canonical_manifest.py "
            "--overwrite"
        )

    print()
    print(
        "============================================"
    )

    print(
        "=== LSM CANONICAL V2 - GROUP MANIFEST ==="
    )

    print(
        "============================================"
    )

    # ------------------------------------------------------
    # Cargar.
    # ------------------------------------------------------

    df = load_genuine_lsm()

    print()
    print(
        "Muestras LSM genuinas:",
        len(df)
    )

    print(
        "Clases:",
        df[
            "class"
        ].nunique()
    )

    print(
        "Grupos:",
        df[
            [
                "class",
                "group"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # ------------------------------------------------------
    # Agrupar.
    # ------------------------------------------------------

    manifest = (
        build_group_manifest(
            df
        )
    )

    # ------------------------------------------------------
    # Auditoría geométrica.
    # ------------------------------------------------------

    manifest = (
        merge_geometric_audit(
            manifest
        )
    )

    # ------------------------------------------------------
    # Curation.
    # ------------------------------------------------------

    manifest = (
        initialize_curation_columns(
            manifest
        )
    )

    manifest = (
        apply_manual_overrides(
            manifest
        )
    )

    manifest = (
        add_review_priority(
            manifest
        )
    )

    # ------------------------------------------------------
    # Orden:
    #
    # EXCLUDE/REVIEW primero,
    # luego prioridad,
    # luego clase/grupo.
    # ------------------------------------------------------

    decision_order = {
        "EXCLUDE": 0,
        "REVIEW": 1,
        "KEEP": 2,
    }

    manifest[
        "_decision_order"
    ] = (
        manifest[
            "decision"
        ]
        .map(
            decision_order
        )
        .fillna(99)
    )

    manifest = (
        manifest
        .sort_values(
            [
                "_decision_order",
                "review_priority",
                "class",
                "group"
            ],
            ascending=[
                True,
                False,
                True,
                True
            ]
        )
        .drop(
            columns=[
                "_decision_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------
    # Orden de columnas.
    # ------------------------------------------------------

    preferred_columns = [

        # Identidad
        "class",
        "group",
        "samples",
        "session_hint",
        "v1_split_final",

        # Decisión
        "decision",
        "reference_checked",
        "laterality",
        "orientation_status",
        "decision_reason",
        "notes",

        # Priorización
        "review_priority",
        "geometric_review_recommended",

        # Auditoría geométrica
        "geometric_flag",
        "nearest_group_class",
        "nearest_group",
        "nearest_group_distance",
        "nearest_same_class_group",
        "nearest_same_class_distance",
        "same_class_rank",
        "same_class_to_global_ratio",
        "within_group_spread",

        # Metadata
        "group_quality",
        "source_dataset",
        "source_archive",
        "source_language",
        "filename_folder_mismatch_count",
    ]

    remaining_columns = [
        column
        for column in manifest.columns
        if column not in preferred_columns
    ]

    final_columns = [
        column
        for column in preferred_columns
        if column in manifest.columns
    ] + remaining_columns

    manifest = manifest[
        final_columns
    ]

    # ------------------------------------------------------
    # Guardar.
    # ------------------------------------------------------

    manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig"
    )

    summary = build_summary(
        manifest
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------
    # CONSOLA
    # ------------------------------------------------------

    print()
    print(
        "============================================"
    )

    print(
        "=== DECISIONES INICIALES ==="
    )

    print(
        "============================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== REVISIÓN PRIORITARIA ==="
    )

    print(
        "============================================"
    )

    review_columns = [
        "class",
        "group",
        "session_hint",
        "decision",
        "review_priority",
        "orientation_status",
        "geometric_flag",
        "same_class_rank",
        "same_class_to_global_ratio",
    ]

    print(
        manifest[
            review_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "=== ARCHIVOS ==="
    )

    print(
        "============================================"
    )

    print()
    print(
        OUTPUT_MANIFEST
    )

    print(
        OUTPUT_SUMMARY
    )

    print()

    print(
        "✅ Manifiesto Canonical V2 generado."
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "- STRONG_REVIEW NO implica EXCLUDE."
    )

    print(
        "- LEFT tampoco implica EXCLUDE."
    )

    print(
        "- La decisión final debe basarse en "
        "la referencia adoptada y revisión visual."
    )

    print(
        "- v1_split_final es solo trazabilidad."
    )

    print(
        "- Los splits de V2 se reconstruirán después."
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Genera el manifiesto de curación "
            "LSM Canonical V2."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permite sobrescribir un manifiesto "
            "existente. Usar con cuidado."
        )
    )

    args = parser.parse_args()

    main(
        overwrite=args.overwrite
    )
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# RUTAS
# ==========================================================

SOURCE_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_final.csv"
)

SOURCE_MANIFEST = Path(
    r"D:\Sign2SignData\inventory\lsm_canonical_v2"
    r"\lsm_canonical_group_manifest.csv"
)

OUTPUT_DATASET = Path(
    r"D:\Sign2SignData\processed\landmarks"
    r"\lsm_landmarks_canonical_v2.csv"
)

OUTPUT_DIR = Path(
    r"D:\Sign2SignData\inventory\lsm_canonical_v2"
)

FINAL_MANIFEST = (
    OUTPUT_DIR
    / "lsm_canonical_group_manifest_FINAL.csv"
)

FINAL_MANIFEST_SUMMARY = (
    OUTPUT_DIR
    / "lsm_canonical_group_manifest_FINAL_summary.csv"
)

CLASS_SUMMARY = (
    OUTPUT_DIR
    / "lsm_canonical_v2_class_summary.csv"
)

EXCLUDED_SUMMARY = (
    OUTPUT_DIR
    / "lsm_canonical_v2_excluded_groups.csv"
)

BUILD_SUMMARY = (
    OUTPUT_DIR
    / "lsm_canonical_v2_build_summary.csv"
)


# ==========================================================
# DATOS CONOCIDOS
# ==========================================================

PROVISIONAL_CLASSES = {
    "E",
    "L",
    "Z",
}

EXPECTED_PROVISIONAL_COUNTS = {
    "E": 929,
    "L": 930,
    "Z": 900,
}

EXPECTED_V1_TOTAL = 88642
EXPECTED_GENUINE_V1 = 85883
EXPECTED_PROVISIONAL_TOTAL = 2759

EXPECTED_GENUINE_V2 = 83933
EXPECTED_V2_TOTAL = 86692
EXPECTED_EXCLUDED = 1950

EXPECTED_EXCLUDED_GROUPS = {
    ("G", "FRAM_G6"),
    ("H", "FRAM_H6"),
    ("Y", "FRAM_Y6"),
}


# ==========================================================
# DECISIONES FINALES DE LA AUDITORÍA
# ==========================================================

FINAL_UPDATES = {

    ("Q", "Q_NORMAL"): {
        "decision": "KEEP",
        "orientation_status": "ACCEPTABLE_VARIATION",
        "decision_reason": (
            "La revisión visual confirma una realización "
            "compatible con Q. Se conserva la variabilidad "
            "de lateralidad y perspectiva."
        ),
        "notes": (
            "KEEP tras revisión visual."
        ),
    },

    ("K", "P_NORMAL"): {
        "decision": "KEEP",
        "orientation_status": "ACCEPTABLE_VARIATION",
        "decision_reason": (
            "La revisión visual confirma que el contenido "
            "corresponde a la clase K. El nombre P_NORMAL "
            "se conserva únicamente como identificador "
            "histórico del grupo."
        ),
        "notes": (
            "Inconsistencia nomenclatural histórica K/P. "
            "No reclasificar."
        ),
    },

    ("A", "PHONE_BURST"): {
        "decision": "KEEP",
        "orientation_status": "ACCEPTABLE_VARIATION",
        "decision_reason": (
            "La revisión visual confirma una A válida. "
            "La diferencia geométrica se atribuye a "
            "lateralidad/handedness y perspectiva."
        ),
        "notes": (
            "Conservar como variabilidad válida."
        ),
    },

    ("P", "K7_FRAME"): {
        "decision": "KEEP",
        "orientation_status": "ACCEPTABLE_VARIATION",
        "decision_reason": (
            "La revisión visual confirma una P válida. "
            "La diferencia se considera variabilidad "
            "aceptable de lateralidad y perspectiva."
        ),
        "notes": (
            "K7_FRAME permanece como identificador histórico."
        ),
    },
}


# ==========================================================
# UTILIDADES
# ==========================================================

def separator():
    print("=" * 64)


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

def load_inputs():

    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"No existe:\n{SOURCE_DATASET}"
        )

    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"No existe:\n{SOURCE_MANIFEST}"
        )

    df = pd.read_csv(
        SOURCE_DATASET,
        low_memory=False
    )

    manifest = pd.read_csv(
        SOURCE_MANIFEST,
        low_memory=False
    )

    return df, manifest


# ==========================================================
# FINALIZAR MANIFIESTO
# ==========================================================

def finalize_manifest(manifest):

    manifest = manifest.copy()

    print()
    separator()
    print("ACTUALIZANDO DECISIONES FINALES")
    separator()

    for (
        class_name,
        group_name
    ), update in FINAL_UPDATES.items():

        mask = (
            manifest["class"]
            .astype(str)
            .eq(class_name)
            &
            manifest["group"]
            .astype(str)
            .eq(group_name)
        )

        count = int(mask.sum())

        if count != 1:
            raise RuntimeError(
                f"No se encontró de forma única:\n"
                f"{class_name} | {group_name}\n"
                f"Coincidencias: {count}"
            )

        for column, value in update.items():

            if column not in manifest.columns:
                manifest[column] = ""

            manifest.loc[
                mask,
                column
            ] = value

        if "reference_checked" in manifest.columns:
            manifest.loc[
                mask,
                "reference_checked"
            ] = "NO"

        print(
            f"✅ {class_name:>2} | "
            f"{group_name:<20} → KEEP"
        )

    # ------------------------------------------------------
    # Validar decisiones
    # ------------------------------------------------------

    manifest["decision"] = (
        manifest["decision"]
        .astype(str)
        .str.upper()
    )

    reviews = manifest[
        manifest["decision"].eq("REVIEW")
    ]

    if len(reviews) > 0:

        raise RuntimeError(
            "Todavía existen grupos REVIEW:\n"
            + reviews[
                [
                    "class",
                    "group"
                ]
            ].to_string(index=False)
        )

    valid = {
        "KEEP",
        "EXCLUDE",
    }

    found = set(
        manifest["decision"].unique()
    )

    unexpected = found - valid

    if unexpected:
        raise RuntimeError(
            f"Decisiones inesperadas: {unexpected}"
        )

    # ------------------------------------------------------
    # Guardar manifiesto final
    # ------------------------------------------------------

    manifest.to_csv(
        FINAL_MANIFEST,
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        manifest
        .groupby("decision")
        .agg(
            groups=("group", "count"),
            samples=("samples", "sum"),
        )
        .reset_index()
    )

    summary.to_csv(
        FINAL_MANIFEST_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    separator()
    print("MANIFIESTO FINAL")
    separator()

    print(
        summary.to_string(
            index=False
        )
    )

    return manifest


# ==========================================================
# SEPARAR GENUINAS / PROVISIONALES
# ==========================================================

def split_sources(df):

    mask = provisional_mask(df)

    provisional = (
        df[mask]
        .copy()
        .reset_index(drop=True)
    )

    genuine = (
        df[~mask]
        .copy()
        .reset_index(drop=True)
    )

    counts = (
        provisional["class"]
        .astype(str)
        .str.upper()
        .value_counts()
        .sort_index()
    )

    actual_counts = counts.to_dict()

    print()
    print("Provisionales detectadas:")

    print(
        counts.to_string()
    )

    if actual_counts != EXPECTED_PROVISIONAL_COUNTS:
        raise RuntimeError(
            "Conteos provisionales inesperados.\n"
            f"Esperados: {EXPECTED_PROVISIONAL_COUNTS}\n"
            f"Encontrados: {actual_counts}"
        )

    if len(provisional) != EXPECTED_PROVISIONAL_TOTAL:
        raise RuntimeError(
            f"Provisionales esperadas: "
            f"{EXPECTED_PROVISIONAL_TOTAL}\n"
            f"Encontradas: {len(provisional)}"
        )

    if len(genuine) != EXPECTED_GENUINE_V1:
        raise RuntimeError(
            f"Genuinas esperadas: "
            f"{EXPECTED_GENUINE_V1}\n"
            f"Encontradas: {len(genuine)}"
        )

    print()
    print("✅ Separación de fuentes correcta.")

    print(
        "LSM genuinas:",
        len(genuine)
    )

    print(
        "Provisionales E/L/Z:",
        len(provisional)
    )

    return genuine, provisional


# ==========================================================
# FILTRAR SEGÚN MANIFIESTO
# ==========================================================

def curate_genuine(
    genuine,
    manifest
):

    # ------------------------------------------------------
    # IMPORTANTE:
    #
    # TODOS los nombres del manifiesto se renombran con
    # prefijo curation_ ANTES DEL MERGE.
    #
    # Así jamás chocan con columnas originales como:
    #
    # notes
    # group_quality
    # etc.
    # ------------------------------------------------------

    decisions = manifest[
        [
            "class",
            "group",
            "decision",
            "decision_reason",
            "orientation_status",
            "notes",
        ]
    ].copy()

    decisions = decisions.rename(
        columns={
            "decision":
                "curation_decision",

            "decision_reason":
                "curation_reason",

            "orientation_status":
                "curation_orientation_status",

            "notes":
                "curation_notes",
        }
    )

    merged = genuine.merge(
        decisions,
        on=[
            "class",
            "group"
        ],
        how="left",
        validate="many_to_one"
    )

    missing = merged[
        "curation_decision"
    ].isna()

    if missing.any():

        missing_groups = (
            merged.loc[
                missing,
                [
                    "class",
                    "group"
                ]
            ]
            .drop_duplicates()
        )

        raise RuntimeError(
            "Hay grupos sin decisión:\n"
            + missing_groups.to_string(
                index=False
            )
        )

    keep_mask = (
        merged["curation_decision"]
        .astype(str)
        .str.upper()
        .eq("KEEP")
    )

    exclude_mask = (
        merged["curation_decision"]
        .astype(str)
        .str.upper()
        .eq("EXCLUDE")
    )

    kept = (
        merged[
            keep_mask
        ]
        .copy()
    )

    excluded = (
        merged[
            exclude_mask
        ]
        .copy()
    )

    print()
    print(
        "Genuinas KEEP:",
        len(kept)
    )

    print(
        "Genuinas EXCLUDE:",
        len(excluded)
    )

    if len(kept) != EXPECTED_GENUINE_V2:
        raise RuntimeError(
            f"KEEP esperadas: "
            f"{EXPECTED_GENUINE_V2}\n"
            f"Encontradas: {len(kept)}"
        )

    if len(excluded) != EXPECTED_EXCLUDED:
        raise RuntimeError(
            f"EXCLUDE esperadas: "
            f"{EXPECTED_EXCLUDED}\n"
            f"Encontradas: {len(excluded)}"
        )

    return kept, excluded


# ==========================================================
# CONSTRUIR V2
# ==========================================================

def build_v2(
    original,
    kept,
    provisional
):

    original_columns = list(
        original.columns
    )

    # ------------------------------------------------------
    # Seleccionamos SOLO columnas originales.
    #
    # Las columnas curation_* desaparecen aquí.
    #
    # Esto evita completamente el bug de "notes".
    # ------------------------------------------------------

    kept_clean = (
        kept[
            original_columns
        ]
        .copy()
    )

    provisional_clean = (
        provisional[
            original_columns
        ]
        .copy()
    )

    v2 = pd.concat(
        [
            kept_clean,
            provisional_clean
        ],
        ignore_index=True
    )

    # ------------------------------------------------------
    # Guardar split viejo
    # ------------------------------------------------------

    if "v1_split_final" in v2.columns:
        raise RuntimeError(
            "Ya existe v1_split_final."
        )

    if "split_final" not in v2.columns:
        raise RuntimeError(
            "No existe split_final "
            "en dataset fuente."
        )

    position = (
        v2.columns.get_loc(
            "split_final"
        )
    )

    v2.insert(
        position,
        "v1_split_final",
        v2["split_final"]
        .astype(str)
    )

    # ------------------------------------------------------
    # Los splits V2 se crearán desde cero.
    # ------------------------------------------------------

    v2["split_final"] = (
        "unassigned"
    )

    # ------------------------------------------------------
    # E/L/Z únicamente TRAIN.
    # ------------------------------------------------------

    mask_provisional = (
        provisional_mask(v2)
    )

    v2["split_constraint"] = np.where(
        mask_provisional,
        "TRAIN_ONLY",
        "FREE"
    )

    return v2


# ==========================================================
# AUDITAR V2
# ==========================================================

def audit_v2(
    v2,
    excluded
):

    print()
    separator()
    print("AUDITORÍA CANONICAL V2")
    separator()

    total = len(v2)

    provisional_count = int(
        provisional_mask(v2).sum()
    )

    genuine_count = (
        total
        - provisional_count
    )

    print()
    print(
        "Muestras totales:",
        total
    )

    print(
        "LSM genuinas:",
        genuine_count
    )

    print(
        "Provisionales E/L/Z:",
        provisional_count
    )

    print(
        "Muestras excluidas:",
        len(excluded)
    )

    # ------------------------------------------------------
    # Grupos excluidos
    # ------------------------------------------------------

    excluded_groups = (
        excluded[
            [
                "class",
                "group"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "class",
                "group"
            ]
        )
    )

    print()
    print("Grupos excluidos:")

    print(
        excluded_groups.to_string(
            index=False
        )
    )

    actual_excluded = set(
        zip(
            excluded_groups["class"]
            .astype(str),

            excluded_groups["group"]
            .astype(str)
        )
    )

    if actual_excluded != (
        EXPECTED_EXCLUDED_GROUPS
    ):

        raise RuntimeError(
            "Grupos excluidos inesperados:\n"
            f"{actual_excluded}"
        )

    # ------------------------------------------------------
    # Conteos
    # ------------------------------------------------------

    if total != EXPECTED_V2_TOTAL:
        raise RuntimeError(
            f"Total esperado: "
            f"{EXPECTED_V2_TOTAL}\n"
            f"Encontrado: {total}"
        )

    if genuine_count != EXPECTED_GENUINE_V2:
        raise RuntimeError(
            f"Genuinas esperadas: "
            f"{EXPECTED_GENUINE_V2}\n"
            f"Encontradas: {genuine_count}"
        )

    if provisional_count != (
        EXPECTED_PROVISIONAL_TOTAL
    ):
        raise RuntimeError(
            f"Provisionales esperadas: "
            f"{EXPECTED_PROVISIONAL_TOTAL}\n"
            f"Encontradas: {provisional_count}"
        )

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    img_cols = [
        c
        for c in v2.columns
        if str(c).startswith(
            "img_norm_"
        )
    ]

    world_cols = [
        c
        for c in v2.columns
        if str(c).startswith(
            "world_norm_"
        )
    ]

    if len(img_cols) != 63:
        raise RuntimeError(
            f"img_norm columns: "
            f"{len(img_cols)}, esperaba 63."
        )

    if len(world_cols) != 63:
        raise RuntimeError(
            f"world_norm columns: "
            f"{len(world_cols)}, esperaba 63."
        )

    X_img = (
        v2[img_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    X_world = (
        v2[world_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    print()
    print(
        "img_norm NaN:",
        int(np.isnan(X_img).sum())
    )

    print(
        "img_norm Inf:",
        int(np.isinf(X_img).sum())
    )

    print(
        "world_norm NaN:",
        int(np.isnan(X_world).sum())
    )

    print(
        "world_norm Inf:",
        int(np.isinf(X_world).sum())
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

    # ------------------------------------------------------
    # Split nuevo aún vacío
    # ------------------------------------------------------

    split_values = set(
        v2["split_final"]
        .astype(str)
        .unique()
    )

    if split_values != {
        "unassigned"
    }:
        raise RuntimeError(
            "split_final no quedó "
            "completamente unassigned."
        )

    # ------------------------------------------------------
    # Restricciones
    # ------------------------------------------------------

    constraint_counts = (
        v2["split_constraint"]
        .value_counts()
    )

    print()
    print(
        "Restricciones de split:"
    )

    print(
        constraint_counts.to_string()
    )

    free_count = int(
        (
            v2["split_constraint"]
            == "FREE"
        ).sum()
    )

    train_only_count = int(
        (
            v2["split_constraint"]
            == "TRAIN_ONLY"
        ).sum()
    )

    if free_count != EXPECTED_GENUINE_V2:
        raise RuntimeError(
            f"FREE esperado: "
            f"{EXPECTED_GENUINE_V2}, "
            f"encontrado: {free_count}"
        )

    if train_only_count != (
        EXPECTED_PROVISIONAL_TOTAL
    ):
        raise RuntimeError(
            f"TRAIN_ONLY esperado: "
            f"{EXPECTED_PROVISIONAL_TOTAL}, "
            f"encontrado: {train_only_count}"
        )

    print()
    print(
        "✅ Auditoría Canonical V2 correcta."
    )

    return genuine_count, provisional_count


# ==========================================================
# GUARDAR RESÚMENES
# ==========================================================

def save_summaries(
    v2,
    excluded
):

    class_summary = (
        v2
        .groupby("class")
        .agg(
            total=(
                "class",
                "size"
            ),

            train_only=(
                "split_constraint",
                lambda values:
                int(
                    (
                        values
                        == "TRAIN_ONLY"
                    ).sum()
                )
            )
        )
        .reset_index()
        .sort_values("class")
    )

    class_summary.to_csv(
        CLASS_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------
    # Excluidos:
    #
    # Usamos nombres curation_*,
    # nunca las columnas originales.
    # ------------------------------------------------------

    excluded_summary = (
        excluded[
            [
                "class",
                "group",
                "session_hint",
                "curation_reason",
            ]
        ]
        .groupby(
            [
                "class",
                "group"
            ],
            as_index=False
        )
        .agg(
            samples=(
                "class",
                "size"
            ),

            session_hint=(
                "session_hint",
                "first"
            ),

            reason=(
                "curation_reason",
                "first"
            )
        )
    )

    excluded_summary.to_csv(
        EXCLUDED_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    build_summary = pd.DataFrame(
        [
            {
                "metric":
                    "canonical_v2_total",
                "value":
                    len(v2),
            },

            {
                "metric":
                    "canonical_v2_genuine",
                "value":
                    EXPECTED_GENUINE_V2,
            },

            {
                "metric":
                    "canonical_v2_provisional",
                "value":
                    EXPECTED_PROVISIONAL_TOTAL,
            },

            {
                "metric":
                    "excluded_samples",
                "value":
                    EXPECTED_EXCLUDED,
            },

            {
                "metric":
                    "excluded_groups",
                "value":
                    3,
            },
        ]
    )

    build_summary.to_csv(
        BUILD_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    separator()
    print("CONTEO POR CLASE")
    separator()

    print()
    print(
        class_summary.to_string(
            index=False
        )
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    separator()
    print("=== BUILD LSM CANONICAL V2 ===")
    separator()

    # ------------------------------------------------------
    # Inputs
    # ------------------------------------------------------

    df, manifest = load_inputs()

    print()
    print(
        "Dataset V1:",
        len(df)
    )

    print(
        "Grupos en manifiesto:",
        len(manifest)
    )

    if len(df) != EXPECTED_V1_TOTAL:
        raise RuntimeError(
            f"Dataset V1 esperado: "
            f"{EXPECTED_V1_TOTAL}\n"
            f"Encontrado: {len(df)}"
        )

    # ------------------------------------------------------
    # Manifest
    # ------------------------------------------------------

    manifest = (
        finalize_manifest(
            manifest
        )
    )

    # ------------------------------------------------------
    # Separar fuentes
    # ------------------------------------------------------

    genuine, provisional = (
        split_sources(
            df
        )
    )

    # ------------------------------------------------------
    # Curar genuinas
    # ------------------------------------------------------

    kept, excluded = (
        curate_genuine(
            genuine,
            manifest
        )
    )

    # ------------------------------------------------------
    # Crear V2
    # ------------------------------------------------------

    v2 = build_v2(
        df,
        kept,
        provisional
    )

    # ------------------------------------------------------
    # Auditar ANTES de guardar
    # ------------------------------------------------------

    audit_v2(
        v2,
        excluded
    )

    # ------------------------------------------------------
    # Guardar dataset
    # ------------------------------------------------------

    v2.to_csv(
        OUTPUT_DATASET,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------
    # Resúmenes
    # ------------------------------------------------------

    save_summaries(
        v2,
        excluded
    )

    # ------------------------------------------------------
    # FIN
    # ------------------------------------------------------

    print()
    separator()
    print("✅ LSM CANONICAL V2 GENERADO")
    separator()

    print()
    print("Dataset:")
    print(
        OUTPUT_DATASET
    )

    print()
    print("Manifiesto FINAL:")
    print(
        FINAL_MANIFEST
    )

    print()
    print("Resumen por clase:")
    print(
        CLASS_SUMMARY
    )

    print()
    print("Grupos excluidos:")
    print(
        EXCLUDED_SUMMARY
    )

    print()
    print("Build summary:")
    print(
        BUILD_SUMMARY
    )

    print()
    print(
        "split_final = unassigned"
    )

    print(
        "E/L/Z = TRAIN_ONLY"
    )

    print()
    print(
        "Siguiente paso: construir "
        "splits Canonical V2."
    )


if __name__ == "__main__":
    main()
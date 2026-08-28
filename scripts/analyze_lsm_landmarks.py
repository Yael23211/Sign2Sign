import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

N_LANDMARKS = 21
TOLERANCE = 1e-6

LANDMARK_PREFIXES = [
    "img_raw",
    "img_norm",
    "world_raw",
    "world_norm",
]


# ==========================================================
# COLUMNAS
# ==========================================================

def landmark_columns(prefix):

    cols = []

    for i in range(N_LANDMARKS):

        cols.extend([
            f"{prefix}_x{i}",
            f"{prefix}_y{i}",
            f"{prefix}_z{i}",
        ])

    return cols


def required_landmark_columns():

    cols = []

    for prefix in LANDMARK_PREFIXES:
        cols.extend(
            landmark_columns(prefix)
        )

    return cols


# ==========================================================
# CONTADORES
# ==========================================================

def make_stats():

    return {
        "total": 0,
        "valid": 0,
        "failed": 0,
    }


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    input_path = Path(args.input)

    if not input_path.exists():

        raise FileNotFoundError(
            f"No existe el archivo:\n"
            f"{input_path}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Columnas que esperamos
    # ------------------------------------------------------

    lm_cols = (
        required_landmark_columns()
    )

    # Solo cargaremos las columnas necesarias.
    metadata_cols = [
        "class",
        "path",
        "group",
        "group_quality",
        "landmarks_ok",
        "detection_variant",
        "handedness",
        "handedness_score",
        "image_scale_0_9",
        "world_scale_0_9",
    ]

    # Leer header primero.
    header = pd.read_csv(
        input_path,
        nrows=0
    )

    available_columns = set(
        header.columns
    )

    # ------------------------------------------------------
    # Verificar columnas
    # ------------------------------------------------------

    missing_landmark_cols = [
        col
        for col in lm_cols
        if col not in available_columns
    ]

    if missing_landmark_cols:

        print(
            "\n[ERROR] Faltan columnas "
            "de landmarks."
        )

        print(
            f"Cantidad faltante: "
            f"{len(missing_landmark_cols)}"
        )

        for col in missing_landmark_cols[:20]:
            print(f"  - {col}")

        raise ValueError(
            "El CSV no contiene todas las "
            "columnas esperadas."
        )

    use_metadata = [
        col
        for col in metadata_cols
        if col in available_columns
    ]

    columns_to_read = (
        use_metadata
        +
        lm_cols
    )

    print(
        "\n=== AUDITORÍA LSM LANDMARKS ==="
    )

    print(
        f"Archivo:"
        f"\n{input_path}"
    )

    print(
        f"\nColumnas de landmarks: "
        f"{len(lm_cols)}"
    )

    print(
        f"Tamaño de chunk      : "
        f"{args.chunk_size:,}"
    )

    # ======================================================
    # ESTADÍSTICAS GENERALES
    # ======================================================

    total_rows = 0
    valid_rows = 0
    failed_rows = 0

    # Integridad
    valid_with_nan = 0
    valid_with_inf = 0

    incomplete_img_raw = 0
    incomplete_img_norm = 0
    incomplete_world_raw = 0
    incomplete_world_norm = 0

    # Normalización
    img_wrist_fail = 0
    img_scale_fail = 0

    world_wrist_fail = 0
    world_scale_fail = 0

    max_img_wrist_error = 0.0
    max_img_scale_error = 0.0

    max_world_wrist_error = 0.0
    max_world_scale_error = 0.0

    # ------------------------------------------------------
    # Estadísticas categóricas
    # ------------------------------------------------------

    class_stats = defaultdict(
        make_stats
    )

    group_stats = defaultdict(
        make_stats
    )

    method_counter = Counter()
    handedness_counter = Counter()

    # Para listar anomalías concretas.
    problematic_rows = []

    # ======================================================
    # LEER POR CHUNKS
    # ======================================================

    chunk_number = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=columns_to_read,
        chunksize=args.chunk_size,
        low_memory=False,
    ):

        chunk_number += 1

        total_rows += len(chunk)

        # --------------------------------------------------
        # landmarks_ok
        # --------------------------------------------------

        ok = pd.to_numeric(
            chunk["landmarks_ok"],
            errors="coerce"
        ).fillna(0).astype(int)

        valid_mask = (
            ok == 1
        )

        failed_mask = (
            ~valid_mask
        )

        current_valid = int(
            valid_mask.sum()
        )

        current_failed = int(
            failed_mask.sum()
        )

        valid_rows += current_valid
        failed_rows += current_failed

        # --------------------------------------------------
        # Estadísticas por clase
        # --------------------------------------------------

        for (
            label,
            ok_value
        ) in zip(
            chunk["class"],
            ok
        ):

            label = str(label)

            class_stats[
                label
            ]["total"] += 1

            if ok_value == 1:

                class_stats[
                    label
                ]["valid"] += 1

            else:

                class_stats[
                    label
                ]["failed"] += 1

        # --------------------------------------------------
        # Estadísticas por grupo
        # --------------------------------------------------

        if "group" in chunk.columns:

            groups = (
                chunk["group"]
                .fillna("")
                .astype(str)
            )

        else:

            groups = pd.Series(
                [""] * len(chunk),
                index=chunk.index
            )

        for (
            label,
            group,
            ok_value
        ) in zip(
            chunk["class"],
            groups,
            ok
        ):

            key = (
                str(label),
                str(group)
            )

            group_stats[
                key
            ]["total"] += 1

            if ok_value == 1:

                group_stats[
                    key
                ]["valid"] += 1

            else:

                group_stats[
                    key
                ]["failed"] += 1

        # --------------------------------------------------
        # Cascade
        # --------------------------------------------------

        if "detection_variant" in chunk.columns:

            variants = (
                chunk.loc[
                    valid_mask,
                    "detection_variant"
                ]
                .fillna("")
                .astype(str)
            )

            method_counter.update(
                variants
            )

        # --------------------------------------------------
        # Handedness
        # --------------------------------------------------

        if "handedness" in chunk.columns:

            hands = (
                chunk.loc[
                    valid_mask,
                    "handedness"
                ]
                .fillna("")
                .astype(str)
            )

            handedness_counter.update(
                hands
            )

        # ==================================================
        # VALIDAR SOLO MUESTRAS CON LANDMARKS
        # ==================================================

        valid_chunk = (
            chunk.loc[
                valid_mask
            ]
            .copy()
        )

        if len(valid_chunk) == 0:

            print(
                f"Chunk {chunk_number}: "
                f"{total_rows:,} filas leídas"
            )

            continue

        # --------------------------------------------------
        # Convertir landmarks a numérico
        # --------------------------------------------------

        numeric = (
            valid_chunk[
                lm_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )

        values = (
            numeric.to_numpy(
                dtype=np.float64
            )
        )

        # --------------------------------------------------
        # NaN
        # --------------------------------------------------

        nan_per_row = (
            np.isnan(values)
            .any(axis=1)
        )

        current_nan = int(
            nan_per_row.sum()
        )

        valid_with_nan += (
            current_nan
        )

        # --------------------------------------------------
        # Inf
        # --------------------------------------------------

        inf_per_row = (
            np.isinf(values)
            .any(axis=1)
        )

        current_inf = int(
            inf_per_row.sum()
        )

        valid_with_inf += (
            current_inf
        )

        # --------------------------------------------------
        # Integridad por representación
        # --------------------------------------------------

        prefix_incomplete = {}

        for prefix in LANDMARK_PREFIXES:

            cols = (
                landmark_columns(
                    prefix
                )
            )

            arr = (
                numeric[
                    cols
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            incomplete = (
                ~np.isfinite(arr)
            ).any(axis=1)

            prefix_incomplete[
                prefix
            ] = incomplete

        incomplete_img_raw += int(
            prefix_incomplete[
                "img_raw"
            ].sum()
        )

        incomplete_img_norm += int(
            prefix_incomplete[
                "img_norm"
            ].sum()
        )

        incomplete_world_raw += int(
            prefix_incomplete[
                "world_raw"
            ].sum()
        )

        incomplete_world_norm += int(
            prefix_incomplete[
                "world_norm"
            ].sum()
        )

        # ==================================================
        # VALIDACIÓN DE IMAGE_NORM
        # ==================================================

        img_norm_cols = (
            landmark_columns(
                "img_norm"
            )
        )

        img = (
            numeric[
                img_norm_cols
            ]
            .to_numpy(
                dtype=np.float64
            )
            .reshape(
                -1,
                N_LANDMARKS,
                3
            )
        )

        # Muñeca debería ser (0,0,0)
        img_wrist_errors = (
            np.linalg.norm(
                img[:, 0, :],
                axis=1
            )
        )

        finite_img_wrist = (
            np.isfinite(
                img_wrist_errors
            )
        )

        if finite_img_wrist.any():

            max_img_wrist_error = max(
                max_img_wrist_error,
                float(
                    img_wrist_errors[
                        finite_img_wrist
                    ].max()
                )
            )

        img_wrist_bad = (
            ~finite_img_wrist
            |
            (
                img_wrist_errors
                >
                TOLERANCE
            )
        )

        img_wrist_fail += int(
            img_wrist_bad.sum()
        )

        # Distancia 0 -> 9 debería ser 1
        img_d09 = (
            np.linalg.norm(
                img[:, 9, :]
                -
                img[:, 0, :],
                axis=1
            )
        )

        img_scale_errors = (
            np.abs(
                img_d09 - 1.0
            )
        )

        finite_img_scale = (
            np.isfinite(
                img_scale_errors
            )
        )

        if finite_img_scale.any():

            max_img_scale_error = max(
                max_img_scale_error,
                float(
                    img_scale_errors[
                        finite_img_scale
                    ].max()
                )
            )

        img_scale_bad = (
            ~finite_img_scale
            |
            (
                img_scale_errors
                >
                TOLERANCE
            )
        )

        img_scale_fail += int(
            img_scale_bad.sum()
        )

        # ==================================================
        # VALIDACIÓN DE WORLD_NORM
        # ==================================================

        world_norm_cols = (
            landmark_columns(
                "world_norm"
            )
        )

        world = (
            numeric[
                world_norm_cols
            ]
            .to_numpy(
                dtype=np.float64
            )
            .reshape(
                -1,
                N_LANDMARKS,
                3
            )
        )

        world_wrist_errors = (
            np.linalg.norm(
                world[:, 0, :],
                axis=1
            )
        )

        finite_world_wrist = (
            np.isfinite(
                world_wrist_errors
            )
        )

        if finite_world_wrist.any():

            max_world_wrist_error = max(
                max_world_wrist_error,
                float(
                    world_wrist_errors[
                        finite_world_wrist
                    ].max()
                )
            )

        world_wrist_bad = (
            ~finite_world_wrist
            |
            (
                world_wrist_errors
                >
                TOLERANCE
            )
        )

        world_wrist_fail += int(
            world_wrist_bad.sum()
        )

        world_d09 = (
            np.linalg.norm(
                world[:, 9, :]
                -
                world[:, 0, :],
                axis=1
            )
        )

        world_scale_errors = (
            np.abs(
                world_d09 - 1.0
            )
        )

        finite_world_scale = (
            np.isfinite(
                world_scale_errors
            )
        )

        if finite_world_scale.any():

            max_world_scale_error = max(
                max_world_scale_error,
                float(
                    world_scale_errors[
                        finite_world_scale
                    ].max()
                )
            )

        world_scale_bad = (
            ~finite_world_scale
            |
            (
                world_scale_errors
                >
                TOLERANCE
            )
        )

        world_scale_fail += int(
            world_scale_bad.sum()
        )

        # ==================================================
        # GUARDAR EJEMPLOS PROBLEMÁTICOS
        # ==================================================

        any_problem = (
            nan_per_row
            |
            inf_per_row
            |
            img_wrist_bad
            |
            img_scale_bad
            |
            world_wrist_bad
            |
            world_scale_bad
        )

        if any_problem.any():

            problem_indices = (
                np.where(
                    any_problem
                )[0]
            )

            for local_index in problem_indices:

                if (
                    len(problematic_rows)
                    >= 100
                ):
                    break

                source_row = (
                    valid_chunk.iloc[
                        local_index
                    ]
                )

                problematic_rows.append({
                    "class":
                        source_row[
                            "class"
                        ],

                    "group":
                        (
                            source_row[
                                "group"
                            ]
                            if "group"
                            in valid_chunk.columns
                            else ""
                        ),

                    "path":
                        source_row[
                            "path"
                        ],

                    "has_nan":
                        bool(
                            nan_per_row[
                                local_index
                            ]
                        ),

                    "has_inf":
                        bool(
                            inf_per_row[
                                local_index
                            ]
                        ),

                    "img_wrist_error":
                        img_wrist_errors[
                            local_index
                        ],

                    "img_scale_error":
                        img_scale_errors[
                            local_index
                        ],

                    "world_wrist_error":
                        world_wrist_errors[
                            local_index
                        ],

                    "world_scale_error":
                        world_scale_errors[
                            local_index
                        ],
                })

        print(
            f"Chunk {chunk_number:>3} "
            f"| filas leídas: "
            f"{total_rows:>6,} "
            f"| válidas: "
            f"{valid_rows:>6,} "
            f"| fallidas: "
            f"{failed_rows:>5,}"
        )

    # ======================================================
    # CREAR TABLA POR CLASE
    # ======================================================

    class_rows = []

    for label in sorted(
        class_stats
    ):

        s = class_stats[
            label
        ]

        rate = (
            s["valid"]
            /
            s["total"]
            * 100
            if s["total"]
            else 0
        )

        class_rows.append({
            "class":
                label,

            "total":
                s["total"],

            "valid":
                s["valid"],

            "failed":
                s["failed"],

            "extraction_rate":
                rate,
        })

    class_df = pd.DataFrame(
        class_rows
    )

    # ======================================================
    # CREAR TABLA POR GRUPO
    # ======================================================

    group_rows = []

    for (
        label,
        group
    ), s in group_stats.items():

        rate = (
            s["valid"]
            /
            s["total"]
            * 100
            if s["total"]
            else 0
        )

        group_rows.append({
            "class":
                label,

            "group":
                group,

            "total":
                s["total"],

            "valid":
                s["valid"],

            "failed":
                s["failed"],

            "extraction_rate":
                rate,
        })

    group_df = pd.DataFrame(
        group_rows
    )

    if not group_df.empty:

        group_df = (
            group_df
            .sort_values(
                by=[
                    "extraction_rate",
                    "class",
                    "group",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ]
            )
        )

    # ======================================================
    # GUARDAR RESULTADOS
    # ======================================================

    class_output = (
        output_dir
        /
        "lsm_landmarks_by_class.csv"
    )

    group_output = (
        output_dir
        /
        "lsm_landmarks_by_group.csv"
    )

    problems_output = (
        output_dir
        /
        "lsm_landmark_integrity_problems.csv"
    )

    class_df.to_csv(
        class_output,
        index=False,
        encoding="utf-8-sig"
    )

    group_df.to_csv(
        group_output,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        problematic_rows
    ).to_csv(
        problems_output,
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================================
    # RESULTADOS GENERALES
    # ======================================================

    print(
        "\n\n=== RESULTADO GENERAL ==="
    )

    print(
        f"Filas totales       : "
        f"{total_rows:,}"
    )

    print(
        f"Landmarks válidos   : "
        f"{valid_rows:,}"
    )

    print(
        f"Sin landmarks       : "
        f"{failed_rows:,}"
    )

    if total_rows:

        print(
            f"Tasa extracción     : "
            f"{valid_rows / total_rows * 100:.2f}%"
        )

    # ======================================================
    # INTEGRIDAD
    # ======================================================

    print(
        "\n=== INTEGRIDAD NUMÉRICA ==="
    )

    print(
        f"Válidas con NaN             : "
        f"{valid_with_nan:,}"
    )

    print(
        f"Válidas con infinito        : "
        f"{valid_with_inf:,}"
    )

    print(
        f"img_raw incompletas         : "
        f"{incomplete_img_raw:,}"
    )

    print(
        f"img_norm incompletas        : "
        f"{incomplete_img_norm:,}"
    )

    print(
        f"world_raw incompletas       : "
        f"{incomplete_world_raw:,}"
    )

    print(
        f"world_norm incompletas      : "
        f"{incomplete_world_norm:,}"
    )

    # ======================================================
    # NORMALIZACIÓN
    # ======================================================

    print(
        "\n=== VALIDACIÓN NORMALIZACIÓN ==="
    )

    print(
        f"Image muñeca incorrecta : "
        f"{img_wrist_fail:,}"
    )

    print(
        f"Image escala incorrecta : "
        f"{img_scale_fail:,}"
    )

    print(
        f"World muñeca incorrecta : "
        f"{world_wrist_fail:,}"
    )

    print(
        f"World escala incorrecta : "
        f"{world_scale_fail:,}"
    )

    print(
        "\nErrores máximos:"
    )

    print(
        f"  image wrist : "
        f"{max_img_wrist_error:.12f}"
    )

    print(
        f"  image scale : "
        f"{max_img_scale_error:.12f}"
    )

    print(
        f"  world wrist : "
        f"{max_world_wrist_error:.12f}"
    )

    print(
        f"  world scale : "
        f"{max_world_scale_error:.12f}"
    )

    # ======================================================
    # CASCADE
    # ======================================================

    print(
        "\n=== MÉTODO DE DETECCIÓN ==="
    )

    for method, count in (
        method_counter
        .most_common()
    ):

        percentage = (
            count
            /
            valid_rows
            * 100
            if valid_rows
            else 0
        )

        print(
            f"{method:<14} "
            f"{count:>7,} "
            f"{percentage:6.2f}%"
        )

    # ======================================================
    # HANDEDNESS
    # ======================================================

    print(
        "\n=== HANDEDNESS ==="
    )

    for hand, count in (
        handedness_counter
        .most_common()
    ):

        label = (
            hand
            if hand
            else "EMPTY"
        )

        print(
            f"{label:<10} "
            f"{count:>7,}"
        )

    # ======================================================
    # POR CLASE
    # ======================================================

    print(
        "\n=== POR CLASE ==="
    )

    for _, row in (
        class_df
        .iterrows()
    ):

        print(
            f"{row['class']:<2} "
            f"total={int(row['total']):>5,} "
            f"OK={int(row['valid']):>5,} "
            f"FAIL={int(row['failed']):>5,} "
            f"{row['extraction_rate']:6.2f}%"
        )

    # ======================================================
    # GRUPOS MÁS DIFÍCILES
    # ======================================================

    print(
        "\n=== GRUPOS CON MENOR TASA ==="
    )

    if group_df.empty:

        print(
            "No hay información de grupos."
        )

    else:

        worst_groups = (
            group_df
            .head(
                args.show_worst_groups
            )
        )

        for _, row in (
            worst_groups
            .iterrows()
        ):

            print(
                f"{row['class']} | "
                f"{str(row['group']):<20} "
                f"total={int(row['total']):>5,} "
                f"OK={int(row['valid']):>5,} "
                f"FAIL={int(row['failed']):>5,} "
                f"{row['extraction_rate']:6.2f}%"
            )

    # ======================================================
    # ARCHIVOS GENERADOS
    # ======================================================

    print(
        "\n=== ARCHIVOS GENERADOS ==="
    )

    print(
        f"Por clase:"
        f"\n  {class_output}"
    )

    print(
        f"Por grupo:"
        f"\n  {group_output}"
    )

    print(
        f"Problemas de integridad:"
        f"\n  {problems_output}"
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Audita el dataset numérico "
            "de landmarks LSM."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Ruta a lsm_landmarks.csv"
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directorio para los "
            "reportes generados"
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help=(
            "Filas procesadas por bloque."
        )
    )

    parser.add_argument(
        "--show-worst-groups",
        type=int,
        default=25,
        help=(
            "Número de grupos con menor "
            "tasa mostrados en consola."
        )
    )

    main(
        parser.parse_args()
    )
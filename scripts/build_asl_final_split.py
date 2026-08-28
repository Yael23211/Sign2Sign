import argparse
import hashlib
from pathlib import Path

import pandas as pd


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

VALIDATION_RATIO = 0.10
SEED = 42


# ==========================================================
# UTILIDADES
# ==========================================================

def landmarks_ok(value):

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
            "si",
            "sí",
        }
    )


def stable_hash(path, seed):

    value = (
        f"{seed}|{path}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        value
    ).hexdigest()


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{input_path}"
        )

    if (
        output_path.exists()
        and
        not args.overwrite
    ):

        raise FileExistsError(
            f"\nEl archivo ya existe:\n"
            f"{output_path}\n\n"
            f"Usa --overwrite si quieres "
            f"reemplazarlo."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n=== CONSTRUCCIÓN SPLIT FINAL ASL ==="
    )

    print(
        f"Entrada:\n  {input_path}"
    )

    # ======================================================
    # LEER DATASET
    # ======================================================

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    print(
        f"\nFilas cargadas : "
        f"{len(df):,}"
    )

    required = [
        "class",
        "path",
        "landmarks_ok",
        "split_original",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(missing)
        )

    # ======================================================
    # SOLO LANDMARKS VÁLIDOS
    # ======================================================

    valid_mask = (
        df["landmarks_ok"]
        .apply(landmarks_ok)
    )

    valid_df = (
        df[
            valid_mask
        ]
        .copy()
    )

    invalid_count = (
        len(df)
        -
        len(valid_df)
    )

    print(
        "\n=== FILTRO DE LANDMARKS ==="
    )

    print(
        f"Landmarks válidos : "
        f"{len(valid_df):,}"
    )

    print(
        f"Filas omitidas    : "
        f"{invalid_count:,}"
    )

    # ======================================================
    # NORMALIZAR SPLIT ORIGINAL
    # ======================================================

    valid_df[
        "split_original"
    ] = (
        valid_df[
            "split_original"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown_splits = (
        valid_df[
            ~valid_df[
                "split_original"
            ]
            .isin(
                [
                    "train",
                    "test",
                ]
            )
        ]
    )

    if len(
        unknown_splits
    ) > 0:

        print(
            "\nSplits desconocidos:"
        )

        print(
            unknown_splits[
                "split_original"
            ]
            .value_counts(
                dropna=False
            )
        )

        raise ValueError(
            "Se encontraron valores "
            "inesperados en split_original."
        )

    original_train = (
        valid_df[
            valid_df[
                "split_original"
            ]
            .eq("train")
        ]
        .copy()
    )

    original_test = (
        valid_df[
            valid_df[
                "split_original"
            ]
            .eq("test")
        ]
        .copy()
    )

    print(
        "\n=== SPLIT ORIGINAL VÁLIDO ==="
    )

    print(
        f"Train original : "
        f"{len(original_train):,}"
    )

    print(
        f"Test original  : "
        f"{len(original_test):,}"
    )

    # ======================================================
    # INICIALIZAR SPLIT FINAL
    # ======================================================

    valid_df[
        "split_final"
    ] = ""

    # Test original se preserva tal cual.
    valid_df.loc[
        valid_df[
            "split_original"
        ].eq("test"),
        "split_final"
    ] = "test"

    # ======================================================
    # GENERAR VALIDATION POR CLASE
    # ======================================================

    print(
        "\n=== ASIGNACIÓN TRAIN / VALIDATION ==="
    )

    classes = sorted(
        original_train[
            "class"
        ]
        .astype(str)
        .str.upper()
        .unique()
    )

    validation_indices = set()

    for label in classes:

        class_rows = (
            original_train[
                original_train[
                    "class"
                ]
                .astype(str)
                .str.upper()
                .eq(label)
            ]
            .copy()
        )

        n = len(
            class_rows
        )

        if n < 2:

            raise ValueError(
                f"La clase {label} "
                f"solo tiene {n} muestras "
                f"en train original."
            )

        # Aproximadamente 10%.
        n_validation = max(
            1,
            round(
                n
                *
                args.validation_ratio
            )
        )

        # Nunca dejar toda la clase
        # dentro de validation.
        n_validation = min(
            n_validation,
            n - 1
        )

        # ----------------------------------------------
        # Selección determinista
        # ----------------------------------------------
        #
        # No depende del orden del CSV.
        # Cada path recibe un hash estable.
        # ----------------------------------------------

        class_rows[
            "_split_hash"
        ] = (
            class_rows[
                "path"
            ]
            .astype(str)
            .apply(
                lambda p:
                    stable_hash(
                        p,
                        args.seed
                    )
            )
        )

        class_rows = (
            class_rows
            .sort_values(
                "_split_hash"
            )
        )

        selected_validation = (
            class_rows
            .head(
                n_validation
            )
            .index
            .tolist()
        )

        validation_indices.update(
            selected_validation
        )

        n_train = (
            n
            -
            n_validation
        )

        print(
            f"{label:<2} "
            f"original={n:>4,} "
            f"train={n_train:>4,} "
            f"validation="
            f"{n_validation:>3,}"
        )

    # ======================================================
    # APLICAR ASIGNACIONES
    # ======================================================

    train_original_mask = (
        valid_df[
            "split_original"
        ]
        .eq("train")
    )

    valid_df.loc[
        train_original_mask,
        "split_final"
    ] = "train"

    valid_df.loc[
        list(
            validation_indices
        ),
        "split_final"
    ] = "validation"

    # ======================================================
    # VALIDACIONES
    # ======================================================

    print(
        "\n=== VALIDACIONES ==="
    )

    # ----------------------------------------------
    # Test original no debe cambiar
    # ----------------------------------------------

    test_changed = (
        valid_df[
            valid_df[
                "split_original"
            ]
            .eq("test")
        ][
            "split_final"
        ]
        .ne("test")
        .sum()
    )

    print(
        f"Test original movido      : "
        f"{int(test_changed):,}"
    )

    if test_changed != 0:

        raise RuntimeError(
            "Se movieron muestras del "
            "test original."
        )

    # ----------------------------------------------
    # Validation debe venir solo de train original
    # ----------------------------------------------

    invalid_validation = (
        valid_df[
            valid_df[
                "split_final"
            ]
            .eq("validation")
        ][
            "split_original"
        ]
        .ne("train")
        .sum()
    )

    print(
        f"Validation fuera de train : "
        f"{int(invalid_validation):,}"
    )

    if invalid_validation != 0:

        raise RuntimeError(
            "Hay muestras de validation "
            "que no vienen del train original."
        )

    # ----------------------------------------------
    # Ninguna muestra sin split
    # ----------------------------------------------

    no_split = (
        valid_df[
            "split_final"
        ]
        .eq("")
        .sum()
    )

    print(
        f"Filas sin split final     : "
        f"{int(no_split):,}"
    )

    if no_split != 0:

        raise RuntimeError(
            "Hay muestras válidas "
            "sin split_final."
        )

    # ======================================================
    # DISTRIBUCIÓN GLOBAL
    # ======================================================

    print(
        "\n=== DISTRIBUCIÓN GLOBAL ASL ==="
    )

    split_counts = (
        valid_df[
            "split_final"
        ]
        .value_counts()
    )

    total = len(
        valid_df
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        count = int(
            split_counts.get(
                split,
                0
            )
        )

        pct = (
            count
            /
            total
            *
            100
        )

        print(
            f"{split:<11} "
            f"{count:>6,} "
            f"{pct:6.2f}%"
        )

    print(
        f"\nTOTAL       "
        f"{total:>6,}"
    )

    # ======================================================
    # DISTRIBUCIÓN POR CLASE
    # ======================================================

    print(
        "\n=== POR CLASE ==="
    )

    class_split = (
        valid_df
        .groupby(
            [
                "class",
                "split_final",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for col in [
        "train",
        "validation",
        "test",
    ]:

        if col not in class_split.columns:

            class_split[
                col
            ] = 0

    class_split = (
        class_split[
            [
                "train",
                "validation",
                "test",
            ]
        ]
    )

    class_split[
        "total"
    ] = (
        class_split.sum(
            axis=1
        )
    )

    print(
        class_split.to_string()
    )

    # ======================================================
    # COBERTURA
    # ======================================================

    print(
        "\n=== COBERTURA DE CLASES ==="
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = (
            valid_df[
                valid_df[
                    "split_final"
                ]
                .eq(split)
            ]
        )

        split_classes = sorted(
            subset[
                "class"
            ]
            .astype(str)
            .str.upper()
            .unique()
        )

        print(
            f"{split:<11}: "
            f"{len(split_classes)} clases"
        )

    expected_classes = set(
        classes
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        present = set(
            valid_df.loc[
                valid_df[
                    "split_final"
                ]
                .eq(split),
                "class"
            ]
            .astype(str)
            .str.upper()
        )

        missing_classes = (
            expected_classes
            -
            present
        )

        if missing_classes:

            raise RuntimeError(
                f"El split {split} "
                f"no contiene las clases: "
                f"{sorted(missing_classes)}"
            )

    print(
        "\n✅ Las 26 clases aparecen "
        "en train, validation y test."
    )

    # ======================================================
    # GUARDAR
    # ======================================================

    valid_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_path = (
        output_path.parent
        /
        "asl_final_split_summary.csv"
    )

    class_split.to_csv(
        summary_path,
        encoding="utf-8-sig"
    )

    print(
        "\n=== ARCHIVOS GENERADOS ==="
    )

    print(
        f"Dataset final:"
        f"\n  {output_path}"
    )

    print(
        f"\nResumen por clase:"
        f"\n  {summary_path}"
    )

    print(
        "\n✅ Split final ASL construido "
        "correctamente."
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Construye train/validation/test "
            "final de ASL preservando el "
            "test original."
        )
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=VALIDATION_RATIO
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = (
        parser.parse_args()
    )

    if not (
        0
        <
        args.validation_ratio
        <
        1
    ):

        raise ValueError(
            "--validation-ratio "
            "debe estar entre 0 y 1."
        )

    main(args)
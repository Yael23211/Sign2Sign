import argparse
from pathlib import Path

import pandas as pd


# ==========================================================
# CONFIGURACIÓN DEL SPLIT
# ==========================================================

VALIDATION_SESSION = "3"
TEST_SESSION = "6"


def is_provisional(value):

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


def normalize_session(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Pandas a veces puede convertir
    # números a strings tipo "3.0".
    if value.endswith(".0"):

        try:
            return str(
                int(
                    float(value)
                )
            )

        except ValueError:
            pass

    return value


def main(args):

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"No existe:\n{input_path}"
        )

    if output_path.exists() and not args.overwrite:

        raise FileExistsError(
            f"\nEl archivo ya existe:\n"
            f"{output_path}\n\n"
            f"Usa --overwrite si deseas "
            f"reemplazarlo."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # LEER DATASET
    # ======================================================

    print(
        "\n=== CONSTRUCCIÓN SPLIT FINAL LSM ==="
    )

    print(
        f"Entrada:\n  {input_path}"
    )

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
        "group",
        "session_hint",
        "source_dataset",
        "provisional",
        "split_final",
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
    # NORMALIZAR
    # ======================================================

    df["session_hint_clean"] = (
        df["session_hint"]
        .apply(normalize_session)
    )

    provisional_mask = (
        df["provisional"]
        .apply(is_provisional)
    )

    real_mask = (
        ~provisional_mask
    )

    # ======================================================
    # ASIGNACIÓN
    # ======================================================

    # Por defecto todo entra a train.
    df["split_final"] = "train"

    # ------------------------------------------------------
    # LSM genuino de sesión 3 -> validation
    # ------------------------------------------------------

    val_mask = (
        real_mask
        &
        df["session_hint_clean"]
        .eq(
            VALIDATION_SESSION
        )
    )

    df.loc[
        val_mask,
        "split_final"
    ] = "validation"

    # ------------------------------------------------------
    # LSM genuino de sesión 6 -> test
    # ------------------------------------------------------

    test_mask = (
        real_mask
        &
        df["session_hint_clean"]
        .eq(
            TEST_SESSION
        )
    )

    df.loc[
        test_mask,
        "split_final"
    ] = "test"

    # ------------------------------------------------------
    # Provisionales SIEMPRE train
    # ------------------------------------------------------

    df.loc[
        provisional_mask,
        "split_final"
    ] = "train"

    # ======================================================
    # VALIDACIONES
    # ======================================================

    print(
        "\n=== VALIDACIONES ==="
    )

    # ------------------------------------------------------
    # E/L/Z provisionales fuera de train
    # ------------------------------------------------------

    provisional_not_train = (
        provisional_mask
        &
        df["split_final"]
        .ne("train")
    )

    print(
        "Provisionales fuera de train : "
        f"{int(provisional_not_train.sum()):,}"
    )

    if provisional_not_train.any():

        raise RuntimeError(
            "Hay muestras provisionales "
            "fuera de train."
        )

    # ------------------------------------------------------
    # Session 3 únicamente validation
    # ------------------------------------------------------

    session3_splits = (
        df.loc[
            real_mask
            &
            df["session_hint_clean"]
            .eq(
                VALIDATION_SESSION
            ),
            "split_final"
        ]
        .unique()
        .tolist()
    )

    print(
        "Splits de session 3           : "
        f"{session3_splits}"
    )

    # ------------------------------------------------------
    # Session 6 únicamente test
    # ------------------------------------------------------

    session6_splits = (
        df.loc[
            real_mask
            &
            df["session_hint_clean"]
            .eq(
                TEST_SESSION
            ),
            "split_final"
        ]
        .unique()
        .tolist()
    )

    print(
        "Splits de session 6           : "
        f"{session6_splits}"
    )

    if set(session3_splits) != {
        "validation"
    }:

        raise RuntimeError(
            "Session 3 no quedó "
            "exclusivamente en validation."
        )

    if set(session6_splits) != {
        "test"
    }:

        raise RuntimeError(
            "Session 6 no quedó "
            "exclusivamente en test."
        )

    # ======================================================
    # COBERTURA DE CLASES
    # ======================================================

    print(
        "\n=== COBERTURA POR SPLIT ==="
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = (
            df[
                df["split_final"]
                .eq(split)
            ]
        )

        classes = sorted(
            subset[
                "class"
            ]
            .astype(str)
            .unique()
        )

        print(
            f"\n{split.upper()}"
        )

        print(
            f"  muestras : "
            f"{len(subset):,}"
        )

        print(
            f"  clases   : "
            f"{len(classes)}"
        )

        print(
            "  letras   : "
            + ", ".join(classes)
        )

    # ======================================================
    # DISTRIBUCIÓN GLOBAL
    # ======================================================

    print(
        "\n=== DISTRIBUCIÓN GLOBAL ==="
    )

    split_counts = (
        df["split_final"]
        .value_counts()
    )

    total = len(df)

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
            f"{count:>7,} "
            f"{pct:6.2f}%"
        )

    print(
        f"\nTOTAL       "
        f"{total:>7,}"
    )

    # ======================================================
    # DISTRIBUCIÓN POR CLASE
    # ======================================================

    print(
        "\n=== POR CLASE ==="
    )

    class_split = (
        df
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
            class_split[col] = 0

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
    # COMPROBAR LEAKAGE DE SESSION 3/6
    # ======================================================

    print(
        "\n=== AUDITORÍA DE SESIONES HOLDOUT ==="
    )

    audit = (
        df[
            real_mask
            &
            df[
                "session_hint_clean"
            ]
            .isin(
                [
                    VALIDATION_SESSION,
                    TEST_SESSION,
                ]
            )
        ]
        .groupby(
            "session_hint_clean"
        )[
            "split_final"
        ]
        .nunique()
    )

    leakage = (
        audit > 1
    )

    if leakage.any():

        raise RuntimeError(
            "Una sesión holdout aparece "
            "en más de un split."
        )

    print(
        "✅ Session 3 está aislada "
        "en validation."
    )

    print(
        "✅ Session 6 está aislada "
        "en test."
    )

    print(
        "✅ E/L/Z provisionales "
        "permanecen en train."
    )

    # ======================================================
    # ELIMINAR COLUMNA AUXILIAR
    # ======================================================

    df.drop(
        columns=[
            "session_hint_clean"
        ],
        inplace=True
    )

    # ======================================================
    # GUARDAR
    # ======================================================

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Reporte pequeño separado.
    report_path = (
        output_path.parent
        /
        "lsm_final_split_summary.csv"
    )

    class_split.to_csv(
        report_path,
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
        f"\n  {report_path}"
    )

    print(
        "\n✅ Split final construido "
        "correctamente."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Construye el split final "
            "del dataset de landmarks LSM "
            "aislando sesiones completas."
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
        "--overwrite",
        action="store_true"
    )

    main(
        parser.parse_args()
    )
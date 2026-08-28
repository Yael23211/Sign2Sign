import argparse
import itertools
from collections import Counter
from pathlib import Path

import pandas as pd


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

SPLITS = (
    "train",
    "validation",
    "test",
)


# ==========================================================
# UTILIDADES
# ==========================================================

def normalize_bool(value):

    value = (
        str(value)
        .strip()
        .lower()
    )

    return value in {
        "1",
        "true",
        "yes",
        "y",
        "si",
        "sí",
    }


def score_assignment(
    counts,
    total,
    target_train,
    target_val,
    target_test,
):

    if total == 0:
        return float("inf")

    achieved = {
        "train":
            counts["train"] / total,

        "validation":
            counts["validation"] / total,

        "test":
            counts["test"] / total,
    }

    target = {
        "train":
            target_train,

        "validation":
            target_val,

        "test":
            target_test,
    }

    # Error cuadrático respecto a
    # las proporciones deseadas.
    score = 0.0

    for split in SPLITS:

        error = (
            achieved[split]
            -
            target[split]
        )

        score += (
            error ** 2
        )

    return score


# ==========================================================
# ENCONTRAR MEJOR ASIGNACIÓN PARA UNA CLASE
# ==========================================================

def optimize_class_groups(
    class_name,
    group_counts,
    target_train,
    target_val,
    target_test,
):

    groups = list(
        group_counts.keys()
    )

    n_groups = len(groups)

    if n_groups == 0:

        raise ValueError(
            f"La clase {class_name} "
            f"no tiene grupos."
        )

    total = sum(
        group_counts.values()
    )

    # ------------------------------------------------------
    # Caso excepcional:
    # menos de 3 grupos
    # ------------------------------------------------------

    if n_groups < 3:

        # No podemos crear train/val/test
        # sin romper un grupo.
        #
        # Por seguridad todo queda en train
        # y lo reportamos.
        assignment = {
            group: "train"
            for group in groups
        }

        return {
            "assignment":
                assignment,

            "counts": {
                "train":
                    total,

                "validation":
                    0,

                "test":
                    0,
            },

            "score":
                None,

            "warning":
                (
                    "Menos de 3 grupos; "
                    "no es posible generar "
                    "train/validation/test "
                    "sin dividir grupos."
                ),
        }

    # ------------------------------------------------------
    # Búsqueda exhaustiva
    # ------------------------------------------------------
    #
    # Cada grupo puede ir a:
    #
    # 0 -> train
    # 1 -> validation
    # 2 -> test
    #
    # Para los pocos grupos por clase que
    # tiene LSM esto es perfectamente manejable.
    # ------------------------------------------------------

    best_score = None
    best_assignment = None
    best_counts = None

    split_names = {
        0: "train",
        1: "validation",
        2: "test",
    }

    for encoded_assignment in itertools.product(
        range(3),
        repeat=n_groups
    ):

        used = set(
            encoded_assignment
        )

        # Queremos los tres splits presentes.
        if used != {
            0,
            1,
            2,
        }:

            continue

        counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        for group, encoded in zip(
            groups,
            encoded_assignment
        ):

            split = (
                split_names[
                    encoded
                ]
            )

            counts[
                split
            ] += (
                group_counts[
                    group
                ]
            )

        score = score_assignment(
            counts,
            total,
            target_train,
            target_val,
            target_test,
        )

        if (
            best_score is None
            or
            score < best_score
        ):

            best_score = score

            best_counts = (
                counts.copy()
            )

            best_assignment = {
                group:
                    split_names[
                        encoded
                    ]

                for group, encoded
                in zip(
                    groups,
                    encoded_assignment
                )
            }

    return {
        "assignment":
            best_assignment,

        "counts":
            best_counts,

        "score":
            best_score,

        "warning":
            "",
    }


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # LEER SOLO METADATOS
    # ======================================================

    columns = [
        "class",
        "path",
        "group",
        "source_dataset",
        "provisional",
        "provisional_source",
        "split_final",
    ]

    df = pd.read_csv(
        input_path,
        usecols=columns,
        low_memory=False
    )

    print(
        "\n=== PROPUESTA DE SPLIT LSM ==="
    )

    print(
        f"Filas dataset : "
        f"{len(df):,}"
    )

    # ======================================================
    # SEPARAR LSM REAL Y PROVISIONAL
    # ======================================================

    provisional_mask = (
        df["provisional"]
        .apply(normalize_bool)
    )

    provisional_df = (
        df[
            provisional_mask
        ]
        .copy()
    )

    real_df = (
        df[
            ~provisional_mask
        ]
        .copy()
    )

    print(
        f"LSM genuino   : "
        f"{len(real_df):,}"
    )

    print(
        f"Provisionales : "
        f"{len(provisional_df):,}"
    )

    # ======================================================
    # VALIDACIONES
    # ======================================================

    if real_df["group"].isna().any():

        missing = int(
            real_df["group"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"Hay {missing:,} muestras LSM "
            f"sin grupo. No se debe generar "
            f"un split hasta resolverlas."
        )

    empty_groups = (
        real_df["group"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_groups.any():

        raise ValueError(
            f"Hay {int(empty_groups.sum()):,} "
            f"muestras LSM con group vacío."
        )

    # ======================================================
    # SALIDAS
    # ======================================================

    proposal_rows = []
    class_summary_rows = []

    # ======================================================
    # PROCESAR CADA CLASE LSM REAL
    # ======================================================

    real_classes = sorted(
        real_df[
            "class"
        ]
        .astype(str)
        .unique()
    )

    print(
        "\n=== PROPUESTA POR CLASE ==="
    )

    for class_name in real_classes:

        class_df = (
            real_df[
                real_df["class"]
                .astype(str)
                ==
                class_name
            ]
            .copy()
        )

        group_counts = (
            class_df
            .groupby("group")
            .size()
            .to_dict()
        )

        result = (
            optimize_class_groups(
                class_name,
                group_counts,

                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
            )
        )

        assignment = (
            result[
                "assignment"
            ]
        )

        counts = (
            result[
                "counts"
            ]
        )

        total = len(
            class_df
        )

        train_pct = (
            counts["train"]
            / total
            * 100
        )

        val_pct = (
            counts["validation"]
            / total
            * 100
        )

        test_pct = (
            counts["test"]
            / total
            * 100
        )

        print(
            f"\n--- CLASE {class_name} ---"
        )

        print(
            f"Total válido : "
            f"{total:,}"
        )

        print(
            f"Grupos       : "
            f"{len(group_counts)}"
        )

        for group in sorted(
            group_counts
        ):

            split = (
                assignment[
                    group
                ]
            )

            count = (
                group_counts[
                    group
                ]
            )

            print(
                f"  {group:<24} "
                f"{count:>5,} "
                f"-> {split}"
            )

            proposal_rows.append({
                "class":
                    class_name,

                "group":
                    group,

                "samples":
                    count,

                "proposed_split":
                    split,
            })

        print(
            "\nDistribución:"
        )

        print(
            f"  train      "
            f"{counts['train']:>5,} "
            f"({train_pct:6.2f}%)"
        )

        print(
            f"  validation "
            f"{counts['validation']:>5,} "
            f"({val_pct:6.2f}%)"
        )

        print(
            f"  test       "
            f"{counts['test']:>5,} "
            f"({test_pct:6.2f}%)"
        )

        if result["warning"]:

            print(
                f"  ADVERTENCIA: "
                f"{result['warning']}"
            )

        class_summary_rows.append({
            "class":
                class_name,

            "total":
                total,

            "groups":
                len(
                    group_counts
                ),

            "train":
                counts["train"],

            "train_pct":
                train_pct,

            "validation":
                counts[
                    "validation"
                ],

            "validation_pct":
                val_pct,

            "test":
                counts["test"],

            "test_pct":
                test_pct,

            "optimization_score":
                (
                    result["score"]
                    if result["score"]
                    is not None
                    else ""
                ),

            "warning":
                result["warning"],
        })

    # ======================================================
    # PROVISIONALES E / L / Z
    # ======================================================

    print(
        "\n=== CLASES PROVISIONALES ==="
    )

    provisional_counts = (
        provisional_df
        .groupby(
            [
                "class",
                "group",
            ]
        )
        .size()
    )

    for (
        class_name,
        group
    ), count in (
        provisional_counts.items()
    ):

        print(
            f"{class_name} | "
            f"{group:<24} "
            f"{count:>5,} "
            f"-> train_only"
        )

        proposal_rows.append({
            "class":
                class_name,

            "group":
                group,

            "samples":
                int(count),

            "proposed_split":
                "train_only",
        })

    # ======================================================
    # CONSTRUIR RESUMEN GLOBAL
    # ======================================================

    proposal_df = pd.DataFrame(
        proposal_rows
    )

    class_summary_df = (
        pd.DataFrame(
            class_summary_rows
        )
    )

    global_counter = Counter()

    for _, row in (
        proposal_df
        .iterrows()
    ):

        split = (
            row[
                "proposed_split"
            ]
        )

        count = int(
            row[
                "samples"
            ]
        )

        if split == "train_only":

            global_counter[
                "train"
            ] += count

        else:

            global_counter[
                split
            ] += count

    global_total = sum(
        global_counter.values()
    )

    print(
        "\n=== DISTRIBUCIÓN GLOBAL PROPUESTA ==="
    )

    for split in SPLITS:

        count = (
            global_counter[
                split
            ]
        )

        percentage = (
            count
            / global_total
            * 100
            if global_total
            else 0
        )

        print(
            f"{split:<11} "
            f"{count:>7,} "
            f"{percentage:6.2f}%"
        )

    print(
        f"\nTOTAL       "
        f"{global_total:>7,}"
    )

    # ======================================================
    # COMPROBAR LEAKAGE POR CLASS + GROUP
    # ======================================================

    duplicate_assignment = (
        proposal_df
        .groupby(
            [
                "class",
                "group",
            ]
        )[
            "proposed_split"
        ]
        .nunique()
    )

    leaked = (
        duplicate_assignment
        >
        1
    )

    print(
        "\n=== VALIDACIÓN DE GRUPOS ==="
    )

    if leaked.any():

        print(
            "❌ Se encontraron grupos "
            "asignados a más de un split."
        )

    else:

        print(
            "✅ Ningún grupo de una clase "
            "aparece en más de un split."
        )

    # ======================================================
    # GUARDAR REPORTES
    # ======================================================

    proposal_path = (
        output_dir
        /
        "lsm_split_proposal_by_group.csv"
    )

    summary_path = (
        output_dir
        /
        "lsm_split_proposal_by_class.csv"
    )

    proposal_df.to_csv(
        proposal_path,
        index=False,
        encoding="utf-8-sig"
    )

    class_summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n=== ARCHIVOS GENERADOS ==="
    )

    print(
        f"Asignación por grupo:"
        f"\n  {proposal_path}"
    )

    print(
        f"\nResumen por clase:"
        f"\n  {summary_path}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Este script NO modificó "
        "lsm_landmarks_provisional.csv."
    )

    print(
        "La propuesta debe revisarse "
        "antes de escribir split_final."
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Propone un split train/"
            "validation/test para LSM "
            "sin dividir grupos."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Ruta a "
            "lsm_landmarks_provisional.csv"
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directorio donde guardar "
            "los reportes"
        )
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10
    )

    args = (
        parser.parse_args()
    )

    ratio_sum = (
        args.train_ratio
        +
        args.val_ratio
        +
        args.test_ratio
    )

    if abs(
        ratio_sum - 1.0
    ) > 1e-9:

        raise ValueError(
            "train_ratio + val_ratio + "
            "test_ratio debe sumar 1.0"
        )

    main(args)
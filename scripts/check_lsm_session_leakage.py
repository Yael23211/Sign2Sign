import argparse
from pathlib import Path

import pandas as pd


def normalize_bool(value):
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "si",
        "sí",
    }


def main(args):

    dataset_path = Path(args.dataset)
    proposal_path = Path(args.proposal)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not proposal_path.exists():
        raise FileNotFoundError(proposal_path)

    # ======================================================
    # LEER DATASET
    # ======================================================

    df = pd.read_csv(
        dataset_path,
        usecols=[
            "class",
            "group",
            "session_hint",
            "source_dataset",
            "provisional",
        ],
        low_memory=False,
    )

    proposal = pd.read_csv(
        proposal_path,
        low_memory=False,
    )

    # ======================================================
    # SOLO LSM GENUINO
    # ======================================================

    provisional_mask = (
        df["provisional"]
        .apply(normalize_bool)
    )

    real = (
        df[
            ~provisional_mask
        ]
        .copy()
    )

    real = real[
        real["source_dataset"]
        .astype(str)
        .str.upper()
        .eq("LSM")
    ].copy()

    # ======================================================
    # LIMPIAR SESSION_HINT
    # ======================================================

    real["session_hint"] = (
        real["session_hint"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_session = (
        real["session_hint"]
        .eq("")
    )

    print(
        "\n=== AUDITORÍA SESSION_HINT ==="
    )

    print(
        f"Muestras LSM genuinas : {len(real):,}"
    )

    print(
        f"Sin session_hint      : "
        f"{int(empty_session.sum()):,}"
    )

    print(
        f"Con session_hint      : "
        f"{int((~empty_session).sum()):,}"
    )

    # ======================================================
    # UNIR CON PROPUESTA
    # ======================================================

    proposal = proposal[
        [
            "class",
            "group",
            "proposed_split",
        ]
    ].copy()

    merged = real.merge(
        proposal,
        on=[
            "class",
            "group",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_split = (
        merged["proposed_split"]
        .isna()
    )

    print(
        f"Sin split propuesto   : "
        f"{int(missing_split.sum()):,}"
    )

    if missing_split.any():

        print(
            "\n❌ Hay muestras sin asignación."
        )

        print(
            merged.loc[
                missing_split,
                [
                    "class",
                    "group",
                    "session_hint",
                ]
            ]
            .drop_duplicates()
            .head(30)
            .to_string(index=False)
        )

        return

    # ======================================================
    # IGNORAR SESSION_HINT VACÍOS
    # ======================================================

    usable = merged[
        merged["session_hint"]
        .ne("")
    ].copy()

    # ======================================================
    # ¿UNA SESSION APARECE EN VARIOS SPLITS?
    # ======================================================

    session_split_count = (
        usable
        .groupby("session_hint")[
            "proposed_split"
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    leaking_sessions = (
        session_split_count[
            session_split_count > 1
        ]
    )

    print(
        "\nSesiones únicas       : "
        f"{usable['session_hint'].nunique():,}"
    )

    print(
        "Sesiones en >1 split : "
        f"{len(leaking_sessions):,}"
    )

    # ======================================================
    # RESULTADO
    # ======================================================

    if leaking_sessions.empty:

        print(
            "\n✅ NO se detectó leakage "
            "por session_hint."
        )

    else:

        print(
            "\n⚠️ Se detectó posible leakage "
            "por session_hint."
        )

        leaked = usable[
            usable["session_hint"]
            .isin(
                leaking_sessions.index
            )
        ][
            [
                "session_hint",
                "class",
                "group",
                "proposed_split",
            ]
        ].drop_duplicates()

        leaked = leaked.sort_values(
            [
                "session_hint",
                "class",
                "group",
            ]
        )

        print(
            "\n=== SESIONES CONFLICTIVAS ===\n"
        )

        print(
            leaked.to_string(
                index=False
            )
        )

        output = (
            proposal_path.parent
            /
            "lsm_session_leakage.csv"
        )

        leaked.to_csv(
            output,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nReporte generado:\n"
            f"{output}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--proposal",
        required=True,
    )

    main(
        parser.parse_args()
    )
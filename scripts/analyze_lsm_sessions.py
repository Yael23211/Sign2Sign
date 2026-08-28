import pandas as pd
from pathlib import Path


INPUT = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_provisional.csv"
)


def is_provisional(value):

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "si",
        "sí",
    }


df = pd.read_csv(
    INPUT,
    usecols=[
        "class",
        "group",
        "session_hint",
        "pattern",
        "group_quality",
        "source_dataset",
        "provisional",
    ],
    low_memory=False,
)


# ==========================================================
# SOLO LSM GENUINO
# ==========================================================

prov_mask = (
    df["provisional"]
    .apply(is_provisional)
)

real = (
    df[
        ~prov_mask
    ]
    .copy()
)

real = real[
    real["source_dataset"]
    .astype(str)
    .str.upper()
    .eq("LSM")
].copy()


real["session_hint"] = (
    real["session_hint"]
    .fillna("")
    .astype(str)
    .str.strip()
)

real["group"] = (
    real["group"]
    .fillna("")
    .astype(str)
    .str.strip()
)

real["pattern"] = (
    real["pattern"]
    .fillna("")
    .astype(str)
    .str.strip()
)


print(
    "\n=== ESTRUCTURA DE SESIONES LSM ===\n"
)

print(
    f"Muestras LSM genuinas : "
    f"{len(real):,}"
)


# ==========================================================
# SESSION_HINT
# ==========================================================

with_session = (
    real[
        real["session_hint"] != ""
    ]
)

without_session = (
    real[
        real["session_hint"] == ""
    ]
)

print(
    f"Con session_hint      : "
    f"{len(with_session):,}"
)

print(
    f"Sin session_hint      : "
    f"{len(without_session):,}"
)


print(
    "\n=== SESSION_HINT ==="
)

session_summary = (
    with_session
    .groupby("session_hint")
    .agg(
        muestras=("class", "size"),
        clases=("class", "nunique"),
        grupos=("group", "nunique"),
    )
    .sort_values(
        "muestras",
        ascending=False
    )
)

print(
    session_summary.to_string()
)


# ==========================================================
# COBERTURA DE CLASES POR SESIÓN
# ==========================================================

print(
    "\n=== COBERTURA POR SESIÓN ==="
)

for session in sorted(
    with_session[
        "session_hint"
    ].unique()
):

    subset = (
        with_session[
            with_session[
                "session_hint"
            ] == session
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
        f"\nSESSION {session}"
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
        f"  letras   : "
        f"{', '.join(classes)}"
    )


# ==========================================================
# MUESTRAS SIN SESSION_HINT
# ==========================================================

print(
    "\n=== SIN SESSION_HINT: POR PATTERN ==="
)

pattern_summary = (
    without_session
    .groupby("pattern")
    .agg(
        muestras=("class", "size"),
        clases=("class", "nunique"),
        grupos=("group", "nunique"),
    )
    .sort_values(
        "muestras",
        ascending=False
    )
)

print(
    pattern_summary.to_string()
)


print(
    "\n=== SIN SESSION_HINT: POR GROUP_QUALITY ==="
)

quality_summary = (
    without_session
    .groupby("group_quality")
    .agg(
        muestras=("class", "size"),
        clases=("class", "nunique"),
        grupos=("group", "nunique"),
    )
    .sort_values(
        "muestras",
        ascending=False
    )
)

print(
    quality_summary.to_string()
)


# ==========================================================
# GRUPOS SIN SESSION_HINT
# ==========================================================

print(
    "\n=== GRUPOS SIN SESSION_HINT ==="
)

group_summary = (
    without_session
    .groupby(
        [
            "class",
            "group",
            "pattern",
        ]
    )
    .size()
    .reset_index(
        name="muestras"
    )
    .sort_values(
        "muestras",
        ascending=False
    )
)

print(
    group_summary
    .head(50)
    .to_string(index=False)
)


# ==========================================================
# GUARDAR
# ==========================================================

output = (
    Path(
        r"D:\Sign2SignData\inventory\split_analysis"
    )
)

output.mkdir(
    parents=True,
    exist_ok=True
)

session_summary.to_csv(
    output
    / "lsm_sessions_summary.csv",
    encoding="utf-8-sig"
)

pattern_summary.to_csv(
    output
    / "lsm_no_session_patterns.csv",
    encoding="utf-8-sig"
)

group_summary.to_csv(
    output
    / "lsm_no_session_groups.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n✅ Auditoría terminada."
)
import pandas as pd

CSV_PATH = r"D:\Sign2SignData\processed\landmarks\asl_landmarks.csv"

df = pd.read_csv(CSV_PATH)

print("=== ASL LANDMARKS: TRAIN vs TEST ===\n")

summary = (
    df.groupby(["class", "split_original"])
      .agg(
          total=("landmarks_ok", "size"),
          validas=("landmarks_ok", "sum")
      )
      .reset_index()
)

summary["fallidas"] = (
    summary["total"] - summary["validas"]
)

summary["tasa"] = (
    summary["validas"]
    / summary["total"]
    * 100
)

print(
    summary.to_string(
        index=False,
        formatters={
            "tasa": lambda x: f"{x:.2f}%"
        }
    )
)

print("\n=== RESUMEN POR SPLIT ===\n")

split_summary = (
    df.groupby("split_original")
      .agg(
          total=("landmarks_ok", "size"),
          validas=("landmarks_ok", "sum")
      )
)

split_summary["fallidas"] = (
    split_summary["total"]
    - split_summary["validas"]
)

split_summary["tasa"] = (
    split_summary["validas"]
    / split_summary["total"]
    * 100
)

print(split_summary)
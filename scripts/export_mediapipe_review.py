import csv
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


# =========================
# CONFIG RÁPIDA
# =========================
ZIP_LSM = r"D:\Sign2SignData\raw\lsm\LSM_original.zip"

CSV_FILES = [
    r"D:\Sign2SignData\inventory\mediapipe_smoke\lsm_A_detection.csv",
    r"D:\Sign2SignData\inventory\mediapipe_smoke\lsm_M_detection.csv",
    r"D:\Sign2SignData\inventory\mediapipe_smoke\lsm_N_detection.csv",
]

OUT_DIR = r"D:\Sign2SignData\inventory\mediapipe_review"
MAX_OK_PER_GROUP = 3
MAX_FAIL_PER_GROUP = 3


# =========================
# HELPERS
# =========================
def find_col(fieldnames, candidates):
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def sanitize(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def infer_group_from_path(path_str):
    """
    Intenta inferir el grupo a partir del nombre del archivo.
    Ejemplos:
      TT/dataset/N/N1_0238.jpg       -> N1_NUM
      TT/dataset/N/N3_0149.jpg       -> N3_NUM
      TT/dataset/N/Fram_N6_382.jpg   -> FRAM_N6
      TT/dataset/N/N5_Frame238.jpg   -> N5_FRAME
      TT/dataset/N/frame_353.jpg     -> GENERIC_FRAME
      TT/dataset/A/IMG_...BURST016   -> PHONE_BURST
    """
    name = Path(path_str).name
    stem = Path(path_str).stem

    upper = stem.upper()

    if "BURST" in upper:
        return "PHONE_BURST"

    if upper.startswith("FRAME_"):
        return "GENERIC_FRAME"

    # N1_0238, A3_0004, M1_0628
    import re
    m = re.match(r"^([A-Z])(\d+)_\d+$", upper)
    if m:
        return f"{m.group(1)}{m.group(2)}_NUM"

    # Fram_N6_382
    m = re.match(r"^FRAM_([A-Z])(\d+)_(\d+)$", upper)
    if m:
        return f"FRAM_{m.group(1)}{m.group(2)}"

    # N5_Frame238
    m = re.match(r"^([A-Z])(\d+)_FRAME(\d+)$", upper)
    if m:
        return f"{m.group(1)}{m.group(2)}_FRAME"

    return "UNKNOWN"


def infer_letter_from_path(path_str):
    parts = Path(path_str).parts
    # buscar algo como .../dataset/A/archivo.jpg
    for i, p in enumerate(parts):
        if p.lower() == "dataset" and i + 1 < len(parts):
            return parts[i + 1].upper()
    return "UNKNOWN"


def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        path_col = find_col(reader.fieldnames, [
            "image_path", "path", "rel_path", "file", "archivo"
        ])
        detected_col = find_col(reader.fieldnames, [
            "detected", "ok", "success", "mano_detectada"
        ])
        group_col = find_col(reader.fieldnames, [
            "group", "group_name", "source_group", "pattern_group"
        ])
        class_col = find_col(reader.fieldnames, [
            "class", "class_name", "letter", "label", "clase"
        ])

        if not path_col or not detected_col:
            raise ValueError(
                f"No encontré columnas mínimas en {csv_path}. "
                f"Fieldnames: {reader.fieldnames}"
            )

        for row in reader:
            path_str = row[path_col].strip()
            raw_detected = str(row[detected_col]).strip().lower()

            detected = raw_detected in {"1", "true", "yes", "ok", "si", "sí"}

            group_name = row[group_col].strip() if group_col and row.get(group_col) else infer_group_from_path(path_str)
            letter = row[class_col].strip().upper() if class_col and row.get(class_col) else infer_letter_from_path(path_str)

            rows.append({
                "csv": csv_path,
                "path": path_str,
                "detected": detected,
                "group": group_name,
                "letter": letter,
            })

    return rows


def export_from_zip(zip_path, rel_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        data = z.read(rel_path)
    with open(out_path, "wb") as f:
        f.write(data)


# =========================
# MAIN
# =========================
def main():
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            print(f"[WARN] No existe: {csv_file}")
            continue
        rows = read_rows(csv_file)
        print(f"[OK] {csv_file}: {len(rows)} filas")
        all_rows.extend(rows)

    if not all_rows:
        print("No se cargaron filas.")
        return

    bucket = defaultdict(lambda: {"ok": [], "fail": []})

    for r in all_rows:
        key = (r["letter"], r["group"])
        if r["detected"]:
            if len(bucket[key]["ok"]) < MAX_OK_PER_GROUP:
                bucket[key]["ok"].append(r)
        else:
            if len(bucket[key]["fail"]) < MAX_FAIL_PER_GROUP:
                bucket[key]["fail"].append(r)

    exported = 0

    for (letter, group), data in sorted(bucket.items()):
        # Solo nos interesan grupos donde haya al menos un fail o un ok comparativo
        if not data["ok"] and not data["fail"]:
            continue

        base = out_root / letter / sanitize(group)

        for status_name, items in [("ok", data["ok"]), ("fail", data["fail"])]:
            for i, item in enumerate(items, start=1):
                rel_path = item["path"]
                fname = Path(rel_path).name
                out_path = base / status_name / f"{i:02d}_{fname}"
                try:
                    export_from_zip(ZIP_LSM, rel_path, out_path)
                    exported += 1
                except Exception as e:
                    print(f"[ERROR] {rel_path}: {e}")

    print("\n=== EXPORT COMPLETO ===")
    print(f"Imágenes exportadas: {exported}")
    print(f"Salida: {OUT_DIR}")


if __name__ == "__main__":
    main()
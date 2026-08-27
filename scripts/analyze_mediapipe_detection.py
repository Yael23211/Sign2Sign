import csv
import sys
from collections import defaultdict


def main(results_path, manifest_path):

    # -------------------------------------------------
    # Leer manifest
    # -------------------------------------------------

    manifest = {}

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            manifest[row["path"]] = row

    # -------------------------------------------------
    # Leer resultados MediaPipe
    # -------------------------------------------------

    groups = defaultdict(
        lambda: {
            "total": 0,
            "detected": 0,
            "failed": 0,
            "success_examples": [],
            "fail_examples": [],
        }
    )

    unmatched = []

    with open(
        results_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for result in reader:

            path = result["path"]

            if path not in manifest:
                unmatched.append(path)
                continue

            meta = manifest[path]

            group = meta["group"]
            detected = result["detected"] == "1"

            data = groups[group]

            data["total"] += 1

            if detected:

                data["detected"] += 1

                if len(data["success_examples"]) < 3:
                    data["success_examples"].append(path)

            else:

                data["failed"] += 1

                if len(data["fail_examples"]) < 3:
                    data["fail_examples"].append(path)

    # -------------------------------------------------
    # Mostrar resultados
    # -------------------------------------------------

    print("\n=== DETECCIÓN MEDIAPIPE POR GRUPO ===\n")

    rows = []

    for group, data in groups.items():

        rate = (
            data["detected"]
            / data["total"]
            * 100
        )

        rows.append(
            (
                group,
                data["total"],
                data["detected"],
                data["failed"],
                rate,
                data
            )
        )

    rows.sort(
        key=lambda x: x[4]
    )

    for (
        group,
        total,
        detected,
        failed,
        rate,
        data
    ) in rows:

        print(
            f"{group:<24}"
            f"{detected:>3}/{total:<3} "
            f"{rate:>6.2f}%"
        )

        if data["fail_examples"]:

            print("  FAIL:")

            for example in data["fail_examples"]:
                print(
                    f"    {example}"
                )

        if data["success_examples"]:

            print("  OK:")

            for example in data["success_examples"]:
                print(
                    f"    {example}"
                )

        print()

    # -------------------------------------------------
    # Resumen
    # -------------------------------------------------

    total = sum(
        data["total"]
        for data in groups.values()
    )

    detected = sum(
        data["detected"]
        for data in groups.values()
    )

    print("=== RESUMEN ===")

    print(
        f"Imágenes cruzadas : {total}"
    )

    print(
        f"Detectadas        : {detected}"
    )

    if total:

        print(
            f"Tasa global       : "
            f"{detected / total * 100:.2f}%"
        )

    print(
        f"Grupos presentes  : "
        f"{len(groups)}"
    )

    print(
        f"Sin coincidencia  : "
        f"{len(unmatched)}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso:\n"
            "python analyze_mediapipe_detection.py "
            "\"resultados.csv\" "
            "\"lsm_manifest_clean.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2]
    )
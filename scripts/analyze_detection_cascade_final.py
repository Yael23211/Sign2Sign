import csv
import sys
from collections import defaultdict


CASCADE_ORDER = [
    "baseline",
    "center_4_3",
    "crop_85",
    "crop_70",
    "letterbox",
    "stretch",
]


def read_results(csv_path):

    data = {}

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            key = (
                row["class"],
                row["path"]
            )

            variant = row["variant"]

            detected = (
                row["detected"] == "1"
            )

            if key not in data:

                data[key] = {
                    "class": row["class"],
                    "path": row["path"],
                    "group": row["group"],
                    "results": {}
                }

            # Si baseline aparece en dos experimentos,
            # comprobamos que no haya contradicción.
            if (
                variant in data[key]["results"]
                and
                data[key]["results"][variant]
                != detected
            ):

                print(
                    "[WARN] Resultado diferente para "
                    f"{key} / {variant}"
                )

            data[key]["results"][
                variant
            ] = detected

    return data


def merge_results(*datasets):

    merged = {}

    for dataset in datasets:

        for key, item in dataset.items():

            if key not in merged:

                merged[key] = {
                    "class": item["class"],
                    "path": item["path"],
                    "group": item["group"],
                    "results": {}
                }

            merged[key]["results"].update(
                item["results"]
            )

    return merged


def main(
    crop_csv,
    geometry_csv,
    output_csv
):

    crop_data = read_results(
        crop_csv
    )

    geometry_data = read_results(
        geometry_csv
    )

    images = merge_results(
        crop_data,
        geometry_data
    )

    class_stats = defaultdict(
        lambda: {
            "total": 0,
            "baseline": 0,
            "cascade": 0,
            "failed": 0,
            "method_usage": defaultdict(int),
        }
    )

    group_stats = defaultdict(
        lambda: {
            "total": 0,
            "baseline": 0,
            "cascade": 0,
            "failed": 0,
        }
    )

    output_rows = []

    # =========================================
    # SIMULAR CASCADE
    # =========================================

    for item in images.values():

        label = item["class"]
        path = item["path"]
        group = item["group"]
        results = item["results"]

        baseline_ok = results.get(
            "baseline",
            False
        )

        selected_method = None

        for method in CASCADE_ORDER:

            if results.get(
                method,
                False
            ):

                selected_method = method
                break

        cascade_ok = (
            selected_method is not None
        )

        # -------------------------------------
        # POR CLASE
        # -------------------------------------

        cs = class_stats[label]

        cs["total"] += 1
        cs["baseline"] += int(
            baseline_ok
        )

        cs["cascade"] += int(
            cascade_ok
        )

        if cascade_ok:

            cs["method_usage"][
                selected_method
            ] += 1

        else:

            cs["failed"] += 1

        # -------------------------------------
        # POR GRUPO
        # -------------------------------------

        gs = group_stats[
            (label, group)
        ]

        gs["total"] += 1
        gs["baseline"] += int(
            baseline_ok
        )

        gs["cascade"] += int(
            cascade_ok
        )

        gs["failed"] += int(
            not cascade_ok
        )

        # -------------------------------------
        # CSV FINAL
        # -------------------------------------

        row = {
            "class": label,
            "path": path,
            "group": group,
            "baseline_detected":
                int(baseline_ok),
            "cascade_detected":
                int(cascade_ok),
            "selected_method":
                selected_method
                if selected_method
                else "FAILED",
        }

        for method in CASCADE_ORDER:

            row[
                f"{method}_detected"
            ] = int(
                results.get(
                    method,
                    False
                )
            )

        output_rows.append(
            row
        )

    # =========================================
    # RESULTADOS POR CLASE
    # =========================================

    print(
        "\n=== CASCADE FINAL POR CLASE ===\n"
    )

    for label in sorted(
        class_stats
    ):

        s = class_stats[label]

        total = s["total"]

        baseline_rate = (
            s["baseline"]
            / total
            * 100
        )

        cascade_rate = (
            s["cascade"]
            / total
            * 100
        )

        rescued = (
            s["cascade"]
            - s["baseline"]
        )

        print(
            f"--- {label} ---"
        )

        print(
            f"Baseline     : "
            f"{s['baseline']}/{total} "
            f"({baseline_rate:.2f}%)"
        )

        print(
            f"Cascade      : "
            f"{s['cascade']}/{total} "
            f"({cascade_rate:.2f}%)"
        )

        print(
            f"Rescatadas   : "
            f"{rescued}"
        )

        print(
            f"Sin detectar : "
            f"{s['failed']}"
        )

        print(
            "Método seleccionado:"
        )

        for method in CASCADE_ORDER:

            count = (
                s["method_usage"][
                    method
                ]
            )

            print(
                f"  {method:<12} "
                f"{count}"
            )

        print()

    # =========================================
    # RESULTADOS POR GRUPO
    # =========================================

    print(
        "\n=== CASCADE FINAL POR GRUPO ===\n"
    )

    current_class = None

    for (
        label,
        group
    ), s in sorted(
        group_stats.items()
    ):

        if label != current_class:

            current_class = label

            print(
                f"\n--- CLASE {label} ---"
            )

        total = s["total"]

        b_rate = (
            s["baseline"]
            / total
            * 100
        )

        c_rate = (
            s["cascade"]
            / total
            * 100
        )

        rescued = (
            s["cascade"]
            - s["baseline"]
        )

        print(
            f"{group:<22} "
            f"{s['baseline']:>2}/{total:<2} "
            f"{b_rate:6.2f}%"
            f" -> "
            f"{s['cascade']:>2}/{total:<2} "
            f"{c_rate:6.2f}%"
            f"  (+{rescued})"
        )

    # =========================================
    # IMÁGENES QUE SIGUEN FALLANDO
    # =========================================

    print(
        "\n=== IMÁGENES NO RECUPERADAS ===\n"
    )

    failures = [
        row
        for row in output_rows
        if row["cascade_detected"] == 0
    ]

    if not failures:

        print(
            "Todas las imágenes fueron "
            "detectadas por alguna técnica."
        )

    else:

        for row in failures:

            print(
                f"{row['class']} | "
                f"{row['group']} | "
                f"{row['path']}"
            )

    # =========================================
    # GUARDAR CSV
    # =========================================

    fieldnames = [
        "class",
        "path",
        "group",
        "baseline_detected",
        "cascade_detected",
        "selected_method",
    ]

    fieldnames += [
        f"{method}_detected"
        for method in CASCADE_ORDER
    ]

    with open(
        output_csv,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    # =========================================
    # GLOBAL
    # =========================================

    total = sum(
        s["total"]
        for s in class_stats.values()
    )

    baseline = sum(
        s["baseline"]
        for s in class_stats.values()
    )

    cascade = sum(
        s["cascade"]
        for s in class_stats.values()
    )

    print(
        "\n=== RESULTADO TOTAL ==="
    )

    print(
        f"Imágenes evaluadas : "
        f"{total}"
    )

    print(
        f"Baseline           : "
        f"{baseline}/{total} "
        f"({baseline / total * 100:.2f}%)"
    )

    print(
        f"Cascade            : "
        f"{cascade}/{total} "
        f"({cascade / total * 100:.2f}%)"
    )

    print(
        f"Rescatadas         : "
        f"{cascade - baseline}"
    )

    print(
        f"No recuperadas     : "
        f"{total - cascade}"
    )

    print(
        f"\nCSV generado: "
        f"{output_csv}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:

        print(
            "Uso:\n"
            "python "
            "analyze_detection_cascade_final.py "
            "\"crop_comparison.csv\" "
            "\"geometry_comparison.csv\" "
            "\"salida.csv\""
        )

        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3]
    )
import csv
import sys
from collections import defaultdict


VARIANTS = [
    "baseline",
    "crop_85",
    "crop_70",
]


def main(csv_path):

    images = {}

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

            if key not in images:

                images[key] = {
                    "class": row["class"],
                    "path": row["path"],
                    "group": row["group"],
                    "results": {}
                }

            images[key]["results"][
                row["variant"]
            ] = row["detected"] == "1"

    # -----------------------------------------
    # RESUMEN POR CLASE
    # -----------------------------------------

    class_stats = defaultdict(
        lambda: {
            "total": 0,
            "baseline": 0,
            "baseline_85": 0,
            "baseline_70": 0,
            "cascade": 0,
            "rescued_85": 0,
            "rescued_70": 0,
        }
    )

    # -----------------------------------------
    # RESUMEN POR GRUPO
    # -----------------------------------------

    group_stats = defaultdict(
        lambda: {
            "total": 0,
            "baseline": 0,
            "cascade": 0,
            "rescued": 0,
        }
    )

    rescued_examples = []

    for item in images.values():

        label = item["class"]
        group = item["group"]

        r = item["results"]

        baseline = r.get(
            "baseline",
            False
        )

        crop85 = r.get(
            "crop_85",
            False
        )

        crop70 = r.get(
            "crop_70",
            False
        )

        # -------------------------------------
        # Fallbacks
        # -------------------------------------

        baseline_85 = (
            baseline
            or crop85
        )

        baseline_70 = (
            baseline
            or crop70
        )

        cascade = (
            baseline
            or crop85
            or crop70
        )

        stats = class_stats[label]

        stats["total"] += 1
        stats["baseline"] += int(baseline)
        stats["baseline_85"] += int(
            baseline_85
        )
        stats["baseline_70"] += int(
            baseline_70
        )
        stats["cascade"] += int(
            cascade
        )

        # Rescatada específicamente por crop85
        if not baseline and crop85:

            stats["rescued_85"] += 1

        # Rescatada por crop70 y no por crop85
        if (
            not baseline
            and not crop85
            and crop70
        ):

            stats["rescued_70"] += 1

        gs = group_stats[
            (label, group)
        ]

        gs["total"] += 1
        gs["baseline"] += int(
            baseline
        )

        gs["cascade"] += int(
            cascade
        )

        if not baseline and cascade:

            gs["rescued"] += 1

            rescued_examples.append({
                "class": label,
                "group": group,
                "path": item["path"],
                "crop85": crop85,
                "crop70": crop70,
            })

    # =========================================
    # RESULTADO GLOBAL
    # =========================================

    print(
        "\n=== FALLBACK / CASCADE POR CLASE ===\n"
    )

    for label in sorted(class_stats):

        s = class_stats[label]
        total = s["total"]

        def show(name, value):

            rate = value / total * 100

            print(
                f"{name:<18}"
                f"{value:>3}/{total:<3} "
                f"{rate:6.2f}%"
            )

        print(
            f"--- {label} ---"
        )

        show(
            "baseline",
            s["baseline"]
        )

        show(
            "baseline + 85",
            s["baseline_85"]
        )

        show(
            "baseline + 70",
            s["baseline_70"]
        )

        show(
            "cascade completa",
            s["cascade"]
        )

        print(
            f"Rescatadas crop85 : "
            f"{s['rescued_85']}"
        )

        print(
            f"Rescatadas crop70 : "
            f"{s['rescued_70']}"
        )

        print()

    # =========================================
    # RESULTADO POR GRUPO
    # =========================================

    print(
        "\n=== CASCADE POR GRUPO ===\n"
    )

    current_class = None

    for (
        label,
        group
    ), s in sorted(group_stats.items()):

        if label != current_class:

            current_class = label

            print(
                f"\n--- CLASE {label} ---"
            )

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

        print(
            f"{group:<22} "
            f"{s['baseline']:>2}/{total:<2} "
            f"{baseline_rate:6.2f}%"
            f"  ->  "
            f"{s['cascade']:>2}/{total:<2} "
            f"{cascade_rate:6.2f}%"
            f"  (+{s['rescued']})"
        )

    # =========================================
    # EJEMPLOS RESCATADOS
    # =========================================

    print(
        "\n=== IMÁGENES RESCATADAS ===\n"
    )

    for row in rescued_examples:

        method = (
            "crop_85"
            if row["crop85"]
            else "crop_70"
        )

        print(
            f"{row['class']} | "
            f"{row['group']} | "
            f"{method} | "
            f"{row['path']}"
        )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Uso:\n"
            "python analyze_crop_cascade.py "
            "\"crop_comparison.csv\""
        )

        sys.exit(1)

    main(sys.argv[1])
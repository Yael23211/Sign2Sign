import argparse
import csv
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# Permitir imports desde raíz del repo.
REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(REPO_ROOT)
)


from src.landmarks.pipeline import (
    CASCADE_ORDER,
    add_points_to_row,
    create_hand_landmarker,
    detect_with_cascade,
    image_landmarks_to_arrays,
    landmark_fieldnames,
    world_landmarks_to_arrays,
)


# ==========================================================
# MANIFEST
# ==========================================================

def find_column(
    fieldnames,
    candidates
):

    normalized = {
        name.lower(): name
        for name in fieldnames
    }

    for candidate in candidates:

        if candidate.lower() in normalized:
            return normalized[
                candidate.lower()
            ]

    return None


def read_asl_manifest(
    manifest_path
):

    rows = []

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fields = reader.fieldnames

        path_col = find_column(
            fields,
            [
                "path",
                "image_path",
                "zip_path",
                "filename",
            ]
        )

        class_col = find_column(
            fields,
            [
                "class",
                "class_name",
                "label",
                "letter",
            ]
        )

        split_col = find_column(
            fields,
            [
                "split_original",
                "split",
            ]
        )

        type_col = find_column(
            fields,
            [
                "type",
                "sample_type",
            ]
        )

        if not path_col:
            raise ValueError(
                "No encontré columna de path."
            )

        if not class_col:
            raise ValueError(
                "No encontré columna de clase."
            )

        for row in reader:

            label = (
                row[class_col]
                .strip()
            )

            sample_type = (
                row[type_col].strip()
                if type_col
                else ""
            )

            # Blank queda fuera del clasificador.
            if label.lower() == "blank":
                continue

            if (
                sample_type
                and
                sample_type.lower()
                not in {
                    "letter",
                    "positive",
                }
            ):
                continue

            rows.append({
                "path":
                    row[path_col].strip(),
                "class":
                    label.upper(),
                "split_original":
                    (
                        row[split_col].strip()
                        if split_col
                        else ""
                    ),
            })

    return rows


# ==========================================================
# CSV
# ==========================================================

def build_fieldnames():

    fields = [
        "dataset",
        "class",
        "path",
        "split_original",

        "landmarks_ok",
        "detection_variant",

        "original_width",
        "original_height",

        "processed_width",
        "processed_height",

        "handedness",
        "handedness_score",

        "image_scale_0_9",
        "world_scale_0_9",
    ]

    fields += (
        landmark_fieldnames()
    )

    return fields


def existing_paths(
    output_path
):

    path = Path(
        output_path
    )

    if not path.exists():
        return set()

    found = set()

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            found.add(
                row["path"]
            )

    return found


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    manifest_rows = (
        read_asl_manifest(
            args.manifest
        )
    )

    print(
        "\n=== EXTRACCIÓN ASL ==="
    )

    print(
        f"Muestras del manifest : "
        f"{len(manifest_rows):,}"
    )

    already_done = (
        existing_paths(
            args.output
        )
    )

    print(
        f"Ya procesadas         : "
        f"{len(already_done):,}"
    )

    pending = [
        row
        for row in manifest_rows
        if row["path"]
        not in already_done
    ]

    if args.max_new:

        pending = pending[
            :args.max_new
        ]

    print(
        f"A procesar ahora      : "
        f"{len(pending):,}"
    )

    if not pending:

        print(
            "No hay muestras pendientes."
        )
        return

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        output_path.exists()
        and output_path.stat().st_size > 0
    )

    fieldnames = (
        build_fieldnames()
    )

    method_counter = Counter()
    class_valid = Counter()
    class_failed = Counter()

    valid = 0
    failed = 0

    start_time = time.time()

    with zipfile.ZipFile(
        args.zip,
        "r"
    ) as z:

        with create_hand_landmarker(
            args.model
        ) as landmarker:

            with open(
                output_path,
                "a",
                encoding="utf-8-sig",
                newline=""
            ) as out:

                writer = csv.DictWriter(
                    out,
                    fieldnames=fieldnames,
                    extrasaction="ignore"
                )

                if not file_exists:
                    writer.writeheader()

                for idx, meta in enumerate(
                    pending,
                    start=1
                ):

                    row = {
                        "dataset": "ASL",
                        "class": meta["class"],
                        "path": meta["path"],
                        "split_original":
                            meta[
                                "split_original"
                            ],
                    }

                    try:

                        raw = z.read(
                            meta["path"]
                        )

                        arr = np.frombuffer(
                            raw,
                            dtype=np.uint8
                        )

                        original = (
                            cv2.imdecode(
                                arr,
                                cv2.IMREAD_COLOR
                            )
                        )

                        if original is None:

                            raise ValueError(
                                "cv2.imdecode "
                                "devolvió None"
                            )

                        oh, ow = (
                            original.shape[:2]
                        )

                        row[
                            "original_width"
                        ] = ow

                        row[
                            "original_height"
                        ] = oh

                        detection = (
                            detect_with_cascade(
                                landmarker,
                                original
                            )
                        )

                        if not detection["ok"]:

                            row[
                                "landmarks_ok"
                            ] = 0

                            row[
                                "detection_variant"
                            ] = "FAILED"

                            failed += 1

                            class_failed[
                                meta["class"]
                            ] += 1

                        else:

                            result = (
                                detection["result"]
                            )

                            processed = (
                                detection[
                                    "processed"
                                ]
                            )

                            variant = (
                                detection[
                                    "variant"
                                ]
                            )

                            ph, pw = (
                                processed.shape[:2]
                            )

                            row[
                                "landmarks_ok"
                            ] = 1

                            row[
                                "detection_variant"
                            ] = variant

                            row[
                                "processed_width"
                            ] = pw

                            row[
                                "processed_height"
                            ] = ph

                            # ------------------
                            # handedness
                            # ------------------

                            if result.handedness:

                                category = (
                                    result
                                    .handedness[0][0]
                                )

                                row[
                                    "handedness"
                                ] = (
                                    category
                                    .category_name
                                )

                                row[
                                    "handedness_score"
                                ] = float(
                                    category.score
                                )

                            # ------------------
                            # image landmarks
                            # ------------------

                            (
                                img_raw,
                                img_norm,
                                img_scale
                            ) = (
                                image_landmarks_to_arrays(
                                    result
                                    .hand_landmarks[0],
                                    pw,
                                    ph
                                )
                            )

                            row[
                                "image_scale_0_9"
                            ] = float(
                                img_scale
                            )

                            add_points_to_row(
                                row,
                                "img_raw",
                                img_raw
                            )

                            add_points_to_row(
                                row,
                                "img_norm",
                                img_norm
                            )

                            # ------------------
                            # world landmarks
                            # ------------------

                            if (
                                result
                                .hand_world_landmarks
                            ):

                                (
                                    world_raw,
                                    world_norm,
                                    world_scale
                                ) = (
                                    world_landmarks_to_arrays(
                                        result
                                        .hand_world_landmarks[0]
                                    )
                                )

                                row[
                                    "world_scale_0_9"
                                ] = float(
                                    world_scale
                                )

                                add_points_to_row(
                                    row,
                                    "world_raw",
                                    world_raw
                                )

                                add_points_to_row(
                                    row,
                                    "world_norm",
                                    world_norm
                                )

                            valid += 1

                            class_valid[
                                meta["class"]
                            ] += 1

                            method_counter[
                                variant
                            ] += 1

                    except Exception as e:

                        row[
                            "landmarks_ok"
                        ] = 0

                        row[
                            "detection_variant"
                        ] = (
                            f"ERROR:{type(e).__name__}"
                        )

                        failed += 1

                        class_failed[
                            meta["class"]
                        ] += 1

                        print(
                            f"\n[ERROR] "
                            f"{meta['path']}"
                        )

                        print(
                            f"        {e}"
                        )

                    writer.writerow(
                        row
                    )

                    # Permite recuperar progreso
                    # incluso si se interrumpe.
                    out.flush()

                    if (
                        idx % 100 == 0
                        or idx == len(pending)
                    ):

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        rate = (
                            idx / elapsed
                            if elapsed > 0
                            else 0
                        )

                        print(
                            f"Procesadas: "
                            f"{idx:,}/"
                            f"{len(pending):,}"
                            f" | OK: {valid:,}"
                            f" | FAIL: {failed:,}"
                            f" | {rate:.2f} img/s"
                        )

    # ==================================================
    # RESUMEN
    # ==================================================

    total = valid + failed

    print(
        "\n=== RESULTADO ASL ==="
    )

    print(
        f"Procesadas          : "
        f"{total:,}"
    )

    print(
        f"Landmarks válidos   : "
        f"{valid:,}"
    )

    print(
        f"Sin landmarks       : "
        f"{failed:,}"
    )

    if total:

        print(
            f"Tasa extracción     : "
            f"{valid / total * 100:.2f}%"
        )

    print(
        "\nMétodo seleccionado:"
    )

    for method in CASCADE_ORDER:

        print(
            f"  {method:<12} "
            f"{method_counter[method]:,}"
        )

    print(
        "\nPor clase:"
    )

    classes = sorted(
        set(
            list(class_valid.keys())
            +
            list(class_failed.keys())
        )
    )

    for label in classes:

        ok = class_valid[label]
        fail = class_failed[label]

        total_class = (
            ok + fail
        )

        rate = (
            ok / total_class * 100
            if total_class
            else 0
        )

        print(
            f"  {label:<2} "
            f"OK={ok:>4} "
            f"FAIL={fail:>4} "
            f"{rate:6.2f}%"
        )

    print(
        f"\nSalida: "
        f"{output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        required=True
    )

    parser.add_argument(
        "--manifest",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help=(
            "Procesa como máximo N muestras "
            "nuevas. Útil para smoke test."
        )
    )

    main(
        parser.parse_args()
    )
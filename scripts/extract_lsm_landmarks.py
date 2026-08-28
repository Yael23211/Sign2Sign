import argparse
import csv
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# ==========================================================
# IMPORTAR MÓDULO COMÚN DEL PROYECTO
# ==========================================================

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
# UTILIDADES DEL MANIFEST
# ==========================================================

def find_column(fieldnames, candidates):

    if not fieldnames:
        return None

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


def get_value(row, column):

    if not column:
        return ""

    value = row.get(
        column,
        ""
    )

    if value is None:
        return ""

    return str(value).strip()


# ==========================================================
# LEER MANIFEST LSM
# ==========================================================

def read_lsm_manifest(manifest_path):

    rows = []

    excluded_status = 0
    excluded_empty = 0

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fields = (
            reader.fieldnames
            or []
        )

        print(
            "\nColumnas detectadas en manifest:"
        )

        for field in fields:
            print(
                f"  - {field}"
            )

        # ==================================================
        # MAPEO DE COLUMNAS
        # ==================================================

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

        group_col = find_column(
            fields,
            [
                "group",
                "group_id",
                "session",
                "session_group",
            ]
        )

        pattern_col = find_column(
            fields,
            [
                "filename_pattern",
                "pattern",
            ]
        )

        quality_col = find_column(
            fields,
            [
                "group_quality",
                "quality",
            ]
        )

        status_col = find_column(
            fields,
            [
                "status",
            ]
        )

        use_classifier_col = find_column(
            fields,
            [
                "use_letter_classifier",
            ]
        )

        sample_type_col = find_column(
            fields,
            [
                "sample_type",
            ]
        )

        source_language_col = find_column(
            fields,
            [
                "source_language",
            ]
        )

        provisional_source_col = find_column(
            fields,
            [
                "provisional_source",
            ]
        )

        split_original_col = find_column(
            fields,
            [
                "split_original",
            ]
        )

        split_final_col = find_column(
            fields,
            [
                "split_final",
            ]
        )

        reject_reason_col = find_column(
            fields,
            [
                "reject_reason",
            ]
        )

        source_archive_col = find_column(
            fields,
            [
                "source_archive",
            ]
        )

        filename_label_hint_col = find_column(
            fields,
            [
                "filename_label_hint",
            ]
        )

        filename_folder_mismatch_col = find_column(
            fields,
            [
                "filename_folder_mismatch",
            ]
        )

        session_hint_col = find_column(
            fields,
            [
                "session_hint",
            ]
        )

        copy_marker_col = find_column(
            fields,
            [
                "copy_marker",
            ]
        )

        extension_col = find_column(
            fields,
            [
                "extension",
            ]
        )

        size_bytes_col = find_column(
            fields,
            [
                "size_bytes",
            ]
        )

        notes_col = find_column(
            fields,
            [
                "notes",
            ]
        )

        # ==================================================
        # VALIDACIONES
        # ==================================================

        if not path_col:

            raise ValueError(
                "No se encontró la columna "
                "que contiene el path."
            )

        if not class_col:

            raise ValueError(
                "No se encontró la columna "
                "de clase."
            )

        if not status_col:

            raise ValueError(
                "No se encontró la columna "
                "'status'. No se puede distinguir "
                "entre muestras accepted y rejected."
            )

        # ==================================================
        # MOSTRAR MAPEO
        # ==================================================

        print(
            "\nMapeo usado:"
        )

        print(
            f"  path                  -> "
            f"{path_col}"
        )

        print(
            f"  class                 -> "
            f"{class_col}"
        )

        print(
            f"  group                 -> "
            f"{group_col or 'NO DISPONIBLE'}"
        )

        print(
            f"  pattern               -> "
            f"{pattern_col or 'NO DISPONIBLE'}"
        )

        print(
            f"  group_quality         -> "
            f"{quality_col or 'NO DISPONIBLE'}"
        )

        print(
            f"  status                -> "
            f"{status_col}"
        )

        print(
            f"  use_letter_classifier -> "
            f"{use_classifier_col or 'NO DISPONIBLE'}"
        )

        print(
            f"  source_language       -> "
            f"{source_language_col or 'NO DISPONIBLE'}"
        )

        print(
            f"  provisional_source    -> "
            f"{provisional_source_col or 'NO DISPONIBLE'}"
        )

        # ==================================================
        # FILTRAR FILAS
        # ==================================================

        for row in reader:

            # ----------------------------------------------
            # FUENTE DE VERDAD:
            # únicamente status == accepted
            # ----------------------------------------------

            status_value = (
                get_value(
                    row,
                    status_col
                )
                .lower()
            )

            if status_value != "accepted":

                excluded_status += 1
                continue

            path = get_value(
                row,
                path_col
            )

            label = (
                get_value(
                    row,
                    class_col
                )
                .upper()
            )

            if not path or not label:

                excluded_empty += 1
                continue

            rows.append({

                "path":
                    path,

                "class":
                    label,

                "group":
                    get_value(
                        row,
                        group_col
                    ),

                "pattern":
                    get_value(
                        row,
                        pattern_col
                    ),

                "group_quality":
                    get_value(
                        row,
                        quality_col
                    ),

                "status":
                    status_value,

                "use_letter_classifier":
                    get_value(
                        row,
                        use_classifier_col
                    ),

                "sample_type":
                    get_value(
                        row,
                        sample_type_col
                    ),

                "source_language":
                    get_value(
                        row,
                        source_language_col
                    ),

                "provisional_source":
                    get_value(
                        row,
                        provisional_source_col
                    ),

                "split_original":
                    get_value(
                        row,
                        split_original_col
                    ),

                "split_final":
                    get_value(
                        row,
                        split_final_col
                    ),

                "reject_reason":
                    get_value(
                        row,
                        reject_reason_col
                    ),

                "source_archive":
                    get_value(
                        row,
                        source_archive_col
                    ),

                "filename_label_hint":
                    get_value(
                        row,
                        filename_label_hint_col
                    ),

                "filename_folder_mismatch":
                    get_value(
                        row,
                        filename_folder_mismatch_col
                    ),

                "session_hint":
                    get_value(
                        row,
                        session_hint_col
                    ),

                "copy_marker":
                    get_value(
                        row,
                        copy_marker_col
                    ),

                "extension":
                    get_value(
                        row,
                        extension_col
                    ),

                "size_bytes":
                    get_value(
                        row,
                        size_bytes_col
                    ),

                "notes":
                    get_value(
                        row,
                        notes_col
                    ),
            })

    # ======================================================
    # RESUMEN DEL FILTRO
    # ======================================================

    print(
        "\n=== FILTRO DEL MANIFEST ==="
    )

    print(
        f"Filas aceptadas "
        f"(status=accepted) : "
        f"{len(rows):,}"
    )

    print(
        f"Filas excluidas "
        f"(status!=accepted): "
        f"{excluded_status:,}"
    )

    print(
        f"Filas vacías/inválidas      : "
        f"{excluded_empty:,}"
    )

    return rows


# ==========================================================
# COLUMNAS DEL CSV DE SALIDA
# ==========================================================

def build_fieldnames():

    fields = [

        # --------------------------------------------------
        # IDENTIDAD / PROCEDENCIA
        # --------------------------------------------------

        "dataset",
        "source_dataset",
        "provisional",

        "class",
        "path",

        "group",
        "pattern",
        "group_quality",

        # --------------------------------------------------
        # METADATOS HEREDADOS DEL MANIFEST
        # --------------------------------------------------

        "status",
        "use_letter_classifier",
        "sample_type",

        "source_language",
        "provisional_source",

        "split_original",
        "split_final",

        "reject_reason",

        "source_archive",

        "filename_label_hint",
        "filename_folder_mismatch",

        "session_hint",
        "copy_marker",

        "extension",
        "size_bytes",

        "notes",

        # --------------------------------------------------
        # RESULTADO DE MEDIAPIPE
        # --------------------------------------------------

        "landmarks_ok",
        "detection_variant",

        # --------------------------------------------------
        # DIMENSIONES
        # --------------------------------------------------

        "original_width",
        "original_height",

        "processed_width",
        "processed_height",

        # --------------------------------------------------
        # MANO
        # --------------------------------------------------

        "handedness",
        "handedness_score",

        # --------------------------------------------------
        # NORMALIZACIÓN
        # --------------------------------------------------

        "image_scale_0_9",
        "world_scale_0_9",
    ]

    # Agrega:
    #
    # img_raw_x0...z20
    # img_norm_x0...z20
    # world_raw_x0...z20
    # world_norm_x0...z20

    fields += (
        landmark_fieldnames()
    )

    return fields


# ==========================================================
# PERMITIR REANUDACIÓN
# ==========================================================

def existing_paths(output_path):

    path = Path(
        output_path
    )

    if not path.exists():
        return set()

    if path.stat().st_size == 0:
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

            image_path = (
                row.get(
                    "path",
                    ""
                )
                .strip()
            )

            if image_path:

                found.add(
                    image_path
                )

    return found


# ==========================================================
# PROCESAMIENTO PRINCIPAL
# ==========================================================

def main(args):

    # ======================================================
    # LEER MANIFEST
    # ======================================================

    manifest_rows = (
        read_lsm_manifest(
            args.manifest
        )
    )

    print(
        "\n=== EXTRACCIÓN LSM ==="
    )

    print(
        f"Muestras manifest    : "
        f"{len(manifest_rows):,}"
    )

    # ======================================================
    # VER QUÉ YA ESTÁ PROCESADO
    # ======================================================

    already_done = (
        existing_paths(
            args.output
        )
    )

    print(
        f"Ya procesadas        : "
        f"{len(already_done):,}"
    )

    pending = [
        row
        for row in manifest_rows
        if row["path"]
        not in already_done
    ]

    if args.max_new is not None:

        pending = (
            pending[
                :args.max_new
            ]
        )

    print(
        f"A procesar ahora     : "
        f"{len(pending):,}"
    )

    if not pending:

        print(
            "No hay muestras pendientes."
        )

        return

    # ======================================================
    # PREPARAR SALIDA
    # ======================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        output_path.exists()
        and
        output_path.stat().st_size > 0
    )

    fieldnames = (
        build_fieldnames()
    )

    # ======================================================
    # CONTADORES
    # ======================================================

    method_counter = Counter()

    class_valid = Counter()
    class_failed = Counter()

    group_valid = Counter()
    group_failed = Counter()

    valid = 0
    failed = 0

    start_time = time.time()

    # ======================================================
    # ABRIR ZIP
    # ======================================================

    with zipfile.ZipFile(
        args.zip,
        "r"
    ) as z:

        # Para validar rutas sin buscar
        # secuencialmente dentro del ZIP.
        zip_names = set(
            z.namelist()
        )

        # ==================================================
        # MEDIAPIPE
        # ==================================================

        with create_hand_landmarker(
            args.model
        ) as landmarker:

            # ==============================================
            # CSV DE SALIDA
            # ==============================================

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

                # ==========================================
                # RECORRER IMÁGENES
                # ==========================================

                for idx, meta in enumerate(
                    pending,
                    start=1
                ):

                    # ======================================
                    # METADATOS BASE
                    # ======================================

                    row = {

                        "dataset":
                            "LSM",

                        "source_dataset":
                            "LSM",

                        "provisional":
                            0,

                        "class":
                            meta["class"],

                        "path":
                            meta["path"],

                        "group":
                            meta["group"],

                        "pattern":
                            meta["pattern"],

                        "group_quality":
                            meta[
                                "group_quality"
                            ],

                        "status":
                            meta["status"],

                        "use_letter_classifier":
                            meta[
                                "use_letter_classifier"
                            ],

                        "sample_type":
                            meta[
                                "sample_type"
                            ],

                        "source_language":
                            meta[
                                "source_language"
                            ],

                        "provisional_source":
                            meta[
                                "provisional_source"
                            ],

                        "split_original":
                            meta[
                                "split_original"
                            ],

                        "split_final":
                            meta[
                                "split_final"
                            ],

                        "reject_reason":
                            meta[
                                "reject_reason"
                            ],

                        "source_archive":
                            meta[
                                "source_archive"
                            ],

                        "filename_label_hint":
                            meta[
                                "filename_label_hint"
                            ],

                        "filename_folder_mismatch":
                            meta[
                                "filename_folder_mismatch"
                            ],

                        "session_hint":
                            meta[
                                "session_hint"
                            ],

                        "copy_marker":
                            meta[
                                "copy_marker"
                            ],

                        "extension":
                            meta[
                                "extension"
                            ],

                        "size_bytes":
                            meta[
                                "size_bytes"
                            ],

                        "notes":
                            meta[
                                "notes"
                            ],
                    }

                    try:

                        # ==================================
                        # VALIDAR PATH
                        # ==================================

                        if (
                            meta["path"]
                            not in zip_names
                        ):

                            raise FileNotFoundError(
                                "El path del manifest "
                                "no existe dentro del ZIP."
                            )

                        # ==================================
                        # LEER IMAGEN
                        # ==================================

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
                                "devolvió None."
                            )

                        original_height, original_width = (
                            original.shape[:2]
                        )

                        row[
                            "original_width"
                        ] = (
                            original_width
                        )

                        row[
                            "original_height"
                        ] = (
                            original_height
                        )

                        # ==================================
                        # CASCADE MEDIAPIPE
                        # ==================================

                        detection = (
                            detect_with_cascade(
                                landmarker,
                                original
                            )
                        )

                        # ==================================
                        # NO SE ENCONTRÓ LA MANO
                        # ==================================

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

                            group_failed[
                                (
                                    meta["class"],
                                    meta["group"]
                                )
                            ] += 1

                        # ==================================
                        # MANO DETECTADA
                        # ==================================

                        else:

                            result = (
                                detection[
                                    "result"
                                ]
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

                            (
                                processed_height,
                                processed_width
                            ) = (
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
                            ] = (
                                processed_width
                            )

                            row[
                                "processed_height"
                            ] = (
                                processed_height
                            )

                            # ==============================
                            # HANDEDNESS
                            # ==============================

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

                            # ==============================
                            # IMAGE LANDMARKS
                            # ==============================

                            (
                                img_raw,
                                img_norm,
                                img_scale
                            ) = (
                                image_landmarks_to_arrays(
                                    result
                                    .hand_landmarks[0],

                                    processed_width,
                                    processed_height
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

                            # ==============================
                            # WORLD LANDMARKS
                            # ==============================

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
                                        .hand_world_landmarks[
                                            0
                                        ]
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

                            # ==============================
                            # CONTADORES
                            # ==============================

                            valid += 1

                            class_valid[
                                meta["class"]
                            ] += 1

                            group_valid[
                                (
                                    meta["class"],
                                    meta["group"]
                                )
                            ] += 1

                            method_counter[
                                variant
                            ] += 1

                    # ======================================
                    # ERROR INESPERADO
                    # ======================================

                    except Exception as e:

                        row[
                            "landmarks_ok"
                        ] = 0

                        row[
                            "detection_variant"
                        ] = (
                            "ERROR:"
                            f"{type(e).__name__}"
                        )

                        failed += 1

                        class_failed[
                            meta["class"]
                        ] += 1

                        group_failed[
                            (
                                meta["class"],
                                meta["group"]
                            )
                        ] += 1

                        print(
                            f"\n[ERROR] "
                            f"{meta['path']}"
                        )

                        print(
                            f"        {e}"
                        )

                    # ======================================
                    # GUARDAR FILA
                    # ======================================

                    writer.writerow(
                        row
                    )

                    # Importante:
                    # persistir progreso en disco.
                    out.flush()

                    # ======================================
                    # MOSTRAR PROGRESO
                    # ======================================

                    if (
                        idx % 100 == 0
                        or
                        idx == len(pending)
                    ):

                        elapsed = (
                            time.time()
                            -
                            start_time
                        )

                        rate = (
                            idx / elapsed
                            if elapsed > 0
                            else 0
                        )

                        remaining = (
                            len(pending)
                            -
                            idx
                        )

                        eta_seconds = (
                            remaining / rate
                            if rate > 0
                            else 0
                        )

                        eta_minutes = (
                            eta_seconds
                            / 60
                        )

                        extraction_rate = (
                            valid / idx * 100
                            if idx > 0
                            else 0
                        )

                        print(
                            f"Procesadas: "
                            f"{idx:,}/"
                            f"{len(pending):,}"
                            f" | OK: "
                            f"{valid:,}"
                            f" | FAIL: "
                            f"{failed:,}"
                            f" | tasa: "
                            f"{extraction_rate:.2f}%"
                            f" | "
                            f"{rate:.2f} img/s"
                            f" | ETA: "
                            f"{eta_minutes:.1f} min"
                        )

    # ======================================================
    # RESUMEN FINAL
    # ======================================================

    processed_now = (
        valid
        +
        failed
    )

    print(
        "\n=== RESULTADO LSM ==="
    )

    print(
        f"Procesadas ahora     : "
        f"{processed_now:,}"
    )

    print(
        f"Landmarks válidos    : "
        f"{valid:,}"
    )

    print(
        f"Sin landmarks        : "
        f"{failed:,}"
    )

    if processed_now:

        print(
            f"Tasa extracción      : "
            f"{valid / processed_now * 100:.2f}%"
        )

    print(
        f"Filas totales salida : "
        f"{len(already_done) + processed_now:,}"
    )

    # ======================================================
    # MÉTODO UTILIZADO
    # ======================================================

    print(
        "\nMétodo seleccionado:"
    )

    for method in CASCADE_ORDER:

        print(
            f"  {method:<12} "
            f"{method_counter[method]:,}"
        )

    # ======================================================
    # RESULTADO POR CLASE
    # ======================================================

    print(
        "\nPor clase:"
    )

    classes = sorted(
        set(
            list(
                class_valid.keys()
            )
            +
            list(
                class_failed.keys()
            )
        )
    )

    for label in classes:

        ok = (
            class_valid[
                label
            ]
        )

        fail = (
            class_failed[
                label
            ]
        )

        total_class = (
            ok + fail
        )

        rate = (
            ok
            / total_class
            * 100
            if total_class
            else 0
        )

        print(
            f"  {label:<2} "
            f"OK={ok:>5} "
            f"FAIL={fail:>5} "
            f"{rate:6.2f}%"
        )

    print(
        f"\nSalida: "
        f"{output_path}"
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extrae y normaliza landmarks "
            "del dataset LSM aceptado."
        )
    )

    parser.add_argument(
        "--zip",
        required=True,
        help=(
            "Ruta a LSM_original.zip"
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Ruta a lsm_manifest_clean.csv"
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Ruta a hand_landmarker.task"
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Ruta al CSV de landmarks de salida"
        )
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help=(
            "Procesar como máximo N muestras "
            "nuevas. Útil para smoke tests."
        )
    )

    main(
        parser.parse_args()
    )
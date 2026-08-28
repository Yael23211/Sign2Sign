import argparse
import csv
from collections import Counter
from pathlib import Path


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

PROVISIONAL_CLASSES = {
    "E",
    "L",
    "Z",
}


# ==========================================================
# UTILIDADES
# ==========================================================

def parse_landmarks_ok(value):

    if value is None:
        return False

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


def read_header(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.reader(f)

        try:
            return next(reader)

        except StopIteration:

            raise ValueError(
                f"El archivo está vacío:\n"
                f"{path}"
            )


def build_unified_header(
    lsm_path,
    asl_path
):

    lsm_header = (
        read_header(lsm_path)
    )

    asl_header = (
        read_header(asl_path)
    )

    # Empezamos con la estructura LSM,
    # porque será nuestro dataset destino.
    header = list(
        lsm_header
    )

    # Si ASL tiene alguna columna adicional,
    # también la preservamos.
    for column in asl_header:

        if column not in header:
            header.append(column)

    return header


def normalized_row(
    source_row,
    fieldnames
):

    return {
        field:
            source_row.get(
                field,
                ""
            )
        for field in fieldnames
    }


# ==========================================================
# COPIAR LSM ORIGINAL VÁLIDO
# ==========================================================

def copy_valid_lsm(
    input_path,
    writer,
    fieldnames
):

    total = 0
    accepted = 0
    skipped = 0

    class_counter = Counter()

    with open(
        input_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for source_row in reader:

            total += 1

            # El CSV de landmarks conserva
            # tanto detecciones exitosas como fallidas.
            # Para el dataset candidato de ML
            # únicamente queremos vectores existentes.
            if not parse_landmarks_ok(
                source_row.get(
                    "landmarks_ok"
                )
            ):

                skipped += 1
                continue

            row = normalized_row(
                source_row,
                fieldnames
            )

            label = (
                str(
                    source_row.get(
                        "class",
                        ""
                    )
                )
                .strip()
                .upper()
            )

            row["dataset"] = "LSM"
            row["source_dataset"] = "LSM"
            row["provisional"] = "0"

            # El split de LSM todavía NO
            # se ha diseñado.
            #
            # Si está vacío, debe seguir vacío.
            # No asignamos train/test aquí.

            writer.writerow(row)

            accepted += 1

            class_counter[
                label
            ] += 1

            if accepted % 10000 == 0:

                print(
                    f"LSM válidas copiadas: "
                    f"{accepted:,}"
                )

    return {
        "total": total,
        "accepted": accepted,
        "skipped": skipped,
        "class_counter": class_counter,
    }


# ==========================================================
# INYECTAR E / L / Z DESDE ASL
# ==========================================================

def copy_provisional_asl(
    input_path,
    writer,
    fieldnames
):

    total = 0
    selected = 0

    ignored_class = 0
    ignored_failed = 0

    class_counter = Counter()
    split_counter = Counter()

    with open(
        input_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for source_row in reader:

            total += 1

            label = (
                str(
                    source_row.get(
                        "class",
                        ""
                    )
                )
                .strip()
                .upper()
            )

            # ----------------------------------------------
            # SOLO E, L, Z
            # ----------------------------------------------

            if (
                label
                not in PROVISIONAL_CLASSES
            ):

                ignored_class += 1
                continue

            # ----------------------------------------------
            # SOLO LANDMARKS VÁLIDOS
            # ----------------------------------------------

            if not parse_landmarks_ok(
                source_row.get(
                    "landmarks_ok"
                )
            ):

                ignored_failed += 1
                continue

            row = normalized_row(
                source_row,
                fieldnames
            )

            # ==============================================
            # PROCEDENCIA
            # ==============================================

            # Dataset para el que se utilizará
            row["dataset"] = "LSM"

            # Dataset del que realmente proviene
            row["source_dataset"] = "ASL"

            # Marca explícita
            row["provisional"] = "1"

            # Idioma/lengua de origen real
            if (
                "source_language"
                in fieldnames
            ):
                row[
                    "source_language"
                ] = "ASL"

            # Motivo de la incorporación
            if (
                "provisional_source"
                in fieldnames
            ):
                row[
                    "provisional_source"
                ] = (
                    "ASL_missing_LSM_class"
                )

            # ==============================================
            # SPLIT
            # ==============================================

            # Preservamos split_original de ASL
            # para trazabilidad.
            original_split = (
                str(
                    source_row.get(
                        "split_original",
                        ""
                    )
                )
                .strip()
                .lower()
            )

            # PERO para el modelo LSM estas muestras
            # jamás serán validation/test.
            if (
                "split_final"
                in fieldnames
            ):
                row[
                    "split_final"
                ] = "train_only"

            # ==============================================
            # GRUPO
            # ==============================================

            # No fingimos que pertenece a una sesión LSM.
            # Le asignamos un grupo explícitamente
            # provisional para que el split futuro
            # pueda identificarlo fácilmente.
            if (
                "group"
                in fieldnames
            ):
                row[
                    "group"
                ] = (
                    f"ASL_PROVISIONAL_{label}"
                )

            if (
                "group_quality"
                in fieldnames
            ):
                row[
                    "group_quality"
                ] = "provisional"

            if (
                "status"
                in fieldnames
            ):
                row[
                    "status"
                ] = "accepted"

            if (
                "sample_type"
                in fieldnames
            ):
                row[
                    "sample_type"
                ] = "letter"

            # No existe motivo de rechazo:
            # está siendo aceptada como muestra
            # provisional.
            if (
                "reject_reason"
                in fieldnames
            ):
                row[
                    "reject_reason"
                ] = ""

            writer.writerow(row)

            selected += 1

            class_counter[
                label
            ] += 1

            split_counter[
                original_split
                or "unknown"
            ] += 1

    return {
        "total": total,
        "selected": selected,
        "ignored_class":
            ignored_class,
        "ignored_failed":
            ignored_failed,
        "class_counter":
            class_counter,
        "split_counter":
            split_counter,
    }


# ==========================================================
# VALIDACIÓN DEL RESULTADO
# ==========================================================

def analyze_output(
    output_path
):

    total = 0

    source_counter = Counter()
    provisional_counter = Counter()
    class_counter = Counter()

    provisional_class_counter = Counter()

    invalid_rows = 0

    with open(
        output_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            total += 1

            label = (
                str(
                    row.get(
                        "class",
                        ""
                    )
                )
                .strip()
                .upper()
            )

            source = (
                str(
                    row.get(
                        "source_dataset",
                        ""
                    )
                )
                .strip()
                .upper()
            )

            provisional = (
                str(
                    row.get(
                        "provisional",
                        ""
                    )
                )
                .strip()
            )

            class_counter[
                label
            ] += 1

            source_counter[
                source
            ] += 1

            provisional_counter[
                provisional
            ] += 1

            if provisional == "1":

                provisional_class_counter[
                    label
                ] += 1

            # Todo el archivo de salida debería
            # tener landmarks válidos.
            if not parse_landmarks_ok(
                row.get(
                    "landmarks_ok"
                )
            ):

                invalid_rows += 1

    return {
        "total":
            total,

        "source_counter":
            source_counter,

        "provisional_counter":
            provisional_counter,

        "class_counter":
            class_counter,

        "provisional_class_counter":
            provisional_class_counter,

        "invalid_rows":
            invalid_rows,
    }


# ==========================================================
# MAIN
# ==========================================================

def main(args):

    lsm_path = Path(
        args.lsm
    )

    asl_path = Path(
        args.asl
    )

    output_path = Path(
        args.output
    )

    # ======================================================
    # VALIDAR ENTRADAS
    # ======================================================

    if not lsm_path.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{lsm_path}"
        )

    if not asl_path.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{asl_path}"
        )

    if output_path.exists():

        if not args.overwrite:

            raise FileExistsError(
                f"\nEl archivo de salida ya existe:\n"
                f"{output_path}\n\n"
                f"Usa --overwrite si realmente "
                f"quieres reemplazarlo."
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # HEADER UNIFICADO
    # ======================================================

    fieldnames = (
        build_unified_header(
            lsm_path,
            asl_path
        )
    )

    print(
        "\n=== CONSTRUCCIÓN LSM PROVISIONAL ==="
    )

    print(
        f"LSM:"
        f"\n  {lsm_path}"
    )

    print(
        f"\nASL:"
        f"\n  {asl_path}"
    )

    print(
        f"\nSalida:"
        f"\n  {output_path}"
    )

    print(
        f"\nColumnas unificadas: "
        f"{len(fieldnames)}"
    )

    # ======================================================
    # GENERAR DATASET
    # ======================================================

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as out:

        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        # --------------------------------------------------
        # LSM
        # --------------------------------------------------

        print(
            "\n--- Copiando LSM original válido ---"
        )

        lsm_stats = (
            copy_valid_lsm(
                lsm_path,
                writer,
                fieldnames
            )
        )

        # --------------------------------------------------
        # ASL E / L / Z
        # --------------------------------------------------

        print(
            "\n--- Añadiendo E/L/Z provisionales ---"
        )

        asl_stats = (
            copy_provisional_asl(
                asl_path,
                writer,
                fieldnames
            )
        )

    # ======================================================
    # RESUMEN DE CONSTRUCCIÓN
    # ======================================================

    print(
        "\n=== LSM ORIGINAL ==="
    )

    print(
        f"Filas landmarks       : "
        f"{lsm_stats['total']:,}"
    )

    print(
        f"Vectores válidos      : "
        f"{lsm_stats['accepted']:,}"
    )

    print(
        f"Sin landmarks omitidas: "
        f"{lsm_stats['skipped']:,}"
    )

    print(
        "\n=== ASL PROVISIONAL ==="
    )

    print(
        f"Filas ASL revisadas   : "
        f"{asl_stats['total']:,}"
    )

    print(
        f"E/L/Z válidas añadidas: "
        f"{asl_stats['selected']:,}"
    )

    print(
        f"E/L/Z fallidas omitidas: "
        f"{asl_stats['ignored_failed']:,}"
    )

    print(
        "\nPor clase provisional:"
    )

    for label in sorted(
        PROVISIONAL_CLASSES
    ):

        print(
            f"  {label}: "
            f"{asl_stats['class_counter'][label]:,}"
        )

    print(
        "\nSplit ORIGINAL de "
        "las muestras ASL añadidas:"
    )

    for split, count in sorted(
        asl_stats[
            "split_counter"
        ].items()
    ):

        print(
            f"  {split:<10} "
            f"{count:,}"
        )

    # ======================================================
    # VERIFICACIÓN DEL ARCHIVO FINAL
    # ======================================================

    print(
        "\n=== VERIFICANDO ARCHIVO FINAL ==="
    )

    final_stats = (
        analyze_output(
            output_path
        )
    )

    print(
        f"Filas totales          : "
        f"{final_stats['total']:,}"
    )

    print(
        f"Filas sin landmarks    : "
        f"{final_stats['invalid_rows']:,}"
    )

    print(
        "\nPor procedencia:"
    )

    for source, count in (
        final_stats[
            "source_counter"
        ].most_common()
    ):

        print(
            f"  {source:<8} "
            f"{count:,}"
        )

    print(
        "\nProvisionales:"
    )

    for value, count in (
        final_stats[
            "provisional_counter"
        ].items()
    ):

        label = (
            "sí"
            if value == "1"
            else "no"
        )

        print(
            f"  {label:<3} "
            f"{count:,}"
        )

    print(
        "\n=== DISTRIBUCIÓN FINAL POR CLASE ==="
    )

    for label in sorted(
        final_stats[
            "class_counter"
        ]
    ):

        total_class = (
            final_stats[
                "class_counter"
            ][label]
        )

        provisional_class = (
            final_stats[
                "provisional_class_counter"
            ][label]
        )

        original_class = (
            total_class
            -
            provisional_class
        )

        print(
            f"{label:<2} "
            f"total={total_class:>5,} "
            f"LSM={original_class:>5,} "
            f"provisional="
            f"{provisional_class:>5,}"
        )

    # ======================================================
    # SANITY CHECKS
    # ======================================================

    expected_total = (
        lsm_stats["accepted"]
        +
        asl_stats["selected"]
    )

    if (
        final_stats["total"]
        != expected_total
    ):

        raise RuntimeError(
            "El total del archivo final "
            "no coincide con las filas "
            "que deberían haberse escrito."
        )

    if (
        final_stats["invalid_rows"]
        != 0
    ):

        raise RuntimeError(
            "El dataset provisional "
            "contiene filas sin landmarks."
        )

    for label in PROVISIONAL_CLASSES:

        count = (
            final_stats[
                "provisional_class_counter"
            ][label]
        )

        if count == 0:

            raise RuntimeError(
                f"No se añadieron muestras "
                f"provisionales para {label}."
            )

    print(
        "\n✅ Dataset provisional "
        "construido correctamente."
    )

    print(
        f"\nArchivo:"
        f"\n{output_path}"
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Construye el dataset provisional "
            "de landmarks LSM añadiendo las "
            "clases faltantes E, L y Z "
            "desde ASL."
        )
    )

    parser.add_argument(
        "--lsm",
        required=True,
        help=(
            "Ruta a lsm_landmarks.csv"
        )
    )

    parser.add_argument(
        "--asl",
        required=True,
        help=(
            "Ruta a asl_landmarks.csv"
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Ruta del dataset provisional "
            "LSM resultante"
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Permite reemplazar el archivo "
            "de salida si ya existe."
        )
    )

    main(
        parser.parse_args()
    )
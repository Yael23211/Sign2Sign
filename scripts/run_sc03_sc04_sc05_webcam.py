from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.recognition import SignRecognizer

from src.text import (
    EmptyTextError,
    TextBuilder,
)

from src.text_processing import (
    CorrectionRequest,
    GeminiClient,
    Vocabulary,
)

from src.visual_mapping import (
    VisualMappingError,
    VisualResourceMapper,
)


# ============================================================
# MODELOS
# ============================================================

MODEL_CONFIG = {
    "ASL": {
        "model": (
            ROOT
            / "models"
            / "asl"
            / "A3_mlp_img_norm.keras"
        ),
        "scaler": (
            ROOT
            / "models"
            / "asl"
            / "scaler_img_norm.joblib"
        ),
    },

    "LSM": {
        "model": (
            ROOT
            / "models"
            / "lsm"
            / "V2_M2_mlp_world_norm.keras"
        ),
        "scaler": (
            ROOT
            / "models"
            / "lsm"
            / "scaler_world_norm.joblib"
        ),
    },
}


MEDIAPIPE_MODEL = (
    ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)


VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


TARGET_LANGUAGE = {
    "LSM": "ASL",
    "ASL": "LSM",
}


# ============================================================
# VENTANA
# ============================================================

WINDOW_NAME = (
    "Sign2Sign - SC-03 + SC-04 + SC-05"
)

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Sign2Sign integrated "
            "SC-03 + SC-04 + SC-05 webcam flow."
        )
    )

    parser.add_argument(
        "--language",
        choices=(
            "ASL",
            "LSM",
        ),
        help=(
            "Source sign language. "
            "If omitted, it will be requested."
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "OpenCV camera index. Default: 0."
        ),
    )

    return parser.parse_args()


# ============================================================
# SELECCIÓN DE IDIOMA
# ============================================================

def select_language(
    argument: str | None,
) -> str:

    if argument is not None:
        return argument.upper()

    while True:

        value = input(
            "Lengua de entrada [ASL/LSM]: "
        ).strip().upper()

        if value in MODEL_CONFIG:
            return value

        print(
            "Valor no válido. "
            "Escribe ASL o LSM."
        )


# ============================================================
# UTILIDADES
# ============================================================

def truncate_text(
    text: str,
    max_length: int = 48,
) -> str:

    if len(text) <= max_length:
        return text

    return (
        "..."
        + text[-(max_length - 3):]
    )


def build_resources_summary(
    mapping_result,
) -> str:

    if mapping_result is None:
        return "(ninguno)"

    names = [
        Path(resource.location).name
        for resource
        in mapping_result.resources
    ]

    if not names:
        return "(ninguno)"

    return " | ".join(
        names
    )


# ============================================================
# INTERFAZ DE CAPTURA
# ============================================================

def draw_capture_interface(
    frame,
    source_language: str,
    target_language: str,
    text: str,
    status: str,
    last_top_k,
):

    height, width = frame.shape[:2]

    # Panel superior
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 165),
        (0, 0, 0),
        -1,
    )

    # Panel inferior
    cv2.rectangle(
        frame,
        (0, height - 110),
        (width, height),
        (0, 0, 0),
        -1,
    )

    # Título
    cv2.putText(
        frame,
        (
            f"Sign2Sign - "
            f"{source_language} -> "
            f"{target_language}"
        ),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Texto
    cv2.putText(
        frame,
        "Texto:",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    visible_text = truncate_text(
        text
    )

    cv2.putText(
        frame,
        (
            visible_text
            if visible_text
            else "(vacio)"
        ),
        (100, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Top-3
    if last_top_k:

        top_text = " | ".join(
            (
                f"{label}: "
                f"{probability * 100:.1f}%"
            )
            for label, probability
            in last_top_k
        )

        cv2.putText(
            frame,
            (
                "Ultimo Top-3: "
                + top_text
            ),
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # Estado
    cv2.putText(
        frame,
        (
            "Estado: "
            + truncate_text(
                status,
                75,
            )
        ),
        (20, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Controles
    cv2.putText(
        frame,
        (
            "ENTER: capturar   "
            "SPACE: espacio   "
            "BACKSPACE: borrar"
        ),
        (20, height - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            "ESC: limpiar   "
            "T: enviar a SC-04/SC-05   "
            "Q: salir"
        ),
        (20, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# INTERFAZ DE RESULTADO
# ============================================================

def draw_result_interface(
    frame,
    call_result,
    mapping_result,
):

    height, width = frame.shape[:2]

    # Panel superior más alto,
    # ahora incluye resultado de SC-05.
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 325),
        (0, 0, 0),
        -1,
    )

    cv2.rectangle(
        frame,
        (0, height - 75),
        (width, height),
        (0, 0, 0),
        -1,
    )

    result = call_result.result

    cv2.putText(
        frame,
        "Sign2Sign - Resultado SC-03 -> SC-04 -> SC-05",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    resources_summary = (
        build_resources_summary(
            mapping_result
        )
    )

    lines = [
        (
            "Reconocido: "
            + truncate_text(
                result.recognized_text
            )
        ),
        (
            "Corregido: "
            + truncate_text(
                result.corrected_text
                if result.corrected_text
                else "(sin resolver)"
            )
        ),
        (
            "Traducido: "
            + truncate_text(
                result.translated_text
                if result.translated_text
                else "(sin resolver)"
            )
        ),
        (
            "Conceptos: "
            + (
                ", ".join(
                    result.concepts
                )
                if result.concepts
                else "(ninguno)"
            )
        ),
        (
            "Unresolved: "
            + (
                ", ".join(
                    result.unresolved
                )
                if result.unresolved
                else "(ninguno)"
            )
        ),
        (
            "Recursos: "
            + resources_summary
        ),
        (
            f"Latencia SC-04: "
            f"{call_result.latency_seconds:.3f} s"
        ),
    ]

    y = 68

    for line in lines:

        cv2.putText(
            frame,
            truncate_text(
                line,
                85,
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        y += 34

    cv2.putText(
        frame,
        (
            "ESC: nueva cadena   "
            "Q: salir"
        ),
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# IMPRESIÓN
# ============================================================

def print_integrated_result(
    call_result,
    mapping_result,
):

    print()
    print("=" * 72)
    print(
        "RESULTADO INTEGRADO "
        "SC-03 -> SC-04 -> SC-05"
    )
    print("=" * 72)

    print()
    print("SC-04 TranslationResult")
    print("-----------------------")

    print(
        json.dumps(
            call_result.result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    print(
        f"Modelo Gemini: "
        f"{call_result.model}"
    )

    print(
        f"Latencia SC-04: "
        f"{call_result.latency_seconds:.3f} s"
    )

    print()

    print("SC-05 ResourceMappingResult")
    print("---------------------------")

    if mapping_result is None:

        print(
            "No se generaron recursos visuales."
        )

    else:

        print(
            json.dumps(
                mapping_result.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )

        print()

        print(
            "Secuencia visual resuelta:"
        )

        for index, resource in enumerate(
            mapping_result.resources,
            start=1,
        ):

            print(
                f"  {index}. "
                f"{resource.concept_id} "
                f"-> "
                f"{resource.location}"
            )

    print()
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    source_language = select_language(
        args.language
    )

    target_language = TARGET_LANGUAGE[
        source_language
    ]

    config = MODEL_CONFIG[
        source_language
    ]

    # --------------------------------------------------------
    # INFORMACIÓN
    # --------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "Sign2Sign - "
        "Integracion SC-03 + SC-04 + SC-05"
    )

    print("=" * 72)

    print(
        f"Flujo: "
        f"{source_language}"
        f" -> "
        f"{target_language}"
    )

    print(
        f"Modelo reconocimiento: "
        f"{config['model'].name}"
    )

    print(
        "Modelo LLM: "
        "gemini-3.5-flash-lite"
    )

    print(
        "Mapeo visual: "
        "delivery=local"
    )

    # --------------------------------------------------------
    # VOCABULARIO
    # --------------------------------------------------------

    print()
    print(
        "Cargando vocabulario..."
    )

    vocabulary = Vocabulary(
        VOCAB_PATH
    )

    print(
        f"Conceptos disponibles: "
        f"{len(vocabulary)}"
    )

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    print(
        "Inicializando Gemini..."
    )

    gemini_client = GeminiClient(
        vocabulary
    )

    # --------------------------------------------------------
    # SC-05
    # --------------------------------------------------------

    print(
        "Inicializando VisualResourceMapper..."
    )

    resource_mapper = (
        VisualResourceMapper()
    )

    # --------------------------------------------------------
    # RECONOCEDOR
    # --------------------------------------------------------

    print(
        "Cargando reconocedor..."
    )

    recognizer = SignRecognizer(
        source_language=source_language,
        mediapipe_model_path=MEDIAPIPE_MODEL,
        classifier_model_path=config[
            "model"
        ],
        scaler_path=config[
            "scaler"
        ],
        top_k_size=3,
    )

    # --------------------------------------------------------
    # TEXT BUILDER
    # --------------------------------------------------------

    builder = TextBuilder(
        source_language=source_language,
        top_k_size=3,
    )

    # --------------------------------------------------------
    # CÁMARA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        args.camera
    )

    if not camera.isOpened():

        recognizer.close()

        raise RuntimeError(
            f"No se pudo abrir la camara "
            f"{args.camera}."
        )

    # --------------------------------------------------------
    # VENTANA
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    status = (
        "Listo. Presiona ENTER "
        "para capturar una sena."
    )

    last_top_k = ()

    result_mode = False

    last_call_result = None
    last_mapping_result = None

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

    print()
    print("Controles:")
    print("  ENTER      Capturar letra")
    print("  SPACE      Separar palabras")
    print("  BACKSPACE  Borrar")
    print("  ESC        Limpiar / nueva cadena")
    print(
        "  T          Finalizar y enviar "
        "a SC-04 + SC-05"
    )
    print("  Q          Salir")
    print()

    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            ok, frame = camera.read()

            if not ok:

                status = (
                    "No se pudo leer "
                    "la camara."
                )

                continue

            display = frame.copy()

            # =================================================
            # MODO RESULTADO
            # =================================================

            if (
                result_mode
                and last_call_result is not None
            ):

                draw_result_interface(
                    frame=display,
                    call_result=last_call_result,
                    mapping_result=(
                        last_mapping_result
                    ),
                )

            # =================================================
            # MODO CAPTURA
            # =================================================

            else:

                draw_capture_interface(
                    frame=display,
                    source_language=(
                        source_language
                    ),
                    target_language=(
                        target_language
                    ),
                    text=builder.current_text,
                    status=status,
                    last_top_k=last_top_k,
                )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKeyEx(1)

            if key == -1:
                continue

            # =================================================
            # Q - SALIR
            # =================================================

            if key in (
                ord("q"),
                ord("Q"),
            ):

                print(
                    "[INFO] Ejecucion finalizada."
                )

                break

            # =================================================
            # MODO RESULTADO
            # =================================================

            if result_mode:

                if key == 27:

                    builder.clear()

                    last_top_k = ()

                    last_call_result = None
                    last_mapping_result = None

                    result_mode = False

                    status = (
                        "Nueva cadena. "
                        "Presiona ENTER para capturar."
                    )

                    print()
                    print(
                        "[INFO] Nueva cadena iniciada."
                    )

                continue

            # =================================================
            # ENTER - RECONOCIMIENTO
            # =================================================

            if key in (
                10,
                13,
            ):

                status = (
                    "Procesando captura..."
                )

                result = recognizer.recognize(
                    frame
                )

                if result is None:

                    status = (
                        "No se detecto una mano. "
                        "No se agrego ninguna letra."
                    )

                    last_top_k = ()

                    print(
                        "[WARN] "
                        "No se detecto una mano."
                    )

                    continue

                builder.add_prediction(
                    letter=result.letter,
                    confidence=result.confidence,
                    top_k=result.top_k,
                )

                last_top_k = (
                    result.top_k
                )

                status = (
                    f"Letra agregada: "
                    f"{result.letter} "
                    f"({result.confidence * 100:.1f}%)"
                )

                print()

                print(
                    f"[CAPTURA] "
                    f"{result.letter}"
                )

                print(
                    "  Top-3: "
                    + " | ".join(
                        (
                            f"{label} "
                            f"{probability * 100:.2f}%"
                        )
                        for label, probability
                        in result.top_k
                    )
                )

                print(
                    f"  Texto: "
                    f"{builder.current_text}"
                )

                continue

            # =================================================
            # SPACE
            # =================================================

            if key == 32:

                inserted = (
                    builder.add_space()
                )

                status = (
                    "Separador agregado."
                    if inserted
                    else "Espacio ignorado."
                )

                continue

            # =================================================
            # BACKSPACE
            # =================================================

            if key in (
                8,
                127,
            ):

                removed = (
                    builder.backspace()
                )

                if removed is None:

                    status = (
                        "No hay elementos "
                        "para borrar."
                    )

                elif removed.kind == "space":

                    status = (
                        "Espacio eliminado."
                    )

                else:

                    status = (
                        f"Letra eliminada: "
                        f"{removed.value}"
                    )

                continue

            # =================================================
            # ESC - LIMPIAR
            # =================================================

            if key == 27:

                builder.clear()

                last_top_k = ()

                status = (
                    "Secuencia limpiada."
                )

                print(
                    "[INFO] Texto limpiado."
                )

                continue

            # =================================================
            # T
            # SC-03 -> SC-04 -> SC-05
            # =================================================

            if key in (
                ord("t"),
                ord("T"),
            ):

                # ---------------------------------------------
                # FINALIZAR SC-03
                # ---------------------------------------------

                try:

                    built_text = (
                        builder.finalize()
                    )

                except EmptyTextError:

                    status = (
                        "No se puede enviar "
                        "una secuencia vacia."
                    )

                    continue

                print()
                print("=" * 72)
                print(
                    "SC-03 FINALIZADO"
                )
                print("=" * 72)

                print(
                    f"Texto reconocido: "
                    f"{built_text.text}"
                )

                # ---------------------------------------------
                # REQUEST SC-04
                # ---------------------------------------------

                request = (
                    CorrectionRequest
                    .from_built_text(
                        built_text,
                        target_language=(
                            target_language
                        ),
                    )
                )

                status = (
                    "Procesando SC-04 "
                    "con Gemini..."
                )

                # Mostrar estado antes de llamada bloqueante.
                display_processing = (
                    frame.copy()
                )

                draw_capture_interface(
                    frame=display_processing,
                    source_language=(
                        source_language
                    ),
                    target_language=(
                        target_language
                    ),
                    text=builder.current_text,
                    status=status,
                    last_top_k=last_top_k,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_processing,
                )

                cv2.waitKey(1)

                # ---------------------------------------------
                # SC-04 - GEMINI REAL
                # ---------------------------------------------

                try:

                    last_call_result = (
                        gemini_client.process(
                            request
                        )
                    )

                except Exception as exc:

                    status = (
                        "Error en SC-04. "
                        "La cadena se conserva."
                    )

                    print()
                    print(
                        "[ERROR SC-04]"
                    )

                    print(
                        type(exc).__name__,
                        str(exc),
                    )

                    # No limpiar builder.
                    continue

                # ---------------------------------------------
                # SC-05
                # ---------------------------------------------

                translation_result = (
                    last_call_result.result
                )

                if not translation_result.concepts:

                    # Caso válido:
                    # SC-04 no pudo resolver ningún concepto.
                    last_mapping_result = None

                    print()
                    print(
                        "[SC-05] "
                        "No hay conceptos canónicos "
                        "para mapear."
                    )

                else:

                    status = (
                        "Resolviendo recursos "
                        "visuales con SC-05..."
                    )

                    try:

                        last_mapping_result = (
                            resource_mapper
                            .resolve_translation_result(
                                translation_result,
                                delivery="local",
                            )
                        )

                    except (
                        VisualMappingError,
                        ValueError,
                        TypeError,
                    ) as exc:

                        status = (
                            "Error en SC-05. "
                            "La cadena se conserva."
                        )

                        print()
                        print(
                            "[ERROR SC-05]"
                        )

                        print(
                            type(exc).__name__,
                            str(exc),
                        )

                        # SC-04 funcionó,
                        # pero no entramos todavía
                        # al modo de resultado porque
                        # el flujo SC-05 no terminó.
                        last_mapping_result = None

                        continue

                # ---------------------------------------------
                # RESULTADO INTEGRADO CORRECTO
                # ---------------------------------------------

                print_integrated_result(
                    last_call_result,
                    last_mapping_result,
                )

                status = (
                    "SC-03 -> SC-04 -> "
                    "SC-05 completado."
                )

                result_mode = True

    # ========================================================
    # LIBERACIÓN
    # ========================================================

    finally:

        camera.release()

        cv2.destroyAllWindows()

        recognizer.close()


if __name__ == "__main__":
    main()
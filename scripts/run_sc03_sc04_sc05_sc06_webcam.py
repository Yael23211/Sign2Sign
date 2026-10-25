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

from src.visual_output import (
    VisualOutputError,
    VisualSequencePlayer,
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
# VENTANA DE CAPTURA
# ============================================================

WINDOW_NAME = (
    "Sign2Sign - SC-03 -> SC-06"
)

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Sign2Sign complete local pipeline: "
            "SC-03 -> SC-04 -> SC-05 -> SC-06."
        )
    )

    parser.add_argument(
        "--language",
        choices=(
            "ASL",
            "LSM",
        ),
        help=(
            "Lengua de señas de entrada. "
            "Si se omite se solicitará por consola."
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "Índice OpenCV de la cámara. "
            "Default: 0."
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


def mapping_resource_names(
    mapping_result,
) -> str:

    if mapping_result is None:
        return "(ninguno)"

    resources = getattr(
        mapping_result,
        "resources",
        (),
    )

    if not resources:
        return "(ninguno)"

    return " | ".join(
        Path(
            resource.location
        ).name
        for resource in resources
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

    height, width = (
        frame.shape[:2]
    )

    # --------------------------------------------------------
    # PANELES
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 165),
        (0, 0, 0),
        -1,
    )

    cv2.rectangle(
        frame,
        (0, height - 110),
        (width, height),
        (0, 0, 0),
        -1,
    )

    # --------------------------------------------------------
    # TÍTULO
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEXTO ACTUAL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TOP-3
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

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
            "T: traducir + reproducir   "
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
    result,
    latency_seconds: float,
    mapping_result=None,
    playback_result=None,
    visual_error: str | None = None,
):

    height, width = (
        frame.shape[:2]
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 360),
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

    # --------------------------------------------------------
    # TÍTULO
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Sign2Sign - Resultado completo",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # DATOS SC-04 / SC-05 / SC-06
    # --------------------------------------------------------

    lines = [
        (
            "Reconocido: "
            + truncate_text(
                result.recognized_text,
                65,
            )
        ),
        (
            "Corregido: "
            + truncate_text(
                (
                    result.corrected_text
                    if result.corrected_text
                    else "(sin resolver)"
                ),
                65,
            )
        ),
        (
            "Traducido: "
            + truncate_text(
                (
                    result.translated_text
                    if result.translated_text
                    else "(sin resolver)"
                ),
                65,
            )
        ),
        (
            "Conceptos: "
            + truncate_text(
                (
                    ", ".join(
                        result.concepts
                    )
                    if result.concepts
                    else "(ninguno)"
                ),
                65,
            )
        ),
        (
            "Unresolved: "
            + truncate_text(
                (
                    ", ".join(
                        result.unresolved
                    )
                    if result.unresolved
                    else "(ninguno)"
                ),
                65,
            )
        ),
        (
            "Recursos SC-05: "
            + truncate_text(
                mapping_resource_names(
                    mapping_result
                ),
                65,
            )
        ),
        (
            f"Latencia SC-04: "
            f"{latency_seconds:.3f} s"
        ),
    ]

    if visual_error:

        lines.append(
            (
                "Salida visual: ERROR - "
                + truncate_text(
                    visual_error,
                    50,
                )
            )
        )

    elif playback_result is not None:

        if playback_result.completed:

            lines.append(
                (
                    "SC-06: secuencia completada "
                    f"({playback_result.played_count}/"
                    f"{playback_result.total_count})"
                )
            )

        elif playback_result.interrupted:

            lines.append(
                (
                    "SC-06: reproducción interrumpida "
                    f"({playback_result.played_count}/"
                    f"{playback_result.total_count})"
                )
            )

    elif not result.concepts:

        lines.append(
            "SC-06: sin conceptos para reproducir"
        )

    else:

        lines.append(
            "SC-06: no ejecutado"
        )

    y = 68

    for line in lines:

        cv2.putText(
            frame,
            truncate_text(
                line,
                80,
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        y += 34

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            "P: reproducir otra vez   "
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
# IMPRESIÓN SC-04
# ============================================================

def print_sc04_result(
    call_result,
):

    print()
    print("=" * 72)
    print(
        "RESULTADO SC-03 -> SC-04"
    )
    print("=" * 72)

    print(
        json.dumps(
            call_result.result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    print(
        f"Modelo: "
        f"{call_result.model}"
    )

    print(
        f"Latencia SC-04: "
        f"{call_result.latency_seconds:.3f} s"
    )

    print("=" * 72)


# ============================================================
# IMPRESIÓN SC-05
# ============================================================

def print_sc05_result(
    mapping_result,
):

    print()
    print("=" * 72)
    print(
        "RESULTADO SC-05 - MAPEO VISUAL"
    )
    print("=" * 72)

    print(
        f"Lengua destino: "
        f"{mapping_result.target_language}"
    )

    print(
        f"Delivery: "
        f"{mapping_result.delivery}"
    )

    print()

    for index, resource in enumerate(
        mapping_result.resources,
        start=1,
    ):

        print(
            f"  {index}. "
            f"{resource.concept_id}"
            f" -> "
            f"{resource.location}"
        )

    print("=" * 72)


# ============================================================
# IMPRESIÓN SC-06
# ============================================================

def print_sc06_result(
    playback_result,
):

    print()
    print("=" * 72)
    print(
        "RESULTADO SC-06 - SALIDA VISUAL"
    )
    print("=" * 72)

    print(
        f"Completado: "
        f"{playback_result.completed}"
    )

    print(
        f"Interrumpido: "
        f"{playback_result.interrupted}"
    )

    print(
        f"Videos reproducidos: "
        f"{playback_result.played_count}/"
        f"{playback_result.total_count}"
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    source_language = (
        select_language(
            args.language
        )
    )

    target_language = (
        TARGET_LANGUAGE[
            source_language
        ]
    )

    config = (
        MODEL_CONFIG[
            source_language
        ]
    )

    # --------------------------------------------------------
    # INFORMACIÓN INICIAL
    # --------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "Sign2Sign - Pipeline completo local"
    )

    print(
        "SC-03 -> SC-04 -> SC-05 -> SC-06"
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
        "Inicializando SC-05..."
    )

    resource_mapper = (
        VisualResourceMapper()
    )

    # --------------------------------------------------------
    # SC-06
    # --------------------------------------------------------

    print(
        "Inicializando SC-06..."
    )

    sequence_player = (
        VisualSequencePlayer()
    )

    # --------------------------------------------------------
    # SC-03
    # --------------------------------------------------------

    print(
        "Cargando reconocedor..."
    )

    recognizer = SignRecognizer(
        source_language=source_language,
        mediapipe_model_path=(
            MEDIAPIPE_MODEL
        ),
        classifier_model_path=(
            config[
                "model"
            ]
        ),
        scaler_path=(
            config[
                "scaler"
            ]
        ),
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
    last_playback_result = None

    last_visual_error = None

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

    print()
    print("Controles:")
    print(
        "  ENTER      Capturar letra"
    )
    print(
        "  SPACE      Separar palabras"
    )
    print(
        "  BACKSPACE  Borrar"
    )
    print(
        "  ESC        Limpiar / nueva cadena"
    )
    print(
        "  T          Traducir y reproducir"
    )
    print(
        "  P          Repetir salida visual"
    )
    print(
        "  Q          Salir"
    )
    print()

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    try:

        while True:

            ok, frame = (
                camera.read()
            )

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
                    result=(
                        last_call_result.result
                    ),
                    latency_seconds=(
                        last_call_result
                        .latency_seconds
                    ),
                    mapping_result=(
                        last_mapping_result
                    ),
                    playback_result=(
                        last_playback_result
                    ),
                    visual_error=(
                        last_visual_error
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
                    text=(
                        builder.current_text
                    ),
                    status=status,
                    last_top_k=(
                        last_top_k
                    ),
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

                # ---------------------------------------------
                # P - REPETIR SC-06
                # ---------------------------------------------

                if key in (
                    ord("p"),
                    ord("P"),
                ):

                    if (
                        last_mapping_result
                        is None
                    ):

                        print(
                            "[WARN] "
                            "No hay una secuencia visual "
                            "para reproducir."
                        )

                        continue

                    print()
                    print(
                        "[INFO] Repitiendo SC-06..."
                    )

                    last_visual_error = None

                    try:

                        last_playback_result = (
                            sequence_player.play(
                                last_mapping_result,
                                translated_text=(
                                    last_call_result
                                    .result
                                    .translated_text
                                ),
                                unresolved=(
                                    last_call_result
                                    .result
                                    .unresolved
                                ),
                            )
                        )

                        print_sc06_result(
                            last_playback_result
                        )

                    except VisualOutputError as exc:

                        last_playback_result = None

                        last_visual_error = (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                        print()
                        print(
                            "[ERROR SC-06]"
                        )

                        print(
                            last_visual_error
                        )

                    continue

                # ---------------------------------------------
                # ESC - NUEVA CADENA
                # ---------------------------------------------

                if key == 27:

                    builder.clear()

                    last_top_k = ()

                    last_call_result = None
                    last_mapping_result = None
                    last_playback_result = None

                    last_visual_error = None

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

                result = (
                    recognizer.recognize(
                        frame
                    )
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
                    letter=(
                        result.letter
                    ),
                    confidence=(
                        result.confidence
                    ),
                    top_k=(
                        result.top_k
                    ),
                )

                last_top_k = (
                    result.top_k
                )

                status = (
                    f"Letra agregada: "
                    f"{result.letter} "
                    f"("
                    f"{result.confidence * 100:.1f}%"
                    f")"
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
            # T - PIPELINE COMPLETO
            # =================================================

            if key in (
                ord("t"),
                ord("T"),
            ):

                # =============================================
                # SC-03
                # =============================================

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

                # =============================================
                # SC-04
                # =============================================

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

                display_processing = (
                    frame.copy()
                )

                draw_capture_interface(
                    frame=(
                        display_processing
                    ),
                    source_language=(
                        source_language
                    ),
                    target_language=(
                        target_language
                    ),
                    text=(
                        builder.current_text
                    ),
                    status=status,
                    last_top_k=(
                        last_top_k
                    ),
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_processing,
                )

                cv2.waitKey(1)

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

                    # La cadena permanece disponible
                    # para volver a presionar T.
                    continue

                print_sc04_result(
                    last_call_result
                )

                translation_result = (
                    last_call_result.result
                )

                # Limpiar resultados visuales anteriores.
                last_mapping_result = None
                last_playback_result = None
                last_visual_error = None

                # =============================================
                # SIN CONCEPTOS
                # =============================================

                if not translation_result.concepts:

                    print()
                    print(
                        "[WARN] "
                        "SC-04 no produjo conceptos "
                        "canónicos reproducibles."
                    )

                    result_mode = True

                    continue

                # =============================================
                # SC-05
                # =============================================

                status = (
                    "Resolviendo recursos SC-05..."
                )

                display_processing = (
                    frame.copy()
                )

                draw_capture_interface(
                    frame=(
                        display_processing
                    ),
                    source_language=(
                        source_language
                    ),
                    target_language=(
                        target_language
                    ),
                    text=(
                        builder.current_text
                    ),
                    status=status,
                    last_top_k=(
                        last_top_k
                    ),
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_processing,
                )

                cv2.waitKey(1)

                try:

                    last_mapping_result = (
                        resource_mapper
                        .resolve_translation_result(
                            translation_result,
                            delivery="local",
                        )
                    )

                except VisualMappingError as exc:

                    last_visual_error = (
                        f"SC-05: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    print()
                    print(
                        "[ERROR SC-05]"
                    )

                    print(
                        last_visual_error
                    )

                    result_mode = True

                    continue

                print_sc05_result(
                    last_mapping_result
                )

                # =============================================
                # SC-06
                # =============================================

                status = (
                    "Iniciando salida visual SC-06..."
                )

                display_processing = (
                    frame.copy()
                )

                draw_capture_interface(
                    frame=(
                        display_processing
                    ),
                    source_language=(
                        source_language
                    ),
                    target_language=(
                        target_language
                    ),
                    text=(
                        builder.current_text
                    ),
                    status=status,
                    last_top_k=(
                        last_top_k
                    ),
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_processing,
                )

                cv2.waitKey(1)

                try:

                    last_playback_result = (
                        sequence_player.play(
                            last_mapping_result,
                            translated_text=(
                                translation_result
                                .translated_text
                            ),
                            unresolved=(
                                translation_result
                                .unresolved
                            ),
                        )
                    )

                    print_sc06_result(
                        last_playback_result
                    )

                except VisualOutputError as exc:

                    last_playback_result = None

                    last_visual_error = (
                        f"SC-06: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    print()
                    print(
                        "[ERROR SC-06]"
                    )

                    print(
                        last_visual_error
                    )

                # =============================================
                # PIPELINE FINALIZADO
                # =============================================

                print()
                print("=" * 72)
                print(
                    "PIPELINE COMPLETO FINALIZADO"
                )
                print(
                    "SC-03 -> SC-04 -> SC-05 -> SC-06"
                )
                print("=" * 72)

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
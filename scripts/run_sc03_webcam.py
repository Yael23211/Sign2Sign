from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

# Permite importar desde la raíz del repositorio cuando se ejecuta:
#
# python scripts/run_sc03_webcam.py
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.recognition import SignRecognizer
from src.text import EmptyTextError, TextBuilder


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


# ============================================================
# CONFIGURACIÓN DE LA VENTANA
# ============================================================

WINDOW_NAME = "Sign2Sign - SC-03"

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sign2Sign SC-03 recognition and "
            "text construction."
        )
    )

    parser.add_argument(
        "--language",
        choices=("ASL", "LSM"),
        help=(
            "Source sign language. "
            "If omitted, it is requested interactively."
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0.",
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
            "Valor no valido. "
            "Escribe ASL o LSM."
        )


# ============================================================
# UTILIDADES DE INTERFAZ
# ============================================================

def truncate_text(
    text: str,
    max_length: int = 55,
) -> str:
    """
    Evita que textos demasiado largos salgan de la ventana.
    """

    if len(text) <= max_length:
        return text

    return "..." + text[-(max_length - 3):]


def draw_interface(
    frame,
    language: str,
    text: str,
    status: str,
    last_top_k,
):
    """
    Dibuja la interfaz temporal de SC-03.

    Esta ventana NO representa el frontend final del sistema.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # PANEL SUPERIOR
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 150),
        (0, 0, 0),
        -1,
    )

    # --------------------------------------------------------
    # PANEL INFERIOR
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, height - 105),
        (width, height),
        (0, 0, 0),
        -1,
    )

    # --------------------------------------------------------
    # TÍTULO
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Sign2Sign - SC-03 - {language}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
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
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    visible_text = truncate_text(
        text
    )

    cv2.putText(
        frame,
        visible_text if visible_text else "(vacio)",
        (95, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # TOP-3
    # --------------------------------------------------------

    if last_top_k:

        top_text = " | ".join(
            f"{label}: {probability * 100:.1f}%"
            for label, probability
            in last_top_k
        )

        cv2.putText(
            frame,
            f"Ultimo Top-3: {top_text}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Estado: {status}",
        (20, 133),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------

    controls_1 = (
        "ENTER: capturar letra   "
        "SPACE: espacio   "
        "BACKSPACE: borrar"
    )

    controls_2 = (
        "ESC: limpiar   "
        "T: finalizar texto   "
        "Q: salir"
    )

    cv2.putText(
        frame,
        controls_1,
        (20, height - 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        controls_2,
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    args = parse_args()

    language = select_language(
        args.language
    )

    config = MODEL_CONFIG[
        language
    ]

    print()
    print("=" * 72)
    print(
        f"Sign2Sign - SC-03 - {language}"
    )
    print("=" * 72)

    print(
        f"Modelo: {config['model']}"
    )

    print(
        f"Scaler: {config['scaler']}"
    )

    print(
        f"MediaPipe: {MEDIAPIPE_MODEL}"
    )

    print()
    print(
        "Cargando reconocedor..."
    )

    # --------------------------------------------------------
    # RECONOCEDOR
    # --------------------------------------------------------

    recognizer = SignRecognizer(
        source_language=language,
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
    # CONSTRUCTOR TEXTUAL
    # --------------------------------------------------------

    builder = TextBuilder(
        source_language=language,
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
    #
    # WINDOW_NORMAL permite redimensionar manualmente
    # la ventana.
    #
    # resizeWindow define solamente el tamaño inicial.
    #

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
    # ESTADO INICIAL
    # --------------------------------------------------------

    status = (
        "Listo. Presiona ENTER "
        "para capturar una sena."
    )

    last_top_k = ()

    finalized_result = None

    print()
    print("Controles:")
    print("  ENTER     Capturar letra")
    print("  SPACE     Separar palabras")
    print("  BACKSPACE Borrar ultimo elemento")
    print("  ESC       Limpiar texto")
    print("  T         Finalizar texto")
    print("  Q         Salir sin finalizar")
    print()

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    try:

        while True:

            ok, frame = camera.read()

            if not ok:

                status = (
                    "No se pudo leer "
                    "el frame de la camara."
                )

                continue

            # IMPORTANTE:
            # No se aplica horizontal flip.
            #
            # La orientación debe mantenerse igual
            # que durante entrenamiento/evaluación.

            display = frame.copy()

            draw_interface(
                frame=display,
                language=language,
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
            # ENTER
            # Capturar y reconocer una letra
            # =================================================

            if key in (10, 13):

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
                        "[WARN] No se detecto "
                        "una mano."
                    )

                    continue

                builder.add_prediction(
                    letter=result.letter,
                    confidence=result.confidence,
                    top_k=result.top_k,
                )

                last_top_k = result.top_k

                status = (
                    f"Letra agregada: "
                    f"{result.letter} "
                    f"({result.confidence * 100:.1f}%)"
                )

                print()
                print(
                    f"[CAPTURA] {result.letter}"
                )

                print(
                    f"  Confianza: "
                    f"{result.confidence * 100:.2f}%"
                )

                print(
                    f"  Variante: "
                    f"{result.detection_variant}"
                )

                print(
                    "  Top-3: "
                    + " | ".join(
                        f"{label} "
                        f"{probability * 100:.2f}%"
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
            # Separar palabras
            # =================================================

            if key == 32:

                inserted = builder.add_space()

                if inserted:

                    status = (
                        "Separador de palabra agregado."
                    )

                else:

                    status = (
                        "Espacio ignorado."
                    )

                continue

            # =================================================
            # BACKSPACE
            # Eliminar último elemento
            # =================================================

            if key in (8, 127):

                removed = builder.backspace()

                if removed is None:

                    status = (
                        "No hay elementos para borrar."
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
            # ESC
            # Limpiar secuencia
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
            # Finalizar secuencia
            # =================================================

            if key in (
                ord("t"),
                ord("T"),
            ):

                try:

                    finalized_result = (
                        builder.finalize()
                    )

                except EmptyTextError:

                    status = (
                        "No se puede finalizar "
                        "una secuencia vacia."
                    )

                    continue

                print()
                print("=" * 72)
                print(
                    "TEXTO RECONOCIDO FINAL"
                )
                print("=" * 72)

                print(
                    finalized_result.text
                )

                print()
                print(
                    "Salida estructurada:"
                )

                print(
                    json.dumps(
                        finalized_result.to_dict(),
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                print("=" * 72)

                status = (
                    "Texto finalizado."
                )

                break

            # =================================================
            # Q
            # Salir sin finalizar
            # =================================================

            if key in (
                ord("q"),
                ord("Q"),
            ):

                print(
                    "[INFO] Ejecucion cancelada "
                    "sin finalizar."
                )

                break

    # ========================================================
    # LIBERACIÓN DE RECURSOS
    # ========================================================

    finally:

        camera.release()

        cv2.destroyAllWindows()

        recognizer.close()

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    if finalized_result is not None:

        print()
        print(
            "SC-03 produjo correctamente "
            "una salida textual."
        )


if __name__ == "__main__":
    main()
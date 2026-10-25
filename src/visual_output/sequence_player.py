from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .contracts import (
    InvalidPlaybackPlanError,
    PlaybackItem,
    PlaybackPlan,
    PlaybackResult,
    UnsupportedDeliveryError,
    VideoFileMissingError,
    VideoOpenError,
    VisualOutputError,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


class VisualSequencePlayer:
    """
    SC-06: reproducción secuencial de recursos visuales.

    Responsabilidades:
    - recibir la salida ordenada de SC-05;
    - validar que los recursos puedan reproducirse;
    - reproducir los videos en el mismo orden;
    - respetar dinámicamente los FPS de cada video;
    - mostrar información básica de la traducción;
    - permitir pausar o interrumpir la secuencia.

    SC-06 NO modifica el orden semántico recibido.
    """

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        window_name: str = "Sign2Sign - SC-06",
        video_width: int = 1152,
        video_height: int = 648,
        gap_ms: int = 150,
    ) -> None:

        self.project_root = Path(
            project_root
        )

        self.window_name = window_name

        self.video_width = video_width
        self.video_height = video_height

        self.gap_ms = max(
            0,
            int(gap_ms),
        )

    # ========================================================
    # PLAN DE REPRODUCCIÓN
    # ========================================================

    def build_plan(
        self,
        mapping_result: Any,
    ) -> PlaybackPlan:
        """
        Convierte ResourceMappingResult de SC-05
        en un plan de reproducción para SC-06.
        """

        if not hasattr(
            mapping_result,
            "target_language",
        ):
            raise InvalidPlaybackPlanError(
                "mapping_result debe contener "
                "'target_language'."
            )

        if not hasattr(
            mapping_result,
            "resources",
        ):
            raise InvalidPlaybackPlanError(
                "mapping_result debe contener "
                "'resources'."
            )

        resources = tuple(
            mapping_result.resources
        )

        if not resources:
            raise InvalidPlaybackPlanError(
                "No hay recursos visuales "
                "para reproducir."
            )

        target_language = str(
            mapping_result.target_language
        ).strip().upper()

        items: list[
            PlaybackItem
        ] = []

        total = len(
            resources
        )

        for position, resource in enumerate(
            resources,
            start=1,
        ):

            if resource.delivery != "local":
                raise UnsupportedDeliveryError(
                    "SC-06 local sólo admite "
                    "delivery='local'. "
                    f"Recibido: {resource.delivery}"
                )

            if resource.type != "video":
                raise InvalidPlaybackPlanError(
                    "SC-06 sólo admite recursos "
                    "de tipo video. "
                    f"Recibido: {resource.type}"
                )

            if not resource.available:
                raise InvalidPlaybackPlanError(
                    "El recurso está marcado como "
                    "no disponible: "
                    f"{resource.concept_id}"
                )

            self._validate_location(
                resource.location
            )

            items.append(
                PlaybackItem(
                    concept_id=(
                        resource.concept_id
                    ),
                    language=(
                        resource.language
                    ),
                    location=(
                        resource.location
                    ),
                    position=position,
                    total=total,
                )
            )

        return PlaybackPlan(
            target_language=target_language,
            items=tuple(
                items
            ),
        )

    # ========================================================
    # REPRODUCCIÓN DE SECUENCIA
    # ========================================================

    def play(
        self,
        mapping_result: Any,
        *,
        translated_text: str = "",
        unresolved: Iterable[str] = (),
    ) -> PlaybackResult:
        """
        Reproduce todos los recursos del mapping
        respetando el orden recibido.

        Controles:
            SPACE -> pausar / continuar
            Q     -> detener secuencia
            ESC   -> detener secuencia
        """

        plan = self.build_plan(
            mapping_result
        )

        unresolved_tuple = tuple(
            str(value)
            for value in unresolved
        )

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        played_count = 0
        interrupted = False

        try:

            for item in plan.items:

                completed_item = (
                    self._play_item(
                        item=item,
                        target_language=(
                            plan.target_language
                        ),
                        translated_text=(
                            translated_text
                        ),
                        unresolved=(
                            unresolved_tuple
                        ),
                    )
                )

                if not completed_item:
                    interrupted = True
                    break

                played_count += 1

                if (
                    self.gap_ms > 0
                    and item.position < item.total
                ):

                    continue_sequence = (
                        self._show_gap(
                            item=item,
                            target_language=(
                                plan.target_language
                            ),
                            translated_text=(
                                translated_text
                            ),
                        )
                    )

                    if not continue_sequence:
                        interrupted = True
                        break

        finally:

            try:
                cv2.destroyWindow(
                    self.window_name
                )
            except cv2.error:
                pass

        return PlaybackResult(
            completed=(
                not interrupted
                and played_count == plan.total
            ),
            interrupted=interrupted,
            played_count=played_count,
            total_count=plan.total,
        )

    # ========================================================
    # REPRODUCCIÓN DE VIDEO INDIVIDUAL
    # ========================================================

    def _play_item(
        self,
        item: PlaybackItem,
        target_language: str,
        translated_text: str,
        unresolved: tuple[str, ...],
    ) -> bool:

        physical_path = (
            self.project_root
            / item.location
        ).resolve()

        capture = cv2.VideoCapture(
            str(
                physical_path
            )
        )

        if not capture.isOpened():

            capture.release()

            raise VideoOpenError(
                "No se pudo abrir el video:\n"
                f"{item.location}"
            )

        # ====================================================
        # INFORMACIÓN DEL VIDEO
        # ====================================================

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        # Protección por si OpenCV no puede obtener
        # correctamente los FPS.
        if (
            fps is None
            or fps <= 1
            or fps > 240
        ):
            print(
                "[SC-06] FPS inválidos o desconocidos. "
                "Se utilizarán 30 FPS."
            )

            fps = 30.0

        # Duración ideal de un frame.
        frame_duration = (
            1.0 / fps
        )

        if (
            frame_count is not None
            and frame_count > 0
        ):
            estimated_duration = (
                frame_count / fps
            )
        else:
            estimated_duration = 0.0

        print()
        print(
            f"[SC-06] Reproduciendo: "
            f"{item.concept_id}"
        )

        print(
            f"  Archivo: "
            f"{item.location}"
        )

        print(
            f"  FPS detectados: "
            f"{fps:.3f}"
        )

        if frame_count > 0:

            print(
                f"  Frames: "
                f"{int(frame_count)}"
            )

        if estimated_duration > 0:

            print(
                f"  Duración estimada: "
                f"{estimated_duration:.3f} s"
            )

        # ====================================================
        # ESTADO DE REPRODUCCIÓN
        # ====================================================

        paused = False

        current_frame = None

        frame_index = 0

        # Reloj real de inicio de reproducción.
        playback_start = (
            time.perf_counter()
        )

        pause_started = None

        try:

            while True:

                # =============================================
                # LEER FRAME
                # =============================================

                if not paused:

                    ok, frame = (
                        capture.read()
                    )

                    if not ok:
                        break

                    current_frame = frame

                    frame_index += 1

                if current_frame is None:
                    break

                # =============================================
                # CONSTRUIR INTERFAZ
                # =============================================

                canvas = (
                    self._compose_frame(
                        frame=current_frame,
                        item=item,
                        target_language=(
                            target_language
                        ),
                        translated_text=(
                            translated_text
                        ),
                        unresolved=(
                            unresolved
                        ),
                        paused=paused,
                    )
                )

                cv2.imshow(
                    self.window_name,
                    canvas,
                )

                # =============================================
                # PAUSA
                # =============================================

                if paused:

                    key = (
                        cv2.waitKey(30)
                        & 0xFF
                    )

                    if key in (
                        ord("q"),
                        ord("Q"),
                        27,
                    ):
                        return False

                    if key == 32:

                        paused = False

                        if pause_started is not None:

                            pause_duration = (
                                time.perf_counter()
                                - pause_started
                            )

                            # Desplazamos el reloj inicial
                            # para que la pausa no altere
                            # la sincronización.
                            playback_start += (
                                pause_duration
                            )

                        pause_started = None

                    continue

                # =============================================
                # SINCRONIZACIÓN DINÁMICA
                # =============================================

                # Momento real en el que este frame
                # debería terminar.
                target_time = (
                    playback_start
                    + (
                        frame_index
                        * frame_duration
                    )
                )

                while True:

                    now = (
                        time.perf_counter()
                    )

                    remaining = (
                        target_time
                        - now
                    )

                    # Ya llegamos o nos retrasamos.
                    if remaining <= 0:
                        break

                    # Si queda suficiente tiempo,
                    # dormimos casi todo el intervalo.
                    #
                    # Dejamos los últimos ~2 ms libres
                    # para evitar sobreesperas.
                    if remaining > 0.004:

                        time.sleep(
                            max(
                                0.0,
                                remaining - 0.002,
                            )
                        )

                    else:
                        break

                # =============================================
                # EVENTOS DE TECLADO
                # =============================================

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    return False

                if key == 32:

                    paused = True

                    pause_started = (
                        time.perf_counter()
                    )

        finally:

            capture.release()

        return True

    # ========================================================
    # COMPOSICIÓN DE INTERFAZ
    # ========================================================

    def _compose_frame(
        self,
        frame,
        item: PlaybackItem,
        target_language: str,
        translated_text: str,
        unresolved: tuple[str, ...],
        paused: bool,
    ):

        resized = self._fit_frame(
            frame
        )

        header_height = 125
        footer_height = 55

        canvas = np.zeros(
            (
                self.video_height
                + header_height
                + footer_height,
                self.video_width,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            header_height:
            header_height + self.video_height,
            :
        ] = resized

        # ====================================================
        # TÍTULO
        # ====================================================

        cv2.putText(
            canvas,
            (
                f"Sign2Sign - "
                f"Salida {target_language}"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ====================================================
        # CONCEPTO ACTUAL
        # ====================================================

        cv2.putText(
            canvas,
            (
                f"Seña "
                f"{item.position}/{item.total}: "
                f"{item.concept_id}"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # TEXTO TRADUCIDO
        # ====================================================

        if translated_text:

            cv2.putText(
                canvas,
                (
                    "Traducción: "
                    + self._truncate(
                        translated_text,
                        75,
                    )
                ),
                (20, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        # ====================================================
        # FOOTER
        # ====================================================

        footer_y = (
            header_height
            + self.video_height
            + 35
        )

        status = (
            "PAUSA"
            if paused
            else "Reproduciendo"
        )

        cv2.putText(
            canvas,
            (
                f"{status}   "
                "SPACE: pausa   "
                "Q/ESC: detener"
            ),
            (20, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # ELEMENTOS SIN RESOLVER
        # ====================================================

        if unresolved:

            unresolved_text = (
                "Sin resolver: "
                + ", ".join(
                    unresolved
                )
            )

            cv2.putText(
                canvas,
                self._truncate(
                    unresolved_text,
                    55,
                ),
                (
                    self.video_width - 390,
                    footer_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return canvas

    # ========================================================
    # AJUSTE DEL VIDEO A LA VENTANA
    # ========================================================

    def _fit_frame(
        self,
        frame,
    ):

        height, width = (
            frame.shape[:2]
        )

        if (
            width <= 0
            or height <= 0
        ):
            raise VisualOutputError(
                "Frame de video inválido."
            )

        # Conservamos aspect ratio.
        scale = min(
            self.video_width / width,
            self.video_height / height,
        )

        new_width = max(
            1,
            int(
                width * scale
            ),
        )

        new_height = max(
            1,
            int(
                height * scale
            ),
        )

        resized = cv2.resize(
            frame,
            (
                new_width,
                new_height,
            ),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR
            ),
        )

        # Fondo negro para letterbox.
        canvas = np.zeros(
            (
                self.video_height,
                self.video_width,
                3,
            ),
            dtype=np.uint8,
        )

        x = (
            self.video_width
            - new_width
        ) // 2

        y = (
            self.video_height
            - new_height
        ) // 2

        canvas[
            y:y + new_height,
            x:x + new_width,
        ] = resized

        return canvas

    # ========================================================
    # TRANSICIÓN ENTRE VIDEOS
    # ========================================================

    def _show_gap(
        self,
        item: PlaybackItem,
        target_language: str,
        translated_text: str,
    ) -> bool:

        canvas = np.zeros(
            (
                self.video_height + 180,
                self.video_width,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            canvas,
            (
                f"Sign2Sign - "
                f"Salida {target_language}"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                f"Completado: "
                f"{item.concept_id}"
            ),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if translated_text:

            cv2.putText(
                canvas,
                (
                    "Traducción: "
                    + self._truncate(
                        translated_text,
                        75,
                    )
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(
            self.window_name,
            canvas,
        )

        key = (
            cv2.waitKey(
                self.gap_ms
            )
            & 0xFF
        )

        if key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            return False

        return True

    # ========================================================
    # VALIDACIÓN DE RUTA
    # ========================================================

    def _validate_location(
        self,
        location: str,
    ) -> None:

        relative_path = Path(
            location
        )

        if relative_path.is_absolute():

            raise InvalidPlaybackPlanError(
                "SC-06 espera rutas relativas: "
                f"{location}"
            )

        physical_path = (
            self.project_root
            / relative_path
        ).resolve()

        root = (
            self.project_root
            .resolve()
        )

        try:

            physical_path.relative_to(
                root
            )

        except ValueError as exc:

            raise InvalidPlaybackPlanError(
                "La ruta sale del proyecto: "
                f"{location}"
            ) from exc

        if not physical_path.is_file():

            raise VideoFileMissingError(
                "No existe el recurso visual:\n"
                f"{location}"
            )

    # ========================================================
    # UTILIDAD DE TEXTO
    # ========================================================

    @staticmethod
    def _truncate(
        text: str,
        max_length: int,
    ) -> str:

        if len(text) <= max_length:
            return text

        return (
            text[:max_length - 3]
            + "..."
        )
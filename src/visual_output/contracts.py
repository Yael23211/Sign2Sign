from __future__ import annotations

from dataclasses import dataclass


class VisualOutputError(Exception):
    """Excepción base de SC-06."""


class InvalidPlaybackPlanError(VisualOutputError):
    """La secuencia visual recibida no es válida."""


class UnsupportedDeliveryError(VisualOutputError):
    """SC-06 todavía no soporta la modalidad solicitada."""


class VideoFileMissingError(VisualOutputError):
    """El video local no existe físicamente."""


class VideoOpenError(VisualOutputError):
    """OpenCV no pudo abrir el video."""


@dataclass(frozen=True)
class PlaybackItem:
    concept_id: str
    language: str
    location: str
    position: int
    total: int


@dataclass(frozen=True)
class PlaybackPlan:
    target_language: str
    items: tuple[PlaybackItem, ...]

    @property
    def total(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class PlaybackResult:
    completed: bool
    interrupted: bool
    played_count: int
    total_count: int
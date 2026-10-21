from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


SUPPORTED_LANGUAGES = {"ASL", "LSM"}
SUPPORTED_DELIVERIES = {"local", "online"}


class VisualMappingError(Exception):
    """Excepción base para SC-05."""


class InvalidLanguageError(VisualMappingError):
    """Idioma destino no soportado."""


class InvalidDeliveryError(VisualMappingError):
    """Modalidad de entrega no soportada."""


class InvalidCatalogError(VisualMappingError):
    """Catálogo visual inválido."""


class ResourceNotFoundError(VisualMappingError):
    """No existe un recurso registrado para la clave solicitada."""


class ResourceUnavailableError(VisualMappingError):
    """El recurso existe en catálogo, pero está marcado como no disponible."""


class ResourceFileMissingError(VisualMappingError):
    """El recurso local está registrado, pero el archivo físico no existe."""


@dataclass(frozen=True)
class VisualResource:
    concept_id: str
    language: str
    delivery: str
    location: str
    type: str
    available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceMappingRequest:
    concepts: tuple[str, ...]
    target_language: str
    delivery: str = "local"

    @classmethod
    def build(
        cls,
        concepts: Iterable[str],
        target_language: str,
        delivery: str = "local",
    ) -> "ResourceMappingRequest":

        normalized_concepts = tuple(
            str(concept).strip().upper()
            for concept in concepts
            if str(concept).strip()
        )

        return cls(
            concepts=normalized_concepts,
            target_language=str(target_language).strip().upper(),
            delivery=str(delivery).strip().lower(),
        )


@dataclass(frozen=True)
class ResourceMappingResult:
    target_language: str
    delivery: str
    resources: tuple[VisualResource, ...]

    @property
    def locations(self) -> tuple[str, ...]:
        return tuple(
            resource.location
            for resource in self.resources
        )

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(
            resource.concept_id
            for resource in self.resources
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_language": self.target_language,
            "delivery": self.delivery,
            "resources": [
                resource.to_dict()
                for resource in self.resources
            ],
        }
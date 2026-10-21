from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    InvalidCatalogError,
    InvalidDeliveryError,
    InvalidLanguageError,
    ResourceFileMissingError,
    ResourceMappingRequest,
    ResourceMappingResult,
    ResourceNotFoundError,
    ResourceUnavailableError,
    SUPPORTED_DELIVERIES,
    SUPPORTED_LANGUAGES,
    VisualResource,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "visual_resources.json"
)


class VisualResourceMapper:
    """
    SC-05: mapeo de conceptos canónicos a recursos visuales.

    Este módulo NO reproduce videos.
    La reproducción corresponde a SC-06.
    """

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        project_root: str | Path = PROJECT_ROOT,
        verify_local_files: bool = True,
    ) -> None:

        self.catalog_path = Path(catalog_path)
        self.project_root = Path(project_root)
        self.verify_local_files = verify_local_files

        self._resources = self._load_catalog()
        self._index = self._build_index(
            self._resources
        )

    def reload(self) -> None:
        """
        Recarga visual_resources.json desde disco.
        """

        self._resources = self._load_catalog()

        self._index = self._build_index(
            self._resources
        )

    def resolve(
        self,
        concepts: Iterable[str],
        target_language: str,
        delivery: str = "local",
    ) -> ResourceMappingResult:
        """
        Resuelve una secuencia ordenada de concept_id.

        El orden original se conserva.
        Los conceptos repetidos también se conservan.
        """

        request = ResourceMappingRequest.build(
            concepts=concepts,
            target_language=target_language,
            delivery=delivery,
        )

        self._validate_request(
            request
        )

        resolved: list[
            VisualResource
        ] = []

        for concept_id in request.concepts:

            key = (
                concept_id,
                request.target_language,
                request.delivery,
            )

            resource = self._index.get(
                key
            )

            if resource is None:
                raise ResourceNotFoundError(
                    "No existe recurso visual registrado para "
                    f"{concept_id} / "
                    f"{request.target_language} / "
                    f"{request.delivery}."
                )

            if not resource.available:
                raise ResourceUnavailableError(
                    "El recurso visual existe, pero está marcado "
                    "como no disponible: "
                    f"{concept_id} / "
                    f"{request.target_language} / "
                    f"{request.delivery}."
                )

            if (
                self.verify_local_files
                and resource.delivery == "local"
            ):
                self._validate_local_file(
                    resource
                )

            resolved.append(
                resource
            )

        return ResourceMappingResult(
            target_language=request.target_language,
            delivery=request.delivery,
            resources=tuple(
                resolved
            ),
        )

    def resolve_translation_result(
        self,
        translation_result: Any,
        delivery: str = "local",
    ) -> ResourceMappingResult:
        """
        Integra directamente la salida formal de SC-04.

        TranslationResult debe exponer:
            - target_language
            - concepts
        """

        if not hasattr(
            translation_result,
            "concepts",
        ):
            raise TypeError(
                "translation_result debe contener "
                "un atributo 'concepts'."
            )

        if not hasattr(
            translation_result,
            "target_language",
        ):
            raise TypeError(
                "translation_result debe contener "
                "un atributo 'target_language'."
            )

        return self.resolve(
            concepts=translation_result.concepts,
            target_language=(
                translation_result.target_language
            ),
            delivery=delivery,
        )

    def has_resource(
        self,
        concept_id: str,
        language: str,
        delivery: str = "local",
        require_available: bool = True,
    ) -> bool:

        key = (
            str(
                concept_id
            ).strip().upper(),
            str(
                language
            ).strip().upper(),
            str(
                delivery
            ).strip().lower(),
        )

        resource = self._index.get(
            key
        )

        if resource is None:
            return False

        if (
            require_available
            and not resource.available
        ):
            return False

        return True

    def _validate_request(
        self,
        request: ResourceMappingRequest,
    ) -> None:

        if (
            request.target_language
            not in SUPPORTED_LANGUAGES
        ):
            raise InvalidLanguageError(
                "Idioma destino no soportado: "
                f"{request.target_language}. "
                f"Permitidos: "
                f"{sorted(SUPPORTED_LANGUAGES)}"
            )

        if (
            request.delivery
            not in SUPPORTED_DELIVERIES
        ):
            raise InvalidDeliveryError(
                "Modalidad no soportada: "
                f"{request.delivery}. "
                f"Permitidas: "
                f"{sorted(SUPPORTED_DELIVERIES)}"
            )

        if not request.concepts:
            raise ValueError(
                "Se requiere al menos un concept_id."
            )

    def _load_catalog(
        self,
    ) -> list[VisualResource]:

        if not self.catalog_path.is_file():
            raise InvalidCatalogError(
                "No se encontró el catálogo visual:\n"
                f"{self.catalog_path}"
            )

        try:
            with self.catalog_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(
                    file
                )

        except json.JSONDecodeError as exc:
            raise InvalidCatalogError(
                "visual_resources.json no contiene "
                "JSON válido."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise InvalidCatalogError(
                "La raíz del catálogo debe ser "
                "un objeto JSON."
            )

        raw_resources = data.get(
            "resources"
        )

        if not isinstance(
            raw_resources,
            list,
        ):
            raise InvalidCatalogError(
                "El catálogo debe contener "
                "un arreglo 'resources'."
            )

        resources: list[
            VisualResource
        ] = []

        for index, item in enumerate(
            raw_resources
        ):

            resources.append(
                self._parse_resource(
                    item,
                    index,
                )
            )

        return resources

    def _parse_resource(
        self,
        item: Any,
        index: int,
    ) -> VisualResource:

        if not isinstance(
            item,
            dict,
        ):
            raise InvalidCatalogError(
                f"El recurso {index} debe ser "
                "un objeto."
            )

        required = {
            "concept_id",
            "language",
            "delivery",
            "location",
            "type",
            "available",
        }

        missing = sorted(
            required - set(
                item
            )
        )

        if missing:
            raise InvalidCatalogError(
                f"El recurso {index} no contiene: "
                + ", ".join(
                    missing
                )
            )

        concept_id = str(
            item["concept_id"]
        ).strip().upper()

        language = str(
            item["language"]
        ).strip().upper()

        delivery = str(
            item["delivery"]
        ).strip().lower()

        location = str(
            item["location"]
        ).strip()

        resource_type = str(
            item["type"]
        ).strip().lower()

        available = item[
            "available"
        ]

        if not concept_id:
            raise InvalidCatalogError(
                f"El recurso {index} tiene "
                "concept_id vacío."
            )

        if (
            language
            not in SUPPORTED_LANGUAGES
        ):
            raise InvalidCatalogError(
                f"Idioma inválido en recurso "
                f"{index}: {language}"
            )

        if (
            delivery
            not in SUPPORTED_DELIVERIES
        ):
            raise InvalidCatalogError(
                f"Delivery inválido en recurso "
                f"{index}: {delivery}"
            )

        if not location:
            raise InvalidCatalogError(
                f"El recurso {index} tiene "
                "location vacío."
            )

        if not resource_type:
            raise InvalidCatalogError(
                f"El recurso {index} tiene "
                "type vacío."
            )

        if not isinstance(
            available,
            bool,
        ):
            raise InvalidCatalogError(
                f"El recurso {index} tiene "
                "available no booleano."
            )

        return VisualResource(
            concept_id=concept_id,
            language=language,
            delivery=delivery,
            location=location,
            type=resource_type,
            available=available,
        )

    def _build_index(
        self,
        resources: list[
            VisualResource
        ],
    ) -> dict[
        tuple[
            str,
            str,
            str,
        ],
        VisualResource,
    ]:

        index: dict[
            tuple[
                str,
                str,
                str,
            ],
            VisualResource,
        ] = {}

        for resource in resources:

            key = (
                resource.concept_id,
                resource.language,
                resource.delivery,
            )

            if key in index:
                raise InvalidCatalogError(
                    "Recurso duplicado detectado: "
                    f"{resource.concept_id} / "
                    f"{resource.language} / "
                    f"{resource.delivery}"
                )

            index[
                key
            ] = resource

        return index

    def _validate_local_file(
        self,
        resource: VisualResource,
    ) -> None:

        relative_path = Path(
            resource.location
        )

        if relative_path.is_absolute():
            raise InvalidCatalogError(
                "Las rutas locales deben ser "
                "relativas: "
                f"{resource.location}"
            )

        physical_path = (
            self.project_root
            / relative_path
        ).resolve()

        project_root_resolved = (
            self.project_root.resolve()
        )

        try:
            physical_path.relative_to(
                project_root_resolved
            )

        except ValueError as exc:
            raise InvalidCatalogError(
                "La ruta sale del directorio "
                "del proyecto: "
                f"{resource.location}"
            ) from exc

        if not physical_path.is_file():
            raise ResourceFileMissingError(
                "El recurso está registrado, "
                "pero el archivo no existe "
                "físicamente: "
                f"{resource.location}"
            )
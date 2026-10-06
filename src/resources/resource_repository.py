from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "visual_resources.json"
)


class ResourceCatalogError(ValueError):
    """
    Indicates that the visual resource catalog
    contains an invalid structure or entry.
    """


@dataclass(frozen=True)
class VisualResource:
    """
    Represents one visual resource registered
    in the Sign2Sign resource catalog.
    """

    concept_id: str
    language: str
    path: str
    type: str
    available: bool

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "language": self.language,
            "path": self.path,
            "type": self.type,
            "available": self.available,
        }


class ResourceRepository:
    """
    Access component for the visual resource repository.

    SC-07 is responsible for loading, validating and
    exposing the resources registered in the catalog.

    Mapping decisions are not performed here. Those
    decisions belong to SC-05.
    """

    SUPPORTED_LANGUAGES = {
        "ASL",
        "LSM",
    }

    SUPPORTED_TYPES = {
        "video",
    }

    REQUIRED_FIELDS = {
        "concept_id",
        "language",
        "path",
        "type",
        "available",
    }

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ):
        self.project_root = Path(
            project_root
            if project_root is not None
            else PROJECT_ROOT
        ).resolve()

        self.catalog_path = Path(
            catalog_path
            if catalog_path is not None
            else DEFAULT_CATALOG_PATH
        ).resolve()

        self._resources = self._load_catalog()

        self._index = {
            (
                resource.concept_id,
                resource.language,
            ): resource
            for resource in self._resources
        }

    def __len__(self) -> int:
        return len(self._resources)

    def _load_catalog(
        self,
    ) -> tuple[VisualResource, ...]:

        if not self.catalog_path.is_file():
            raise ResourceCatalogError(
                "Visual resource catalog was not found: "
                f"{self.catalog_path}"
            )

        try:
            with self.catalog_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ResourceCatalogError(
                "Visual resource catalog is not valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ResourceCatalogError(
                "Catalog root must be a JSON object."
            )

        resources_data = data.get(
            "resources"
        )

        if not isinstance(
            resources_data,
            list,
        ):
            raise ResourceCatalogError(
                "Catalog must contain a "
                "'resources' array."
            )

        resources = []
        seen_keys = set()

        for index, item in enumerate(
            resources_data
        ):
            resource = self._parse_resource(
                item,
                index,
            )

            key = (
                resource.concept_id,
                resource.language,
            )

            if key in seen_keys:
                raise ResourceCatalogError(
                    "Duplicated visual resource for "
                    f"{resource.concept_id} "
                    f"in {resource.language}."
                )

            seen_keys.add(key)
            resources.append(resource)

        return tuple(resources)

    def _parse_resource(
        self,
        item: object,
        index: int,
    ) -> VisualResource:

        if not isinstance(item, dict):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "must be a JSON object."
            )

        fields = set(item.keys())

        missing = (
            self.REQUIRED_FIELDS
            - fields
        )

        unexpected = (
            fields
            - self.REQUIRED_FIELDS
        )

        if missing:
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "is missing fields: "
                f"{sorted(missing)}"
            )

        if unexpected:
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "contains unexpected fields: "
                f"{sorted(unexpected)}"
            )

        concept_id = item[
            "concept_id"
        ]

        language = item[
            "language"
        ]

        path = item[
            "path"
        ]

        resource_type = item[
            "type"
        ]

        available = item[
            "available"
        ]

        if (
            not isinstance(
                concept_id,
                str,
            )
            or not concept_id.strip()
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "has an invalid concept_id."
            )

        concept_id = (
            concept_id
            .strip()
            .upper()
        )

        if (
            not isinstance(
                language,
                str,
            )
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "has an invalid language."
            )

        language = (
            language
            .strip()
            .upper()
        )

        if (
            language
            not in self.SUPPORTED_LANGUAGES
        ):
            raise ResourceCatalogError(
                f"Unsupported language: "
                f"{language}"
            )

        if (
            not isinstance(
                resource_type,
                str,
            )
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "has an invalid type."
            )

        resource_type = (
            resource_type
            .strip()
            .lower()
        )

        if (
            resource_type
            not in self.SUPPORTED_TYPES
        ):
            raise ResourceCatalogError(
                f"Unsupported resource type: "
                f"{resource_type}"
            )

        if (
            not isinstance(
                path,
                str,
            )
            or not path.strip()
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "has an invalid path."
            )

        path = path.strip()

        relative_path = Path(path)

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "must use a safe relative path."
            )

        expected_language_folder = (
            language.lower()
        )

        path_parts = tuple(
            part.lower()
            for part in relative_path.parts
        )

        if (
            len(path_parts) < 3
            or path_parts[0]
            != "resources"
            or path_parts[1]
            != expected_language_folder
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "must be stored under "
                f"resources/"
                f"{expected_language_folder}/"
            )

        if not isinstance(
            available,
            bool,
        ):
            raise ResourceCatalogError(
                f"Resource at index {index} "
                "'available' must be boolean."
            )

        return VisualResource(
            concept_id=concept_id,
            language=language,
            path=path,
            type=resource_type,
            available=available,
        )

    def get_resource(
        self,
        concept_id: str,
        language: str,
    ) -> VisualResource | None:
        """
        Returns the catalog entry associated with
        concept_id + language.

        File existence is not checked here.
        """

        key = (
            concept_id.strip().upper(),
            language.strip().upper(),
        )

        return self._index.get(key)

    def resolve_path(
        self,
        resource: VisualResource,
    ) -> Path:
        """
        Converts the relative catalog path into an
        absolute local filesystem path.
        """

        return (
            self.project_root
            / resource.path
        ).resolve()

    def is_available(
        self,
        concept_id: str,
        language: str,
    ) -> bool:
        """
        A resource is considered usable only when:

        1. it exists in the catalog;
        2. available is true;
        3. the physical file exists.
        """

        resource = self.get_resource(
            concept_id,
            language,
        )

        if resource is None:
            return False

        if not resource.available:
            return False

        return self.resolve_path(
            resource
        ).is_file()

    def get_available_resource(
        self,
        concept_id: str,
        language: str,
    ) -> VisualResource | None:
        """
        Returns a resource only if it is registered,
        enabled and physically present.
        """

        resource = self.get_resource(
            concept_id,
            language,
        )

        if resource is None:
            return None

        if not self.is_available(
            concept_id,
            language,
        ):
            return None

        return resource

    def list_resources(
        self,
        language: str | None = None,
    ) -> tuple[VisualResource, ...]:
        """
        Returns registered resources.

        Optionally filters by sign language.
        """

        if language is None:
            return self._resources

        normalized_language = (
            language
            .strip()
            .upper()
        )

        if (
            normalized_language
            not in self.SUPPORTED_LANGUAGES
        ):
            raise ValueError(
                "language must be "
                "'ASL' or 'LSM'."
            )

        return tuple(
            resource
            for resource in self._resources
            if resource.language
            == normalized_language
        )
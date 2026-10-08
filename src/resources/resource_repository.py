from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "resources"
    / "visual_resources.json"
)


class ResourceCatalogError(ValueError):
    """
    Raised when the visual resource catalog
    contains invalid data.
    """


@dataclass(frozen=True)
class VisualResource:
    concept_id: str
    language: str
    delivery: str
    location: str
    type: str
    available: bool

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "language": self.language,
            "delivery": self.delivery,
            "location": self.location,
            "type": self.type,
            "available": self.available,
        }


class ResourceRepository:

    SUPPORTED_LANGUAGES = {
        "ASL",
        "LSM",
    }

    SUPPORTED_DELIVERIES = {
        "local",
        "online",
    }

    SUPPORTED_TYPES = {
        "video",
    }

    REQUIRED_FIELDS = {
        "concept_id",
        "language",
        "delivery",
        "location",
        "type",
        "available",
    }

    def __init__(
        self,
        catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    ) -> None:

        self.catalog_path = Path(
            catalog_path
        )

        self._resources: list[
            VisualResource
        ] = []

        self._index: dict[
            tuple[str, str, str],
            VisualResource,
        ] = {}

        self._load()

    def _load(self) -> None:

        if not self.catalog_path.is_file():
            raise ResourceCatalogError(
                "Visual resource catalog "
                "was not found:\n"
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
            raise ResourceCatalogError(
                "Visual resource catalog "
                "is not valid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ResourceCatalogError(
                "Visual resource catalog "
                "root must be an object."
            )

        resources = data.get(
            "resources"
        )

        if not isinstance(
            resources,
            list,
        ):
            raise ResourceCatalogError(
                "Visual resource catalog "
                "must contain a "
                "'resources' array."
            )

        loaded_resources = []
        loaded_index = {}

        for index, item in enumerate(
            resources
        ):

            resource = (
                self._validate_resource(
                    item,
                    index,
                )
            )

            key = (
                resource.concept_id,
                resource.language,
                resource.delivery,
            )

            if key in loaded_index:
                raise ResourceCatalogError(
                    "Duplicate visual resource: "
                    f"{resource.concept_id} / "
                    f"{resource.language} / "
                    f"{resource.delivery}"
                )

            loaded_resources.append(
                resource
            )

            loaded_index[
                key
            ] = resource

        self._resources = (
            loaded_resources
        )

        self._index = (
            loaded_index
        )

    def _validate_resource(
        self,
        item: object,
        index: int,
    ) -> VisualResource:

        number = index + 1

        if not isinstance(
            item,
            dict,
        ):
            raise ResourceCatalogError(
                f"Resource #{number} "
                "must be an object."
            )

        fields = set(
            item.keys()
        )

        missing_fields = (
            self.REQUIRED_FIELDS
            - fields
        )

        unexpected_fields = (
            fields
            - self.REQUIRED_FIELDS
        )

        if missing_fields:
            raise ResourceCatalogError(
                f"Resource #{number} "
                "has missing fields: "
                + ", ".join(
                    sorted(
                        missing_fields
                    )
                )
            )

        if unexpected_fields:
            raise ResourceCatalogError(
                f"Resource #{number} "
                "has unexpected fields: "
                + ", ".join(
                    sorted(
                        unexpected_fields
                    )
                )
            )

        concept_id = self._validate_text(
            item["concept_id"],
            "concept_id",
            number,
        ).upper()

        language = self._validate_text(
            item["language"],
            "language",
            number,
        ).upper()

        if (
            language
            not in self.SUPPORTED_LANGUAGES
        ):
            raise ResourceCatalogError(
                f"Resource #{number} "
                "has unsupported language: "
                f"{language}"
            )

        delivery = self._validate_text(
            item["delivery"],
            "delivery",
            number,
        ).lower()

        if (
            delivery
            not in self.SUPPORTED_DELIVERIES
        ):
            raise ResourceCatalogError(
                f"Resource #{number} "
                "has unsupported delivery: "
                f"{delivery}"
            )

        resource_type = (
            self._validate_text(
                item["type"],
                "type",
                number,
            )
            .lower()
        )

        if (
            resource_type
            not in self.SUPPORTED_TYPES
        ):
            raise ResourceCatalogError(
                f"Resource #{number} "
                "has unsupported type: "
                f"{resource_type}"
            )

        location = (
            self._validate_text(
                item["location"],
                "location",
                number,
            )
        )

        if delivery == "local":
            location = (
                self._validate_local_location(
                    location,
                    language,
                    number,
                )
            )

        elif delivery == "online":
            location = (
                self._validate_online_location(
                    location,
                    number,
                )
            )

        available = item[
            "available"
        ]

        if not isinstance(
            available,
            bool,
        ):
            raise ResourceCatalogError(
                f"Resource #{number} "
                "'available' must be boolean."
            )

        return VisualResource(
            concept_id=concept_id,
            language=language,
            delivery=delivery,
            location=location,
            type=resource_type,
            available=available,
        )

    @staticmethod
    def _validate_text(
        value: object,
        field_name: str,
        resource_number: int,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                f"'{field_name}' must be text."
            )

        value = value.strip()

        if not value:
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                f"'{field_name}' "
                "cannot be empty."
            )

        return value

    @staticmethod
    def _validate_local_location(
        location: str,
        language: str,
        resource_number: int,
    ) -> str:

        normalized = (
            location
            .replace(
                "\\",
                "/",
            )
            .strip()
        )

        path = Path(
            normalized
        )

        if path.is_absolute():
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                "local location must "
                "be relative."
            )

        if ".." in path.parts:
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                "local location cannot "
                "contain '..'."
            )

        expected_prefix = (
            f"resources/"
            f"{language.lower()}/"
        )

        if not normalized.startswith(
            expected_prefix
        ):
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                "local location must be "
                f"inside {expected_prefix}"
            )

        return normalized

    @staticmethod
    def _validate_online_location(
        location: str,
        resource_number: int,
    ) -> str:

        parsed = urlparse(
            location
        )

        if (
            parsed.scheme
            not in {
                "http",
                "https",
            }
        ):
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                "online location must "
                "use http or https."
            )

        if not parsed.netloc:
            raise ResourceCatalogError(
                f"Resource #{resource_number} "
                "online location must "
                "contain a valid host."
            )

        return location

    def get_resource(
        self,
        concept_id: str,
        language: str,
        delivery: str,
    ) -> VisualResource | None:

        key = (
            concept_id.strip().upper(),
            language.strip().upper(),
            delivery.strip().lower(),
        )

        return self._index.get(
            key
        )

    def resolve_path(
        self,
        resource: VisualResource,
    ) -> Path:
        """
        Resolves the physical path of a
        local resource.

        Online resources do not have a
        physical local path.
        """

        if (
            resource.delivery
            != "local"
        ):
            raise ResourceCatalogError(
                "resolve_path() can only "
                "be used with local resources."
            )

        return (
            PROJECT_ROOT
            / resource.location
        )

    def is_available(
        self,
        concept_id: str,
        language: str,
        delivery: str,
    ) -> bool:

        resource = self.get_resource(
            concept_id,
            language,
            delivery,
        )

        if resource is None:
            return False

        if not resource.available:
            return False

        if resource.delivery == "local":

            physical_path = (
                self.resolve_path(
                    resource
                )
            )

            return (
                physical_path.is_file()
            )

        # For online resources, SC-07
        # validates the URL structurally.
        #
        # Runtime network/provider
        # availability will be evaluated
        # when the resource is consumed.
        return True

    def get_available_resource(
        self,
        concept_id: str,
        language: str,
        delivery: str,
    ) -> VisualResource | None:

        if not self.is_available(
            concept_id,
            language,
            delivery,
        ):
            return None

        return self.get_resource(
            concept_id,
            language,
            delivery,
        )

    def list_resources(
        self,
        language: str | None = None,
        delivery: str | None = None,
    ) -> list[VisualResource]:

        resources = list(
            self._resources
        )

        if language is not None:

            normalized_language = (
                language
                .strip()
                .upper()
            )

            if (
                normalized_language
                not in self.SUPPORTED_LANGUAGES
            ):
                raise ResourceCatalogError(
                    "Unsupported language: "
                    f"{normalized_language}"
                )

            resources = [
                resource
                for resource
                in resources
                if resource.language
                == normalized_language
            ]

        if delivery is not None:

            normalized_delivery = (
                delivery
                .strip()
                .lower()
            )

            if (
                normalized_delivery
                not in self.SUPPORTED_DELIVERIES
            ):
                raise ResourceCatalogError(
                    "Unsupported delivery: "
                    f"{normalized_delivery}"
                )

            resources = [
                resource
                for resource
                in resources
                if resource.delivery
                == normalized_delivery
            ]

        return resources
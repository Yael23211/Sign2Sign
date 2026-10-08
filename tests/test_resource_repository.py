from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.resources.resource_repository import (
    ResourceCatalogError,
    ResourceRepository,
)


class TestResourceRepository(unittest.TestCase):

    def setUp(self) -> None:

        self.temp_dir = tempfile.TemporaryDirectory()

        self.project_root = Path(
            self.temp_dir.name
        )

        self.catalog_path = (
            self.project_root
            / "data"
            / "resources"
            / "visual_resources.json"
        )

        self.catalog_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            self.project_root
            / "resources"
            / "asl"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            self.project_root
            / "resources"
            / "lsm"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.project_root_patch = patch(
            "src.resources.resource_repository.PROJECT_ROOT",
            self.project_root,
        )

        self.project_root_patch.start()

    def tearDown(self) -> None:

        self.project_root_patch.stop()
        self.temp_dir.cleanup()

    def write_catalog(
        self,
        resources: list,
    ) -> None:

        data = {
            "resources": resources
        }

        with self.catalog_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def local_resource(
        concept_id: str = "HELLO",
        language: str = "ASL",
        available: bool = True,
    ) -> dict:

        return {
            "concept_id":
                concept_id,
            "language":
                language,
            "delivery":
                "local",
            "location":
                (
                    f"resources/"
                    f"{language.lower()}/"
                    f"{concept_id}.mp4"
                ),
            "type":
                "video",
            "available":
                available,
        }

    @staticmethod
    def online_resource(
        concept_id: str = "HELLO",
        language: str = "LSM",
        available: bool = True,
    ) -> dict:

        return {
            "concept_id":
                concept_id,
            "language":
                language,
            "delivery":
                "online",
            "location":
                (
                    "https://example.com/"
                    f"{language.lower()}/"
                    f"{concept_id.lower()}"
                ),
            "type":
                "video",
            "available":
                available,
        }

    def test_empty_catalog_loads(
        self,
    ) -> None:

        self.write_catalog([])

        repository = ResourceRepository(
            self.catalog_path
        )

        self.assertEqual(
            repository.list_resources(),
            [],
        )

    def test_valid_local_resource_is_found(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.local_resource()
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        resource = repository.get_resource(
            "HELLO",
            "ASL",
            "local",
        )

        self.assertIsNotNone(
            resource
        )

        self.assertEqual(
            resource.concept_id,
            "HELLO",
        )

        self.assertEqual(
            resource.language,
            "ASL",
        )

        self.assertEqual(
            resource.delivery,
            "local",
        )

    def test_lookup_normalizes_case(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.local_resource()
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        resource = repository.get_resource(
            "hello",
            "asl",
            "LOCAL",
        )

        self.assertIsNotNone(
            resource
        )

        self.assertEqual(
            resource.concept_id,
            "HELLO",
        )

    def test_duplicate_same_delivery_is_rejected(
        self,
    ) -> None:

        resource = self.local_resource()

        self.write_catalog(
            [
                resource,
                dict(resource),
            ]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_local_and_online_can_coexist(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.local_resource(
                    concept_id="HELLO",
                    language="LSM",
                ),
                self.online_resource(
                    concept_id="HELLO",
                    language="LSM",
                ),
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        local_resource = (
            repository.get_resource(
                "HELLO",
                "LSM",
                "local",
            )
        )

        online_resource = (
            repository.get_resource(
                "HELLO",
                "LSM",
                "online",
            )
        )

        self.assertIsNotNone(
            local_resource
        )

        self.assertIsNotNone(
            online_resource
        )

        self.assertNotEqual(
            local_resource.delivery,
            online_resource.delivery,
        )

    def test_invalid_language_is_rejected(
        self,
    ) -> None:

        resource = self.local_resource()

        resource["language"] = "XYZ"

        self.write_catalog(
            [resource]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_invalid_delivery_is_rejected(
        self,
    ) -> None:

        resource = self.local_resource()

        resource["delivery"] = "cloud"

        self.write_catalog(
            [resource]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_invalid_type_is_rejected(
        self,
    ) -> None:

        resource = self.local_resource()

        resource["type"] = "image"

        self.write_catalog(
            [resource]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_unsafe_local_location_is_rejected(
        self,
    ) -> None:

        resource = self.local_resource()

        resource["location"] = (
            "../HELLO.mp4"
        )

        self.write_catalog(
            [resource]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_invalid_online_url_is_rejected(
        self,
    ) -> None:

        resource = self.online_resource()

        resource["location"] = (
            "youtube.com/video"
        )

        self.write_catalog(
            [resource]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_available_must_be_boolean(
        self,
    ) -> None:

        resource = self.local_resource()

        resource["available"] = "true"

        self.write_catalog(
            [resource]
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            ResourceRepository(
                self.catalog_path
            )

    def test_local_availability_requires_file(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.local_resource()
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        self.assertFalse(
            repository.is_available(
                "HELLO",
                "ASL",
                "local",
            )
        )

        video_path = (
            self.project_root
            / "resources"
            / "asl"
            / "HELLO.mp4"
        )

        video_path.touch()

        self.assertTrue(
            repository.is_available(
                "HELLO",
                "ASL",
                "local",
            )
        )

    def test_online_resource_is_available_structurally(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.online_resource()
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        self.assertTrue(
            repository.is_available(
                "HELLO",
                "LSM",
                "online",
            )
        )

    def test_available_false_returns_false(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.online_resource(
                    available=False
                )
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        self.assertFalse(
            repository.is_available(
                "HELLO",
                "LSM",
                "online",
            )
        )

    def test_resolve_path_rejects_online_resource(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.online_resource()
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        resource = repository.get_resource(
            "HELLO",
            "LSM",
            "online",
        )

        self.assertIsNotNone(
            resource
        )

        with self.assertRaises(
            ResourceCatalogError
        ):
            repository.resolve_path(
                resource
            )

    def test_list_resources_can_filter(
        self,
    ) -> None:

        self.write_catalog(
            [
                self.local_resource(
                    concept_id="HELLO",
                    language="ASL",
                ),
                self.online_resource(
                    concept_id="HELLO",
                    language="ASL",
                ),
                self.local_resource(
                    concept_id="WATER",
                    language="LSM",
                ),
            ]
        )

        repository = ResourceRepository(
            self.catalog_path
        )

        asl_local = (
            repository.list_resources(
                language="ASL",
                delivery="local",
            )
        )

        asl_online = (
            repository.list_resources(
                language="ASL",
                delivery="online",
            )
        )

        lsm_local = (
            repository.list_resources(
                language="LSM",
                delivery="local",
            )
        )

        self.assertEqual(
            len(asl_local),
            1,
        )

        self.assertEqual(
            len(asl_online),
            1,
        )

        self.assertEqual(
            len(lsm_local),
            1,
        )


if __name__ == "__main__":
    unittest.main()
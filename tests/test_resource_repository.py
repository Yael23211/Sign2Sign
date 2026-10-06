from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.resources.resource_repository import (
    ResourceCatalogError,
    ResourceRepository,
)


class ResourceRepositoryTests(
    unittest.TestCase
):

    def _write_catalog(
        self,
        root: Path,
        resources: list,
    ) -> Path:

        catalog_dir = (
            root
            / "data"
            / "resources"
        )

        catalog_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        catalog_path = (
            catalog_dir
            / "visual_resources.json"
        )

        with catalog_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "resources":
                        resources
                },
                file,
            )

        return catalog_path

    def _valid_resource(
        self,
        **overrides,
    ) -> dict:

        resource = {
            "concept_id":
                "WATER",
            "language":
                "ASL",
            "path":
                "resources/asl/WATER.mp4",
            "type":
                "video",
            "available":
                True,
        }

        resource.update(
            overrides
        )

        return resource

    def test_empty_catalog_loads(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [],
            )

            repository = (
                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )
            )

            self.assertEqual(
                len(repository),
                0,
            )

    def test_valid_resource_can_be_found(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource()
                ],
            )

            repository = (
                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )
            )

            resource = (
                repository.get_resource(
                    "WATER",
                    "ASL",
                )
            )

            self.assertIsNotNone(
                resource
            )

            self.assertEqual(
                resource.concept_id,
                "WATER",
            )

    def test_lookup_accepts_lowercase(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource()
                ],
            )

            repository = (
                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )
            )

            resource = (
                repository.get_resource(
                    "water",
                    "asl",
                )
            )

            self.assertIsNotNone(
                resource
            )

    def test_duplicate_resource_is_rejected(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource(),
                    self._valid_resource(),
                ],
            )

            with self.assertRaises(
                ResourceCatalogError
            ):

                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )

    def test_invalid_language_is_rejected(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource(
                        language="XYZ"
                    )
                ],
            )

            with self.assertRaises(
                ResourceCatalogError
            ):

                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )

    def test_invalid_type_is_rejected(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource(
                        type="image"
                    )
                ],
            )

            with self.assertRaises(
                ResourceCatalogError
            ):

                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )

    def test_unsafe_path_is_rejected(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource(
                        path=(
                            "../WATER.mp4"
                        )
                    )
                ],
            )

            with self.assertRaises(
                ResourceCatalogError
            ):

                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )

    def test_available_must_be_boolean(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource(
                        available="yes"
                    )
                ],
            )

            with self.assertRaises(
                ResourceCatalogError
            ):

                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )

    def test_is_available_requires_file(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource()
                ],
            )

            repository = (
                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )
            )

            self.assertFalse(
                repository.is_available(
                    "WATER",
                    "ASL",
                )
            )

            video_path = (
                root
                / "resources"
                / "asl"
                / "WATER.mp4"
            )

            video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            video_path.touch()

            self.assertTrue(
                repository.is_available(
                    "WATER",
                    "ASL",
                )
            )

    def test_unavailable_flag_returns_false(
        self,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            catalog = self._write_catalog(
                root,
                [
                    self._valid_resource(
                        available=False
                    )
                ],
            )

            video_path = (
                root
                / "resources"
                / "asl"
                / "WATER.mp4"
            )

            video_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            video_path.touch()

            repository = (
                ResourceRepository(
                    catalog_path=catalog,
                    project_root=root,
                )
            )

            self.assertFalse(
                repository.is_available(
                    "WATER",
                    "ASL",
                )
            )


if __name__ == "__main__":
    unittest.main()
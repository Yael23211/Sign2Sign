from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.visual_mapping import (
    InvalidCatalogError,
    InvalidLanguageError,
    ResourceFileMissingError,
    ResourceNotFoundError,
    ResourceUnavailableError,
    VisualResourceMapper,
)


class VisualResourceMapperTests(unittest.TestCase):

    def setUp(self) -> None:

        self.temp_dir = tempfile.TemporaryDirectory()

        self.root = Path(
            self.temp_dir.name
        )

        (
            self.root
            / "data"
            / "resources"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            self.root
            / "resources"
            / "asl"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            self.root
            / "resources"
            / "lsm"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.catalog_path = (
            self.root
            / "data"
            / "resources"
            / "visual_resources.json"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _touch_video(
        self,
        language: str,
        concept_id: str,
    ) -> str:

        location = (
            f"resources/"
            f"{language.lower()}/"
            f"{concept_id}.mp4"
        )

        path = (
            self.root
            / location
        )

        path.write_bytes(
            b"fake-video"
        )

        return location

    def _write_catalog(
        self,
        resources: list[dict],
    ) -> None:

        self.catalog_path.write_text(
            json.dumps(
                {
                    "resources": resources
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _resource(
        self,
        concept_id: str,
        language: str,
        *,
        available: bool = True,
        create_file: bool = True,
    ) -> dict:

        location = (
            f"resources/"
            f"{language.lower()}/"
            f"{concept_id}.mp4"
        )

        if create_file:
            self._touch_video(
                language,
                concept_id,
            )

        return {
            "concept_id": concept_id,
            "language": language,
            "delivery": "local",
            "location": location,
            "type": "video",
            "available": available,
        }

    def _mapper(
        self,
    ) -> VisualResourceMapper:

        return VisualResourceMapper(
            catalog_path=self.catalog_path,
            project_root=self.root,
        )

    def test_resolves_single_lsm_resource(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "LSM",
            ),
        ])

        result = (
            self._mapper()
            .resolve(
                ["WATER"],
                "LSM",
            )
        )

        self.assertEqual(
            result.concept_ids,
            ("WATER",),
        )

        self.assertEqual(
            result.locations,
            (
                "resources/lsm/WATER.mp4",
            ),
        )

    def test_preserves_input_order_and_duplicates(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "FIRST_PERSON_SINGULAR",
                "LSM",
            ),
            self._resource(
                "WANT",
                "LSM",
            ),
            self._resource(
                "WATER",
                "LSM",
            ),
        ])

        result = (
            self._mapper()
            .resolve(
                [
                    "WATER",
                    "WANT",
                    "WATER",
                    "FIRST_PERSON_SINGULAR",
                ],
                "LSM",
            )
        )

        self.assertEqual(
            result.concept_ids,
            (
                "WATER",
                "WANT",
                "WATER",
                "FIRST_PERSON_SINGULAR",
            ),
        )

    def test_asl_and_lsm_are_independent(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "ASL",
            ),
            self._resource(
                "WATER",
                "LSM",
            ),
        ])

        mapper = self._mapper()

        asl = mapper.resolve(
            ["WATER"],
            "ASL",
        )

        lsm = mapper.resolve(
            ["WATER"],
            "LSM",
        )

        self.assertEqual(
            asl.locations,
            (
                "resources/asl/WATER.mp4",
            ),
        )

        self.assertEqual(
            lsm.locations,
            (
                "resources/lsm/WATER.mp4",
            ),
        )

    def test_rejects_invalid_language(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "LSM",
            ),
        ])

        with self.assertRaises(
            InvalidLanguageError
        ):
            self._mapper().resolve(
                ["WATER"],
                "XYZ",
            )

    def test_raises_when_resource_is_not_registered(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "LSM",
            ),
        ])

        with self.assertRaises(
            ResourceNotFoundError
        ):
            self._mapper().resolve(
                ["HELLO"],
                "LSM",
            )

    def test_raises_when_resource_is_unavailable(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "LSM",
                available=False,
            ),
        ])

        with self.assertRaises(
            ResourceUnavailableError
        ):
            self._mapper().resolve(
                ["WATER"],
                "LSM",
            )

    def test_raises_when_local_file_is_missing(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "LSM",
                create_file=False,
            ),
        ])

        with self.assertRaises(
            ResourceFileMissingError
        ):
            self._mapper().resolve(
                ["WATER"],
                "LSM",
            )

    def test_rejects_duplicate_catalog_key(
        self,
    ) -> None:

        resource = self._resource(
            "WATER",
            "LSM",
        )

        self._write_catalog([
            resource,
            dict(resource),
        ])

        with self.assertRaises(
            InvalidCatalogError
        ):
            self._mapper()

    def test_normalizes_case_and_whitespace(
        self,
    ) -> None:

        self._write_catalog([
            self._resource(
                "WATER",
                "LSM",
            ),
        ])

        result = (
            self._mapper()
            .resolve(
                [" water "],
                " lsm ",
            )
        )

        self.assertEqual(
            result.concept_ids,
            ("WATER",),
        )


if __name__ == "__main__":
    unittest.main()
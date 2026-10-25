from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.visual_mapping.contracts import (
    ResourceMappingResult,
    VisualResource,
)

from src.visual_output import (
    UnsupportedDeliveryError,
    VideoFileMissingError,
    VisualSequencePlayer,
)


class VisualSequencePlayerTests(
    unittest.TestCase
):

    def setUp(self) -> None:

        self.temp_dir = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_dir.name
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

    def tearDown(self) -> None:

        self.temp_dir.cleanup()

    def _create_resource(
        self,
        concept_id: str,
        language: str,
        *,
        delivery: str = "local",
        create_file: bool = True,
    ) -> VisualResource:

        location = (
            f"resources/"
            f"{language.lower()}/"
            f"{concept_id}.mp4"
        )

        if create_file:

            path = (
                self.root
                / location
            )

            path.write_bytes(
                b"fake-video"
            )

        return VisualResource(
            concept_id=concept_id,
            language=language,
            delivery=delivery,
            location=location,
            type="video",
            available=True,
        )

    def _player(
        self,
    ) -> VisualSequencePlayer:

        return VisualSequencePlayer(
            project_root=self.root,
        )

    def test_builds_plan_in_original_order(
        self,
    ) -> None:

        mapping = ResourceMappingResult(
            target_language="ASL",
            delivery="local",
            resources=(
                self._create_resource(
                    "HELLO",
                    "ASL",
                ),
                self._create_resource(
                    "HOW_ARE_YOU",
                    "ASL",
                ),
            ),
        )

        plan = (
            self._player()
            .build_plan(
                mapping
            )
        )

        self.assertEqual(
            tuple(
                item.concept_id
                for item in plan.items
            ),
            (
                "HELLO",
                "HOW_ARE_YOU",
            ),
        )

    def test_positions_are_correct(
        self,
    ) -> None:

        mapping = ResourceMappingResult(
            target_language="LSM",
            delivery="local",
            resources=(
                self._create_resource(
                    "FIRST_PERSON_SINGULAR",
                    "LSM",
                ),
                self._create_resource(
                    "WANT",
                    "LSM",
                ),
                self._create_resource(
                    "WATER",
                    "LSM",
                ),
            ),
        )

        plan = (
            self._player()
            .build_plan(
                mapping
            )
        )

        self.assertEqual(
            tuple(
                (
                    item.position,
                    item.total,
                )
                for item in plan.items
            ),
            (
                (1, 3),
                (2, 3),
                (3, 3),
            ),
        )

    def test_rejects_online_delivery(
        self,
    ) -> None:

        mapping = ResourceMappingResult(
            target_language="LSM",
            delivery="online",
            resources=(
                self._create_resource(
                    "WATER",
                    "LSM",
                    delivery="online",
                ),
            ),
        )

        with self.assertRaises(
            UnsupportedDeliveryError
        ):

            self._player().build_plan(
                mapping
            )

    def test_detects_missing_file(
        self,
    ) -> None:

        mapping = ResourceMappingResult(
            target_language="ASL",
            delivery="local",
            resources=(
                self._create_resource(
                    "HELLO",
                    "ASL",
                    create_file=False,
                ),
            ),
        )

        with self.assertRaises(
            VideoFileMissingError
        ):

            self._player().build_plan(
                mapping
            )


if __name__ == "__main__":
    unittest.main()
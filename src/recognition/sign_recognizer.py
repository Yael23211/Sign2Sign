from __future__ import annotations

import inspect
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

from src.landmarks.pipeline import (
    create_hand_landmarker,
    detect_with_cascade,
    image_landmarks_to_arrays,
    world_landmarks_to_arrays,
)


SUPPORTED_LANGUAGES = {"ASL", "LSM"}
CLASS_LABELS = tuple(string.ascii_uppercase)


@dataclass(frozen=True)
class RecognitionResult:
    """
    Result produced by one deliberate recognition attempt.
    """

    source_language: str
    letter: str
    confidence: float
    top_k: tuple[tuple[str, float], ...]
    detection_variant: str

    def to_dict(self) -> dict:
        return {
            "source_language": self.source_language,
            "letter": self.letter,
            "confidence": self.confidence,
            "top_k": [
                {
                    "label": label,
                    "probability": probability,
                }
                for label, probability in self.top_k
            ],
            "detection_variant": self.detection_variant,
        }


class SignRecognizer:
    """
    Recognition component shared by ASL and LSM.

    Responsibilities:
    - hand detection;
    - landmark extraction;
    - normalization;
    - scaler transformation;
    - model inference;
    - Top-k generation.

    It does NOT:
    - open the webcam;
    - manage keyboard controls;
    - construct text;
    - correct spelling;
    - translate.
    """

    def __init__(
        self,
        source_language: str,
        mediapipe_model_path: str | Path,
        classifier_model_path: str | Path,
        scaler_path: str | Path,
        top_k_size: int = 3,
    ) -> None:

        language = source_language.strip().upper()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {source_language!r}. "
                f"Expected ASL or LSM."
            )

        if top_k_size < 1:
            raise ValueError(
                "top_k_size must be at least 1."
            )

        self.source_language = language
        self.top_k_size = top_k_size

        self.mediapipe_model_path = Path(
            mediapipe_model_path
        )

        self.classifier_model_path = Path(
            classifier_model_path
        )

        self.scaler_path = Path(
            scaler_path
        )

        self._validate_paths()

        # create_hand_landmarker requires str in the local
        # Sign2Sign environment.
        self.landmarker = create_hand_landmarker(
            str(self.mediapipe_model_path)
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        self.model = tf.keras.models.load_model(
            self.classifier_model_path,
            compile=False,
        )

        self._validate_model()

    def _validate_paths(self) -> None:

        paths = {
            "MediaPipe model": self.mediapipe_model_path,
            "classifier": self.classifier_model_path,
            "scaler": self.scaler_path,
        }

        missing = [
            f"{name}: {path}"
            for name, path in paths.items()
            if not path.exists()
        ]

        if missing:
            details = "\n".join(missing)

            raise FileNotFoundError(
                "Required recognition artifacts were not found:\n"
                f"{details}"
            )

    def _validate_model(self) -> None:
        """
        Basic compatibility checks for the frozen classifiers.
        """

        output_shape = self.model.output_shape

        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        number_of_classes = int(
            output_shape[-1]
        )

        if number_of_classes != len(CLASS_LABELS):
            raise ValueError(
                "Unexpected model output dimension. "
                f"Expected 26 classes, got "
                f"{number_of_classes}."
            )

        if hasattr(self.scaler, "n_features_in_"):
            if int(self.scaler.n_features_in_) != 63:
                raise ValueError(
                    "Unexpected scaler input dimension. "
                    f"Expected 63 features, got "
                    f"{self.scaler.n_features_in_}."
                )

    def recognize(
        self,
        frame: np.ndarray,
    ) -> RecognitionResult | None:
        """
        Recognizes one letter from a single webcam frame.

        Returns None when MediaPipe cannot detect a valid hand.
        """

        if frame is None:
            raise ValueError(
                "Frame cannot be None."
            )

        detection = detect_with_cascade(
            self.landmarker,
            frame,
        )

        if not detection.get("ok", False):
            return None

        result = detection["result"]
        processed = detection["processed"]

        if self.source_language == "ASL":
            features = self._extract_img_norm(
                result=result,
                processed=processed,
            )

        else:
            features = self._extract_world_norm(
                result=result,
                processed=processed,
            )

        x = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        if x.shape != (1, 63):
            raise ValueError(
                "Unexpected feature vector shape: "
                f"{x.shape}. Expected (1, 63)."
            )

        scaled = self.scaler.transform(x)

        probabilities = self.model.predict(
            scaled,
            verbose=0,
        )[0]

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if probabilities.shape[0] != 26:
            raise ValueError(
                "Classifier returned an unexpected "
                f"number of classes: {probabilities.shape[0]}"
            )

        top_indices = np.argsort(
            probabilities
        )[::-1][: self.top_k_size]

        top_k = tuple(
            (
                CLASS_LABELS[int(index)],
                float(probabilities[int(index)]),
            )
            for index in top_indices
        )

        top_index = int(top_indices[0])

        return RecognitionResult(
            source_language=self.source_language,
            letter=CLASS_LABELS[top_index],
            confidence=float(
                probabilities[top_index]
            ),
            top_k=top_k,
            detection_variant=str(
                detection.get(
                    "variant",
                    "unknown",
                )
            ),
        )

    def _extract_img_norm(
        self,
        result: Any,
        processed: np.ndarray,
    ) -> np.ndarray:

        hand_landmarks = getattr(
            result,
            "hand_landmarks",
            None,
        )

        if not hand_landmarks:
            raise ValueError(
                "Detection succeeded but image "
                "landmarks are missing."
            )

        height, width = processed.shape[:2]

        converted = self._call_converter(
            converter=image_landmarks_to_arrays,
            landmarks=hand_landmarks[0],
            width=width,
            height=height,
        )

        return self._extract_normalized_vector(
            converted,
            expected_name="img_norm",
        )

    def _extract_world_norm(
        self,
        result: Any,
        processed: np.ndarray,
    ) -> np.ndarray:

        hand_world_landmarks = getattr(
            result,
            "hand_world_landmarks",
            None,
        )

        if not hand_world_landmarks:
            raise ValueError(
                "Detection succeeded but world "
                "landmarks are missing."
            )

        height, width = processed.shape[:2]

        converted = self._call_converter(
            converter=world_landmarks_to_arrays,
            landmarks=hand_world_landmarks[0],
            width=width,
            height=height,
        )

        return self._extract_normalized_vector(
            converted,
            expected_name="world_norm",
        )

    @staticmethod
    def _call_converter(
        converter,
        landmarks,
        width: int,
        height: int,
    ):
        """
        Calls the existing pipeline converter while supporting
        the signatures observed during Sign2Sign development:

            converter(landmarks)

        or:

            converter(landmarks, width, height)
        """

        signature = inspect.signature(
            converter
        )

        positional_parameters = [
            parameter
            for parameter
            in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if len(positional_parameters) >= 3:
            return converter(
                landmarks,
                width,
                height,
            )

        return converter(
            landmarks
        )

    @classmethod
    def _extract_normalized_vector(
        cls,
        converted,
        expected_name: str,
    ) -> np.ndarray:
        """
        Extracts the normalized 63-value vector returned by
        the shared pipeline.

        The method is intentionally defensive because the
        pipeline evolved during the project.
        """

        # Case 1: mapping/dictionary.
        if isinstance(converted, dict):

            preferred_keys = (
                expected_name,
                "norm",
                "normalized",
                "normalized_points",
            )

            for key in preferred_keys:
                if key in converted:
                    vector = cls._as_63_vector(
                        converted[key]
                    )

                    if vector is not None:
                        return vector

            # Search any key containing "norm".
            for key, value in converted.items():
                if "norm" in str(key).lower():
                    vector = cls._as_63_vector(
                        value
                    )

                    if vector is not None:
                        return vector

            xyz_vector = cls._vector_from_xyz_mapping(
                converted
            )

            if xyz_vector is not None:
                return xyz_vector

        # Case 2: tuple/list.
        if isinstance(converted, (tuple, list)):

            # If the converter returns raw + normalized,
            # normalized is normally the last 63-value item.
            for item in reversed(converted):
                vector = cls._as_63_vector(
                    item
                )

                if vector is not None:
                    return vector

            # Support:
            # (x_raw, y_raw, z_raw, x_norm, y_norm, z_norm)
            if len(converted) >= 3:
                last_three = converted[-3:]

                xyz_vector = cls._combine_xyz(
                    *last_three
                )

                if xyz_vector is not None:
                    return xyz_vector

        # Case 3: converter directly returns 63 values.
        vector = cls._as_63_vector(
            converted
        )

        if vector is not None:
            return vector

        raise ValueError(
            "Could not obtain the normalized 63-value "
            f"vector for {expected_name}. "
            "The return contract of the landmark converter "
            "does not match the supported formats."
        )

    @staticmethod
    def _as_63_vector(
        value,
    ) -> np.ndarray | None:

        try:
            array = np.asarray(
                value,
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return None

        if array.size != 63:
            return None

        return array.reshape(63)

    @staticmethod
    def _combine_xyz(
        x,
        y,
        z,
    ) -> np.ndarray | None:

        try:
            x_array = np.asarray(
                x,
                dtype=np.float32,
            ).reshape(-1)

            y_array = np.asarray(
                y,
                dtype=np.float32,
            ).reshape(-1)

            z_array = np.asarray(
                z,
                dtype=np.float32,
            ).reshape(-1)

        except (TypeError, ValueError):
            return None

        if not (
            x_array.size
            == y_array.size
            == z_array.size
            == 21
        ):
            return None

        return np.column_stack(
            (
                x_array,
                y_array,
                z_array,
            )
        ).reshape(63)

    @classmethod
    def _vector_from_xyz_mapping(
        cls,
        mapping: dict,
    ) -> np.ndarray | None:

        lower_mapping = {
            str(key).lower(): value
            for key, value in mapping.items()
        }

        possible_sets = (
            ("x_norm", "y_norm", "z_norm"),
            (
                "norm_x",
                "norm_y",
                "norm_z",
            ),
            (
                "normalized_x",
                "normalized_y",
                "normalized_z",
            ),
        )

        for x_key, y_key, z_key in possible_sets:

            if all(
                key in lower_mapping
                for key in (
                    x_key,
                    y_key,
                    z_key,
                )
            ):
                vector = cls._combine_xyz(
                    lower_mapping[x_key],
                    lower_mapping[y_key],
                    lower_mapping[z_key],
                )

                if vector is not None:
                    return vector

        return None

    def close(self) -> None:
        """
        Releases MediaPipe resources when supported.
        """

        close_method = getattr(
            self.landmarker,
            "close",
            None,
        )

        if callable(close_method):
            close_method()
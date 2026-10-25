from dataclasses import dataclass, field
from threading import Lock

from .settings import ROOT


@dataclass
class Components:
    gemini: object
    mapper: object
    player: object
    repository: object
    recognizers: dict
    recognition_lock: object = field(default_factory=Lock)
    translation_lock: object = field(default_factory=Lock)

    def close(self):
        for recognizer in self.recognizers.values():
            recognizer.close()
        client = getattr(self.gemini, "client", None)
        if client is not None and callable(getattr(client, "close", None)):
            client.close()


def load_components(settings):
    # Lazy imports let HTTP/controller tests use doubles without loading TF.
    from src.recognition import SignRecognizer
    from src.text_processing import Vocabulary, GeminiClient
    from src.visual_mapping import VisualResourceMapper
    from src.visual_output import VisualSequencePlayer
    from src.resources import ResourceRepository

    components = Components(None, None, None, None, {})
    try:
        vocabulary = Vocabulary(ROOT / "data/vocabulary/sign2sign_vocabulary.json")
        components.gemini = GeminiClient(vocabulary, model=settings.gemini_model)
        components.mapper = VisualResourceMapper()
        components.player = VisualSequencePlayer()
        components.repository = ResourceRepository()
        for language, model, scaler in (
            ("ASL", "A3_mlp_img_norm.keras", "scaler_img_norm.joblib"),
            ("LSM", "V2_M2_mlp_world_norm.keras", "scaler_world_norm.joblib"),
        ):
            folder = ROOT / "models" / language.lower()
            components.recognizers[language] = SignRecognizer(
                source_language=language,
                mediapipe_model_path=ROOT / "models/mediapipe/hand_landmarker.task",
                classifier_model_path=folder / model, scaler_path=folder / scaler,
                top_k_size=3,
            )
        return components
    except BaseException:
        components.close()
        raise

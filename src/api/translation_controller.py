"""SC-02 orchestration; synchronous methods are dispatched off the event loop."""
from contextlib import contextmanager
from io import BytesIO
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.text import EmptyTextError
from src.text_processing.contracts import CorrectionRequest
from .schemas import CaptureImage


class FlowError(Exception):
    def __init__(self, status, code, message, stage=None):
        self.status, self.code, self.message, self.stage = status, code, message, stage


class TranslationController:
    def __init__(self, store, components, settings):
        self.store, self.components, self.settings = store, components, settings

    def get(self, session_id):
        try:
            return self.store.get(session_id)
        except KeyError:
            raise FlowError(404, "session_not_found", "Session not found") from None

    @contextmanager
    def exclusive(self, session_id):
        session = self.get(session_id)
        if not session.lock.acquire(blocking=False):
            raise FlowError(409, "session_busy", "Session is processing another request")
        try:
            if session.state == "closed":
                raise FlowError(404, "session_not_found", "Session not found")
            yield session
        finally:
            session.lock.release()

    @staticmethod
    def snapshot(session):
        return dict(id=session.id, source_language=session.source_language,
                    target_language=session.target_language, state=session.state,
                    text=session.builder.current_text, last_result=session.last_result)

    def create(self, language):
        try:
            return self.snapshot(self.store.create(language))
        except OverflowError:
            raise FlowError(503, "session_capacity", "Close an existing session first") from None

    def inspect(self, session_id):
        # A busy session cannot be read halfway through a mutation.
        with self.exclusive(session_id) as session:
            return self.snapshot(session)

    def command(self, session_id, command):
        with self.exclusive(session_id) as session:
            {"space": session.builder.add_space, "backspace": session.builder.backspace,
             "clear": session.builder.clear}[command]()
            session.state, session.last_result = "capturing", None
            return self.snapshot(session)

    def capture(self, session_id, image: CaptureImage):
        with self.exclusive(session_id) as session:
            if session.builder.letter_count >= self.settings.max_letters:
                raise FlowError(422, "text_limit", "Maximum letter count reached")
            if len(image.data) > self.settings.max_capture_bytes:
                raise FlowError(413, "capture_too_large", "Capture exceeds size limit")
            try:
                with Image.open(BytesIO(image.data)) as header:
                    expected = {"image/jpeg": "JPEG", "image/png": "PNG"}[image.content_type]
                    if header.format != expected:
                        raise ValueError("Image type mismatch")
                    if header.width * header.height > self.settings.max_image_pixels:
                        raise ValueError("Image dimensions exceed limit")
                    header.verify()
                # Ignore EXIF rotation; never flip the pixels sent by the client.
                frame = cv2.imdecode(np.frombuffer(image.data, np.uint8),
                                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if frame is None:
                    raise ValueError("Cannot decode image")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                raise FlowError(422, "invalid_image", "Invalid image or image dimensions") from None
            with self.components.recognition_lock:
                result = self.components.recognizers[session.source_language].recognize(frame)
            if result is not None:
                session.builder.add_prediction(result.letter, result.confidence, result.top_k)
                session.state, session.last_result = "capturing", None
            return dict(session=self.snapshot(session),
                        recognition=result.to_dict() if result else None)

    def translate(self, session_id):
        with self.exclusive(session_id) as session:
            try:
                built = session.builder.finalize()
            except EmptyTextError:
                raise FlowError(422, "empty_text", "Capture text before translating") from None
            session.state, session.last_result = "processing", None
            try:
                # Shared LLM and mapping components are serialized conservatively.
                with self.components.translation_lock:
                    return self._pipeline(session, built)
            except FlowError:
                session.state = "failed"
                raise
            except Exception:
                session.state = "failed"
                raise FlowError(500, "translation_failed", "Translation could not complete") from None

    def _pipeline(self, session, built):
        request = CorrectionRequest.from_built_text(built, session.target_language)
        try:
            call = self.components.gemini.process(request)
        except Exception:
            raise FlowError(502, "sc04_failed", "SC-04 correction/translation failed; input preserved",
                            "SC-04") from None
        result = dict(**call.result.to_dict(), resources=[], latency_seconds=call.latency_seconds,
                      state="ready", error=None)
        if call.result.concepts:
            stage = "SC-05"
            try:
                mapping = self.components.mapper.resolve_translation_result(call.result, delivery="local")
                stage = "SC-06"
                plan = self.components.player.build_plan(mapping)
                result["resources"] = [dict(
                    concept_id=item.concept_id, language=item.language,
                    url=f"/api/v1/resources/{quote(item.language, safe='')}/{quote(item.concept_id, safe='')}",
                    position=item.position, total=item.total,
                ) for item in plan.items]
            except Exception:
                result.update(state="failed", error=dict(
                    stage=stage, code="visual_preparation_failed",
                    message=f"{stage} visual preparation failed; SC-04 result preserved"))
        session.state, session.last_result = result["state"], result
        return result

    def delete(self, session_id):
        with self.exclusive(session_id) as session:
            session.state = "closed"
            session.builder.clear()
            session.last_result = None
            self.store.delete(session_id)

    def resource_path(self, language, concept_id):
        repository = self.components.repository
        resource = repository.get_available_resource(concept_id, language, "local")
        if resource is None:
            raise FlowError(404, "resource_not_found", "Resource is unavailable")
        return repository.resolve_path(resource)

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.dependencies import Components
from src.api.settings import Settings
from src.api.schemas import CaptureImage
from src.api.translation_controller import TranslationController, FlowError
from src.resources.temporary_session_store import TemporarySessionStore
from src.text_processing.contracts import TranslationResult
from src.visual_mapping.contracts import ResourceMappingResult, VisualResource
from src.visual_output import VisualSequencePlayer


def image_bytes():
    output = BytesIO()
    image = Image.new('RGB', (8, 6), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    image.save(output, format='PNG')
    return output.getvalue()


def components(root):
    resource = VisualResource('HELLO', 'ASL', 'local', 'hello.mp4', 'video', True)
    Path(root, 'hello.mp4').write_bytes(b'test-video')
    result = TranslationResult('LSM', 'ASL', 'H', 'HOLA', 'HELLO', ('HELLO',), ())
    recognition = SimpleNamespace(letter='H', confidence=.9, top_k=(('H', .9), ('A', .1)),
                                  to_dict=lambda: {'letter': 'H', 'confidence': .9,
                                                   'top_k': [{'label': 'H', 'probability': .9},
                                                             {'label': 'A', 'probability': .1}]})
    recognizer = Mock()
    recognizer.recognize.return_value = recognition
    gemini = Mock()
    gemini.process.return_value = SimpleNamespace(result=result, latency_seconds=.25)
    repository = Mock()
    repository.get_available_resource.return_value = resource
    repository.resolve_path.return_value = Path(root, 'hello.mp4')
    mapper = Mock()
    mapper.resolve_translation_result.return_value = ResourceMappingResult('ASL', 'local', (resource,))
    player = VisualSequencePlayer(project_root=root)
    player.play = Mock(side_effect=AssertionError('Must never open a window'))
    return Components(gemini, mapper, player, repository, {'ASL': recognizer, 'LSM': recognizer})


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parts = components(self.temp.name)
        self.controller = TranslationController(TemporarySessionStore(), self.parts, Settings())
        self.sid = self.controller.create('LSM')['id']

    def capture(self):
        return self.controller.capture(self.sid, CaptureImage(content_type='image/png', data=image_bytes()))

    def test_capture_preserves_confidence_topk_and_bgr_orientation(self):
        response = self.capture()
        self.assertEqual(response['session']['text'], 'H')
        self.assertEqual(response['recognition']['confidence'], .9)
        frame = self.parts.recognizers['LSM'].recognize.call_args.args[0]
        self.assertEqual(frame.shape, (6, 8, 3))
        np.testing.assert_array_equal(frame[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(frame[0, -1], [0, 0, 0])
        self.assertEqual(self.controller.get(self.sid).builder.tokens[0].top_k[1].label, 'A')

    def test_translation_uses_contracts_plan_and_http_urls(self):
        self.capture()
        response = self.controller.translate(self.sid)
        request = self.parts.gemini.process.call_args.args[0]
        self.assertEqual(request.recognized_text, 'H')
        self.assertEqual(request.tokens[0]['top_k'][0]['probability'], .9)
        self.assertEqual(response['resources'][0]['url'], '/api/v1/resources/ASL/HELLO')
        self.assertEqual(response['state'], 'ready')
        self.parts.player.play.assert_not_called()

    def test_sc04_failure_preserves_input_and_allows_retry(self):
        self.capture()
        self.parts.gemini.process.side_effect = RuntimeError('secret C:/private')
        with self.assertRaises(FlowError) as caught:
            self.controller.translate(self.sid)
        self.assertEqual(caught.exception.stage, 'SC-04')
        self.assertNotIn('private', caught.exception.message)
        self.assertEqual(self.controller.inspect(self.sid)['text'], 'H')
        self.parts.gemini.process.side_effect = None
        self.assertEqual(self.controller.translate(self.sid)['state'], 'ready')

    def test_unresolved_only_skips_visuals(self):
        self.capture()
        self.parts.gemini.process.return_value.result = TranslationResult('LSM', 'ASL', 'H', '', '', (), ('H',))
        result = self.controller.translate(self.sid)
        self.assertEqual(result['unresolved'], ['H'])
        self.assertEqual(result['resources'], [])
        self.parts.mapper.resolve_translation_result.assert_not_called()

    def test_mixed_unresolved_preserved(self):
        self.capture()
        self.parts.gemini.process.return_value.result = TranslationResult('LSM', 'ASL', 'H', 'HOLA', 'HELLO', ('HELLO',), ('X',))
        result = self.controller.translate(self.sid)
        self.assertEqual(result['unresolved'], ['X'])
        self.assertEqual(len(result['resources']), 1)

    def test_visual_failures_preserve_sc04_and_identify_stage(self):
        self.capture()
        self.parts.mapper.resolve_translation_result.side_effect = RuntimeError('private')
        result = self.controller.translate(self.sid)
        self.assertEqual(result['error']['stage'], 'SC-05')
        self.assertEqual(result['translated_text'], 'HELLO')
        self.parts.mapper.resolve_translation_result.side_effect = None
        self.parts.player.build_plan = Mock(side_effect=RuntimeError('private'))
        result = self.controller.translate(self.sid)
        self.assertEqual(result['error']['stage'], 'SC-06')
        self.assertEqual(self.controller.inspect(self.sid)['last_result'], result)

    def test_no_hand_does_not_change_text(self):
        self.parts.recognizers['LSM'].recognize.return_value = None
        self.assertIsNone(self.capture()['recognition'])
        self.assertEqual(self.controller.inspect(self.sid)['text'], '')

    def test_sessions_are_isolated_and_busy_session_rejected(self):
        self.capture()
        second = self.controller.create('ASL')['id']
        self.assertEqual(self.controller.inspect(second)['text'], '')
        with self.controller.exclusive(self.sid):
            with self.assertRaises(FlowError) as caught:
                self.controller.command(self.sid, 'clear')
            self.assertEqual(caught.exception.status, 409)

    def test_dimensions_and_text_limits(self):
        self.controller.settings = Settings(max_image_pixels=1)
        with self.assertRaises(FlowError) as caught:
            self.capture()
        self.assertEqual(caught.exception.code, 'invalid_image')
        self.controller.settings = Settings(max_letters=1)
        self.capture()
        with self.assertRaises(FlowError) as caught:
            self.capture()
        self.assertEqual(caught.exception.code, 'text_limit')

    def test_recognition_serialized_between_sessions(self):
        entered, release = Event(), Event()
        original = self.parts.recognizers['LSM'].recognize.return_value
        second = self.controller.create('ASL')['id']
        def recognize(frame):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('Test timed out')
            return original
        self.parts.recognizers['LSM'].recognize.side_effect = recognize
        with ThreadPoolExecutor() as executor:
            first = executor.submit(self.capture)
            try:
                self.assertTrue(entered.wait(2))
                # The shared lock is held throughout recognition.
                self.assertFalse(self.parts.recognition_lock.acquire(blocking=False))
                other = executor.submit(self.controller.capture, second,
                                        CaptureImage(content_type='image/png', data=image_bytes()))
                self.assertFalse(other.done())
            finally:
                release.set()
            self.assertEqual(first.result()['session']['text'], 'H')
            self.assertEqual(other.result()['session']['text'], 'H')

    def test_empty_translation_rejected(self):
        with self.assertRaises(FlowError) as caught:
            self.controller.translate(self.sid)
        self.assertEqual(caught.exception.status, 422)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parts = components(self.temp.name)
        self.factory = Mock(return_value=self.parts)
        self.client = TestClient(create_app(self.factory, Settings(max_capture_bytes=1000, max_sessions=2)))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.sid = self.client.post('/api/v1/sessions', json={'source_language': 'LSM'}).json()['id']
        self.base = '/api/v1/sessions/' + self.sid

    def capture(self):
        return self.client.post(self.base + '/captures', content=image_bytes(), headers={'content-type': 'image/png'})

    def test_endpoints_and_commands(self):
        self.assertEqual(self.client.get('/api/v1/health').status_code, 200)
        self.factory.assert_called_once()
        self.assertEqual(self.capture().status_code, 200)
        for command in ('space', 'backspace'):
            self.assertEqual(self.client.post(self.base + '/commands', json={'command': command}).status_code, 200)
        result = self.client.post(self.base + '/translations')
        self.assertEqual(result.status_code, 200, result.text)
        video = self.client.get(result.json()['resources'][0]['url'])
        self.assertEqual(video.content, b'test-video')
        self.assertEqual(self.client.get('/api/v1/resources/ASL/HELLO', headers={'Range': 'bytes=0-3'}).status_code, 206)
        self.assertEqual(self.client.post(self.base + '/commands', json={'command': 'clear'}).json()['text'], '')
        self.assertEqual(self.client.delete(self.base).status_code, 204)
        self.assertEqual(self.client.get(self.base).status_code, 404)

    def test_shutdown_clears_store_and_closes_dependencies(self):
        parts = components(self.temp.name)
        with TestClient(create_app(lambda settings: parts)) as client:
            client.post('/api/v1/sessions', json={'source_language': 'ASL'})
            store = client.app.state.controller.store
        self.assertEqual(store._sessions, {})
        parts.recognizers['ASL'].close.assert_called()

    def test_sc04_failure_returns_safe_http_error_and_preserves_text(self):
        self.capture()
        self.parts.gemini.process.side_effect = RuntimeError('C:/secret')
        response = self.client.post(self.base + '/translations')
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()['error']['stage'], 'SC-04')
        self.assertNotIn('secret', response.text)
        self.assertEqual(self.client.get(self.base).json()['text'], 'H')
        self.assertEqual(self.client.get(self.base).json()['state'], 'failed')

    def test_http_validation_and_limits(self):
        self.assertEqual(self.client.post('/api/v1/sessions', json={'source_language': 'ES'}).status_code, 422)
        self.assertEqual(self.client.post(self.base + '/commands', json={'command': 'unknown'}).status_code, 422)
        self.assertEqual(self.client.post(self.base + '/translations').status_code, 422)
        for data, mime, expected in ((b'bad', 'image/png', 422), (b'', 'image/png', 422),
                                     (b'x' * 1001, 'image/png', 413), (image_bytes(), 'image/jpeg', 422),
                                     (b'bad', 'text/plain', 415)):
            self.assertEqual(self.client.post(self.base + '/captures', content=data,
                                             headers={'content-type': mime}).status_code, expected)
        self.client.post('/api/v1/sessions', json={'source_language': 'ASL'})
        self.assertEqual(self.client.post('/api/v1/sessions', json={'source_language': 'ASL'}).status_code, 503)
        self.parts.repository.get_available_resource.return_value = None
        self.assertEqual(self.client.get('/api/v1/resources/ASL/UNKNOWN').status_code, 404)

    def test_blocking_translation_does_not_block_health_or_other_sessions(self):
        self.capture()
        entered, release = Event(), Event()
        original = self.parts.gemini.process.return_value
        def slow_process(request):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test timed out')
            return original
        self.parts.gemini.process.side_effect = slow_process
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.client.post, self.base + '/translations')
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.client.get('/api/v1/health').status_code, 200)
                self.assertEqual(self.client.post(self.base + '/commands', json={'command': 'clear'}).status_code, 409)
                self.assertEqual(self.client.delete(self.base).status_code, 409)
                self.assertEqual(self.client.post('/api/v1/sessions', json={'source_language': 'ASL'}).status_code, 201)
            finally:
                release.set()
            self.assertEqual(future.result().status_code, 200)


if __name__ == '__main__':
    unittest.main()

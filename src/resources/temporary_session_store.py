"""SC-07: process-local, bounded temporary session storage."""
from dataclasses import dataclass, field
from threading import Lock, RLock
from uuid import uuid4

from src.text import TextBuilder


@dataclass
class TranslationSession:
    id: str
    source_language: str
    target_language: str
    builder: TextBuilder
    state: str = "capturing"
    last_result: dict | None = None
    lock: object = field(default_factory=Lock, repr=False)


class TemporarySessionStore:
    def __init__(self, max_sessions: int = 100):
        self.max_sessions = max_sessions
        self._sessions: dict[str, TranslationSession] = {}
        self._lock = RLock()

    def create(self, language: str) -> TranslationSession:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise OverflowError("Session capacity reached")
            session = TranslationSession(
                id=uuid4().hex, source_language=language,
                target_language="LSM" if language == "ASL" else "ASL",
                builder=TextBuilder(language),
            )
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> TranslationSession:
        with self._lock:
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            del self._sessions[session_id]

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

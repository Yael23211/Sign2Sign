from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    max_capture_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 4096 * 4096
    max_letters: int = 500
    max_sessions: int = 100

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        return cls()

    @property
    def gemini_model(self):
        return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

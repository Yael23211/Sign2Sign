from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "No se encontró GEMINI_API_KEY."
        )

    client = genai.Client(
        api_key=api_key
    )

    print(
        "Enviando prueba a Gemini..."
    )

    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=(
            "Respond only with the word OK."
        ),
        generation_config={
            "thinking_level": "minimal"
        },
    )

    print(
        "Respuesta:",
        response.output_text,
    )


if __name__ == "__main__":
    main()
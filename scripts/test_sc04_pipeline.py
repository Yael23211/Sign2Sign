from __future__ import annotations

import json
import sys
from pathlib import Path


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.text import (
    BuiltText,
    PredictionCandidate,
    TextToken,
)

from src.text_processing import (
    CorrectionRequest,
    PromptBuilder,
    ResponseValidator,
    Vocabulary,
)


VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "sign2sign_vocabulary.json"
)


# ============================================================
# CONSTRUCCIÓN DE ENTRADA SIMULADA DE SC-03
# ============================================================

def build_mock_sc03_output() -> BuiltText:
    """
    Simula una salida realista de SC-03.

    La palabra pretendida es AGUA, pero el clasificador
    produjo R como Top-1 en la tercera posición.

    Top-k:
        R = 54 %
        U = 44 %
        V = 2 %

    Esto permite probar que SC-04 recibe suficiente
    información para que posteriormente el LLM pueda
    corregir AGRA -> AGUA.
    """

    return BuiltText(
        source_language="LSM",
        text="AGRA",
        tokens=(
            TextToken(
                kind="letter",
                value="A",
                top_k=(
                    PredictionCandidate(
                        "A",
                        0.97,
                    ),
                    PredictionCandidate(
                        "E",
                        0.02,
                    ),
                    PredictionCandidate(
                        "O",
                        0.01,
                    ),
                ),
            ),

            TextToken(
                kind="letter",
                value="G",
                top_k=(
                    PredictionCandidate(
                        "G",
                        0.94,
                    ),
                    PredictionCandidate(
                        "H",
                        0.04,
                    ),
                    PredictionCandidate(
                        "Q",
                        0.02,
                    ),
                ),
            ),

            TextToken(
                kind="letter",
                value="R",
                top_k=(
                    PredictionCandidate(
                        "R",
                        0.54,
                    ),
                    PredictionCandidate(
                        "U",
                        0.44,
                    ),
                    PredictionCandidate(
                        "V",
                        0.02,
                    ),
                ),
            ),

            TextToken(
                kind="letter",
                value="A",
                top_k=(
                    PredictionCandidate(
                        "A",
                        0.98,
                    ),
                    PredictionCandidate(
                        "E",
                        0.01,
                    ),
                    PredictionCandidate(
                        "O",
                        0.01,
                    ),
                ),
            ),
        ),
    )


# ============================================================
# RESPUESTA SIMULADA DEL LLM
# ============================================================

def build_mock_llm_response() -> str:
    """
    Simula la respuesta que posteriormente debería
    producir Gemini.

    Todavía NO se realiza ninguna llamada externa.
    """

    response = {
        "corrected_text": "AGUA",
        "translated_text": "WATER",
        "concepts": [
            "WATER",
        ],
        "unresolved": [],
    }

    return json.dumps(
        response,
        ensure_ascii=False,
    )


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("SIGN2SIGN - PRUEBA END-TO-END DE SC-04")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Cargar vocabulario
    # --------------------------------------------------------

    print()
    print("[1] Cargando vocabulario canónico...")

    vocabulary = Vocabulary(
        VOCAB_PATH
    )

    print(
        f"    Conceptos cargados: {len(vocabulary)}"
    )

    # --------------------------------------------------------
    # 2. Simular salida de SC-03
    # --------------------------------------------------------

    print()
    print("[2] Generando salida simulada de SC-03...")

    built_text = build_mock_sc03_output()

    print(
        f"    Lengua origen: "
        f"{built_text.source_language}"
    )

    print(
        f"    Texto reconocido: "
        f"{built_text.text}"
    )

    # --------------------------------------------------------
    # 3. Crear CorrectionRequest
    # --------------------------------------------------------

    print()
    print("[3] Construyendo CorrectionRequest...")

    request = CorrectionRequest.from_built_text(
        built_text,
        target_language="ASL",
    )

    print(
        f"    Flujo: "
        f"{request.source_language}"
        f" -> "
        f"{request.target_language}"
    )

    print(
        f"    Idioma puente: "
        f"{request.source_bridge_language}"
        f" -> "
        f"{request.target_bridge_language}"
    )

    # --------------------------------------------------------
    # 4. Construir prompt
    # --------------------------------------------------------

    print()
    print("[4] Construyendo prompt...")

    prompt_builder = PromptBuilder(
        vocabulary
    )

    prompt = prompt_builder.build(
        request
    )

    print(
        f"    Longitud del prompt: "
        f"{len(prompt)} caracteres"
    )

    print()
    print("-" * 72)
    print("FRAGMENTO RELEVANTE DEL PROMPT")
    print("-" * 72)

    # No imprimimos todo el vocabulario para no llenar
    # innecesariamente la consola.
    marker_start = prompt.find(
        "RECOGNIZED TEXT"
    )

    marker_end = prompt.find(
        "CANONICAL SIGN2SIGN VOCABULARY"
    )

    if (
        marker_start != -1
        and marker_end != -1
    ):
        print(
            prompt[
                marker_start:
                marker_end
            ].strip()
        )

    else:
        print(
            "No se pudo localizar "
            "el fragmento esperado."
        )

    # --------------------------------------------------------
    # 5. Simular respuesta del LLM
    # --------------------------------------------------------

    print()
    print("[5] Simulando respuesta del LLM...")

    raw_response = (
        build_mock_llm_response()
    )

    print(
        f"    Respuesta: {raw_response}"
    )

    # --------------------------------------------------------
    # 6. Validar respuesta
    # --------------------------------------------------------

    print()
    print("[6] Validando respuesta...")

    validator = ResponseValidator(
        vocabulary
    )

    result = validator.validate(
        raw_response,
        request,
    )

    print(
        "    Respuesta válida."
    )

    # --------------------------------------------------------
    # 7. Mostrar resultado final
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESULTADO FINAL DE SC-04")
    print("=" * 72)

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 8. Verificaciones finales
    # --------------------------------------------------------

    assert (
        result.recognized_text
        == "AGRA"
    )

    assert (
        result.corrected_text
        == "AGUA"
    )

    assert (
        result.translated_text
        == "WATER"
    )

    assert (
        result.concepts
        == ("WATER",)
    )

    assert (
        result.unresolved
        == ()
    )

    print()
    print("=" * 72)
    print(
        "SC-04 completó correctamente "
        "la prueba end-to-end simulada."
    )
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
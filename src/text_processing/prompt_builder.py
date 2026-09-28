from __future__ import annotations

import json

from src.text_processing.contracts import (
    CorrectionRequest,
)
from src.text_processing.vocabulary import (
    Vocabulary,
)


class PromptBuilder:
    """
    Builds the restricted prompt used by SC-04.

    The prompt combines:

    - SC-03 recognized text
    - Top-k candidates and probabilities
    - source / target languages
    - closed canonical vocabulary
    - required JSON response schema
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
    ):
        self.vocabulary = vocabulary

    def build(
        self,
        request: CorrectionRequest,
    ) -> str:

        token_information = (
            self._format_tokens(
                request.tokens
            )
        )

        vocabulary_catalog = (
            self.vocabulary
            .get_prompt_catalog()
        )

        vocabulary_json = json.dumps(
            vocabulary_catalog,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are the linguistic correction and translation component
of the Sign2Sign prototype.

Your task is to interpret text obtained from fingerspelling
recognition and map its meaning to a CLOSED canonical
vocabulary.

IMPORTANT CONTEXT

The input was generated one letter at a time by a sign
recognition classifier.

For each recognized letter, Top-k alternatives and their
probabilities may be provided.

These probabilities are evidence from the recognition model.
They MUST be considered when correcting possible recognition
errors, but they are NOT hard acceptance thresholds.

SOURCE SIGN LANGUAGE:
{request.source_language}

TARGET SIGN LANGUAGE:
{request.target_language}

SOURCE BRIDGE LANGUAGE:
{request.source_bridge_language}

TARGET BRIDGE LANGUAGE:
{request.target_bridge_language}


RECOGNIZED TEXT

{request.recognized_text}


LETTER-LEVEL RECOGNITION INFORMATION

{token_information}


CANONICAL SIGN2SIGN VOCABULARY

The following list contains EVERY concept that the system is
allowed to represent.

{vocabulary_json}


VOCABULARY RULES

1. You may ONLY return concept_id values contained in the
   canonical vocabulary above.

2. NEVER invent a concept_id.

3. The vocabulary is semantic and canonical.
   Inflected or conjugated forms do not require independent
   concepts.

4. Normalize conjugated forms to their canonical concept.

   Example:
   QUIERO, QUIERES, QUEREMOS
   may correspond to WANT when the context supports it.

5. Negation is compositional.

   Example:
   NO QUIERO
   should use:
   ["NOT", "WANT"]

6. KNOW_INFORMATION represents knowing information or facts.

7. KNOW_PERSON represents knowing or being acquainted with
   a person.

8. MEET represents meeting a person for the first time.

9. HAVE represents possession.

10. EXIST represents existence, such as:
    HAY / THERE IS / THERE ARE.

11. FINISH represents the action of finishing.

12. END represents the concept of an end or conclusion.

13. Entries with type "expression" may be selected as a
    complete semantic unit when the input meaning matches the
    expression.

14. Use the Top-k letter candidates to resolve recognition
    mistakes when useful.

15. Do not automatically choose the Top-1 character if another
    candidate produces a substantially more plausible word
    within the available vocabulary.

16. Do not force a correction if the information is
    insufficient.

17. If part of the input cannot be represented using the
    closed vocabulary, include that part in "unresolved".

18. Do not represent unavailable meaning using an unrelated
    concept.


TASK

Perform the following operations:

A. Correct the recognized text in the SOURCE bridge language.

B. Translate the corrected meaning into natural text in the
   TARGET bridge language.

C. Determine the ordered canonical Sign2Sign concept IDs that
   represent the meaning.

D. Identify any portion that cannot be represented using the
   available vocabulary.


RESPONSE FORMAT

Return ONLY valid JSON.

Do not include Markdown.

Do not include explanations before or after the JSON.

Use exactly this structure:

{{
  "corrected_text": "string",
  "translated_text": "string",
  "concepts": [
    "CONCEPT_ID"
  ],
  "unresolved": [
    "string"
  ]
}}

The concepts array must contain only concept_id values from
the canonical vocabulary.

If everything can be represented, return:

"unresolved": []
""".strip()

    @staticmethod
    def _format_tokens(
        tokens: tuple[dict, ...],
    ) -> str:

        lines = []

        letter_position = 0

        for token in tokens:

            kind = token.get(
                "kind"
            )

            if kind == "space":

                lines.append(
                    "[WORD SEPARATOR]"
                )

                continue

            if kind != "letter":
                continue

            letter_position += 1

            value = token.get(
                "value",
                "",
            )

            top_k = token.get(
                "top_k",
                [],
            )

            candidates = []

            for candidate in top_k:

                label = candidate.get(
                    "label",
                    "",
                )

                probability = float(
                    candidate.get(
                        "probability",
                        0.0,
                    )
                )

                candidates.append(
                    f"{label}: "
                    f"{probability:.4f}"
                )

            candidate_text = (
                " | ".join(
                    candidates
                )
            )

            lines.append(
                f"Letter {letter_position}: "
                f"selected={value}; "
                f"candidates=[{candidate_text}]"
            )

        if not lines:
            return (
                "No letter-level information "
                "available."
            )

        return "\n".join(
            lines
        )
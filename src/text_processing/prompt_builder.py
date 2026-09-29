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
    - linguistic normalization rules
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
   QUIERO, QUIERES and QUEREMOS may all correspond to WANT,
   but the grammatical person expressed by the conjugation
   must also be represented when an appropriate canonical
   pronoun concept exists.

5. Recover semantic participants that are implicit in verb
   conjugation when the vocabulary can represent them.

   Examples:

   QUIERO AGUA
   -> FIRST_PERSON_SINGULAR, WANT, WATER

   NO ENTIENDO
   -> FIRST_PERSON_SINGULAR, NOT, UNDERSTAND

   QUIERES AGUA
   -> SECOND_PERSON, WANT, WATER

   The concepts array must not omit a representable subject
   merely because it was implicit in the source-language verb.

6. Negation is compositional.

   Example:

   NO QUIERO
   -> FIRST_PERSON_SINGULAR, NOT, WANT

7. KNOW_INFORMATION represents knowing information or facts.

8. KNOW_PERSON represents knowing or being acquainted with
   a person.

9. MEET represents meeting a person for the first time.

10. HAVE represents possession.

11. EXIST represents existence, such as:
    HAY / THERE IS / THERE ARE.

12. FINISH represents the action of finishing.

13. END represents the concept of an end or conclusion.

14. Entries with type "expression" may be selected as a
    complete semantic unit when that expression already
    represents the full intended meaning.

    Example:

    CÓMO ESTÁS
    -> HOW_ARE_YOU

    In this case it is not necessary to additionally emit
    SECOND_PERSON or HOW if HOW_ARE_YOU already preserves the
    complete intended meaning.

15. The concepts array is NOT a list of keywords.

    It is the ordered canonical semantic representation that
    the next Sign2Sign subsystem will consume.

16. Preserve every part of the meaning that can be represented
    by the available canonical vocabulary.

17. Prefer the smallest ordered set of canonical concepts that
    preserves the complete representable meaning.

18. Do not omit a canonical concept only because the natural
    translated sentence expresses that information
    grammatically rather than as an independent word.

19. Use the Top-k letter candidates to resolve recognition
    mistakes when useful.

20. Do not automatically choose the Top-1 character if another
    candidate produces a substantially more plausible word
    within the available vocabulary.

21. A correct interpretation may still be selected when the
    required letter is not present in Top-k, if the recognized
    text and closed vocabulary provide sufficiently strong
    linguistic evidence.

22. Do not force a correction if the information is
    insufficient.

23. If part of the input cannot be represented using the
    closed vocabulary, include that part in "unresolved".

24. Do not represent unavailable meaning using an unrelated
    concept.

25. If nothing can be interpreted reliably:

    - return an empty "corrected_text";
    - return an empty "translated_text";
    - return an empty "concepts" array;
    - place the unresolved input in "unresolved".


TASK

Perform the following operations:

A. Correct the recognized text in the SOURCE bridge language.

B. Translate the corrected meaning into natural text in the
   TARGET bridge language.

C. Determine the ordered canonical Sign2Sign concept IDs that
   preserve ALL meaning representable by the closed
   vocabulary.

D. Recover implicit grammatical participants, such as the
   subject encoded by verb conjugation, whenever an available
   canonical concept can represent them.

E. Prefer a single expression concept when it already
   represents the complete intended meaning.

F. Identify any portion that cannot be represented using the
   available vocabulary.


CONSISTENCY REQUIREMENT

The natural-language translation and the concepts array must
describe the same meaning.

For example, this is inconsistent:

translated_text:
"I DO NOT UNDERSTAND"

concepts:
["NOT", "UNDERSTAND"]

because the first-person subject is representable using
FIRST_PERSON_SINGULAR.

The consistent result is:

translated_text:
"I DO NOT UNDERSTAND"

concepts:
[
  "FIRST_PERSON_SINGULAR",
  "NOT",
  "UNDERSTAND"
]


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
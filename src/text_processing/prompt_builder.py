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
    complete semantic unit when that expression represents the
    full meaning of the SPECIFIC input segment from which it
    was derived.

    Example:

    CÓMO ESTÁS
    -> HOW_ARE_YOU

    In this case it is not necessary to additionally emit
    SECOND_PERSON or HOW because HOW_ARE_YOU already preserves
    the complete meaning of the segment "CÓMO ESTÁS".

    IMPORTANT:
    An expression concept may only replace the input segment
    whose meaning it represents.

    It MUST NOT absorb, replace, or cause the omission of an
    independent semantic unit that appears before or after that
    segment.

    Example:

    HOLA CÓMO ESTÁS
    -> HELLO, HOW_ARE_YOU

    NOT:

    HOLA CÓMO ESTÁS
    -> HOW_ARE_YOU

15. The concepts array is NOT a list of keywords.

    It is the ordered canonical semantic representation that
    the next Sign2Sign subsystem will consume.

16. Preserve every independent semantic unit that can be
    reasonably recovered from the recognized input.

    Correction may modify misspelled words, but it MUST NOT
    silently delete a word or semantic segment merely because
    the resulting sentence would sound more natural, concise,
    redundant, or conversational without it.

17. Every meaningful portion of the recognized input must be
    accounted for in the final result.

    Each meaningful portion must end in one of these states:

    a) corrected and represented by one or more canonical
       concepts; or

    b) included in "unresolved" when it cannot be interpreted
       or represented reliably.

    A meaningful input segment MUST NEVER disappear silently.

18. Prefer the smallest ordered set of canonical concepts that
    preserves the complete representable meaning, but ONLY
    after preserving all independent semantic units.

    "Smallest" means avoiding redundant concepts inside the
    same semantic unit.

    It does NOT mean removing a separate word or idea from the
    input.

19. Do not omit a canonical concept only because the natural
    translated sentence expresses that information
    grammatically rather than as an independent word.

20. Before producing the final JSON, perform a semantic
    coverage check over the recognized input.

    Verify that every meaningful word or recoverable segment
    has either:

    - contributed to corrected_text and to the canonical
      concepts; or

    - been included in unresolved.

    Do not treat a recoverable word as noise only because
    another nearby expression already forms a valid sentence.

    Example:

    Recognized input:
    HOLS CONO ESSAS

    A reasonable correction is:
    HOLA CÓMO ESTÁS

    Therefore the semantic coverage is:

    HOLS
    -> HOLA
    -> HELLO

    CONO ESSAS
    -> CÓMO ESTÁS
    -> HOW_ARE_YOU

    Expected concepts:
    [
      "HELLO",
      "HOW_ARE_YOU"
    ]

    "HELLO" must not be omitted simply because HOW_ARE_YOU is
    itself a complete conversational expression.

21. Before selecting canonical concepts, divide the corrected
    input into the smallest NON-OVERLAPPING semantic segments
    that can be mapped to the canonical vocabulary.

    If two adjacent segments independently map to different
    canonical concepts, preserve BOTH concepts, even when
    their meanings are pragmatically similar, redundant, or
    could be combined into a more natural sentence.

    An expression concept may consume multiple words only when
    those words belong to that canonical expression itself.

    It MUST NOT consume an adjacent word or phrase that
    independently maps to another canonical concept.

    Example:

    PERDÓN | CON PERMISO
    -> SORRY | EXCUSE_ME

    Expected concepts:
    [
      "SORRY",
      "EXCUSE_ME"
    ]

    NOT:
    [
      "EXCUSE_ME"
    ]

    General rule:

    If a corrected word or phrase has a direct mapping to an
    available canonical concept, that concept must not be
    omitted unless that exact word or phrase is part of a
    larger canonical expression selected from the vocabulary.

22. Use the Top-k letter candidates to resolve recognition
    mistakes when useful.

23. Do not automatically choose the Top-1 character if another
    candidate produces a substantially more plausible word
    within the available vocabulary.

24. A correct interpretation may still be selected when the
    required letter is not present in Top-k, if the recognized
    text and closed vocabulary provide sufficiently strong
    linguistic evidence.

25. Do not force a correction if the information is
    insufficient.

26. If part of the input cannot be represented using the
    closed vocabulary, include that part in "unresolved".

27. Do not represent unavailable meaning using an unrelated
    concept.

28. If nothing can be interpreted reliably:

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

E. Prefer a single expression concept when it represents the
   complete meaning of its corresponding input segment, but do
   not use an expression to absorb or remove independent
   semantic units from neighboring segments.

F. Identify any portion that cannot be represented using the
   available vocabulary.

G. Before returning the response, verify semantic coverage:
   every meaningful portion of the recognized input must
   either be represented in the corrected/translated meaning
   and concepts, or appear in unresolved.

H. Verify that canonical concepts were assigned from
   non-overlapping semantic segments.

   Do not merge neighboring segments merely because their
   meanings are similar or because the resulting translation
   would sound more natural.


CONSISTENCY REQUIREMENT

The natural-language translation and the concepts array must
describe the same meaning.

The corrected text must also preserve all reliably recoverable
meaning from the recognized input.

Do not improve fluency by deleting an independent recoverable
word or phrase.

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

A second consistency example:

corrected_text:
"PERDÓN CON PERMISO"

concepts:
[
  "EXCUSE_ME"
]

is inconsistent because the independently representable
segment "PERDÓN" was omitted.

The consistent result is:

corrected_text:
"PERDÓN CON PERMISO"

concepts:
[
  "SORRY",
  "EXCUSE_ME"
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
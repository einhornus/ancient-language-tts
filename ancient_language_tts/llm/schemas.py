"""LLM-typed shapes.

Every call is guided chain-of-thought. Reasoning fields are declared
first, the answer last. Constrained decoding emits keys in declaration
order, so each step is written before the next field opens. A reasoning
field's description says what to cover, in order, and that the step uses
the one before it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IpaTranscription(BaseModel):
    values: str = Field(
        description=(
            "Directed reasoning before any syllable or stress is written. "
            "Cover, in order: each word's consonants and vowels as the rules "
            "assign them; where a letter has two readings, the rule that "
            "picks one and the choice the examples make for that letter. "
            "No syllable dots and no stress marks in this field."
        )
    )
    syllables: str = Field(
        description=(
            "Directed reasoning from the consonants and vowels of the "
            "previous step, before any stress mark is placed. Cover, in "
            "order: how those values split into syllables by the language's "
            "syllable rules, the way the examples split them. Keep the "
            "lengths and qualities already chosen. No stress marks in this "
            "field."
        )
    )
    stress: str = Field(
        description=(
            "Directed reasoning from the syllables of the previous step, "
            "before the IPA line is written. Cover, in order: for each word, "
            "the syllable that takes ˈ and the fact about those syllables "
            "that puts it there, by the language's stress or accent rule. "
            "Do not write the passage as one IPA line."
        )
    )
    ipa: str = Field(
        description=(
            "The IPA line, written after that reasoning, from the stressed "
            "syllables of the previous step. Spaces between words. A dot "
            "between syllables. ˈ immediately before the primary-stress "
            "syllable, ˌ before a secondary stress. No slash brackets, no "
            "asterisks, no translation. A comma for a short pause, an "
            "exclamation mark, a question mark, and a period for a long "
            "pause at the end of a sentence. A semicolon or a colon in the "
            "passage is a period. No other punctuation."
        )
    )


class IpaScore(BaseModel):
    values: str = Field(
        description=(
            "Directed reasoning before any later criterion and before any "
            "score. Cover, in order: each word's consonants and vowels in "
            "this transcription; where a letter has two readings, whether "
            "the transcription takes the reading the rule picks and the "
            "choice the examples make; which segments the rules forbid. No "
            "score in this field."
        )
    )
    syllables: str = Field(
        description=(
            "Directed reasoning from the consonants and vowels of the "
            "previous step, before any later criterion and before any "
            "score. Cover, in order: where this transcription splits those "
            "values into syllables, and whether that split is the one the "
            "syllable rules and the examples require. No score in this field."
        )
    )
    stress: str = Field(
        description=(
            "Directed reasoning from the syllables of the previous step, "
            "before the line is judged and before any score. Cover, in "
            "order: for each word, the syllable this transcription marks "
            "with ˈ and the fact about those syllables that should put it "
            "there, by the language's stress or accent rule, and ˌ where "
            "the rules write one. No score in this field."
        )
    )
    line: str = Field(
        description=(
            "Directed reasoning from the stress of the previous step, "
            "before any score. Cover, in order: spaces, syllable dots, "
            "stress marks, lengths, and pause marks against the way the "
            "output rules and the examples write a line; whether the words "
            "match the text; anything present that is not the transcription. "
            "No score in this field."
        )
    )
    values_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How far the consonants and vowels follow the rules and the "
            "examples, 0-100, scored after the reasoning above. 100 means "
            "every segment is the one the rules assign. Below 30 means a "
            "segment the rules forbid."
        ),
    )
    syllables_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How far the syllable breaks follow the syllable rules and the "
            "examples, 0-100, scored after the values score. 100 means every "
            "break is the one the rules require. Below 30 means a break the "
            "rules forbid."
        ),
    )
    stress_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How far ˈ falls where the stress or accent rule puts it, "
            "0-100, scored after the syllable score. 100 means every word "
            "is marked on the syllable the rule selects. Below 30 means "
            "the stress is on the wrong syllable."
        ),
    )
    line_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How far the line is written the way the output rules and the "
            "examples write one, 0-100, scored after the stress score. 100 "
            "means the writing matches. Below 30 means a missing word, an "
            "added word, or something that is not a transcription."
        ),
    )
    verdict: str = Field(
        description=(
            "One sentence, written after the four scores, on whether this "
            "transcription should be accepted. No number in this field."
        )
    )


class IpaRevision(BaseModel):
    faults: str = Field(
        description=(
            "Directed reasoning before any corrected value is chosen. "
            "Cover, in order: each mistake the assessment names; whether "
            "the phonetic rules agree that it is a mistake; which parts of "
            "the transcription the assessment leaves standing. No corrected "
            "IPA in this field."
        )
    )
    values: str = Field(
        description=(
            "Directed reasoning from the faults of the previous step, "
            "before any syllable or stress is written. Cover, in order: "
            "each word's consonants and vowels as the rules assign them, "
            "with each confirmed fault corrected and each sound the "
            "assessment leaves standing kept; where a letter has two "
            "readings, the rule that picks one and the choice the examples "
            "make for that letter. No syllable dots and no stress marks in "
            "this field."
        )
    )
    syllables: str = Field(
        description=(
            "Directed reasoning from the consonants and vowels of the "
            "previous step, before any stress mark is placed. Cover, in "
            "order: how those values split into syllables by the language's "
            "syllable rules, the way the examples split them. Keep the "
            "lengths and qualities already chosen. No stress marks in this "
            "field."
        )
    )
    stress: str = Field(
        description=(
            "Directed reasoning from the syllables of the previous step, "
            "before the IPA line is written. Cover, in order: for each word, "
            "the syllable that takes ˈ and the fact about those syllables "
            "that puts it there, by the language's stress or accent rule. "
            "Do not write the passage as one IPA line."
        )
    )
    ipa: str = Field(
        description=(
            "The corrected IPA line, written after that reasoning, from the "
            "stressed syllables of the previous step. Spaces between words. "
            "A dot between syllables. ˈ immediately before the primary-stress "
            "syllable, ˌ before a secondary stress. No slash brackets, no "
            "asterisks, no translation. A comma for a short pause, an "
            "exclamation mark, a question mark, and a period for a long "
            "pause at the end of a sentence. A semicolon or a colon in the "
            "passage is a period. No other punctuation."
        )
    )


class RespelledText(BaseModel):
    rules: str = Field(
        description=(
            "Directed reasoning before any word is spelled out. Apply the "
            "spelling rules and the pair rules to this IPA as a whole: "
            "which spelling each sound takes, where stress is marked, and "
            "which letters the rules allow. No line in this field."
        )
    )
    words: str = Field(
        description=(
            "Directed reasoning from the rules of the previous step, before "
            "the line is written. Name the words those rules are hardest to "
            "apply to, and for each one the spelling the rules require. "
            "No line in this field."
        )
    )
    text: str = Field(
        description=(
            "The line, written after that reasoning, in the stated shape. "
            "Ordinary text in the target spelling: each word is one "
            "unbroken word, a space separates words, stress is an accent on "
            "the stressed vowel, and letters are lowercase except the first "
            "letter of a sentence. No hyphen inside a word. No capital "
            "letters used to mark a stressed syllable. No IPA, no notes, "
            "no markdown."
        )
    )


class RespellScore(BaseModel):
    phonetics: str = Field(
        description=(
            "Directed reasoning before any later criterion and before any "
            "score. Cover, in order: for each sound, the sound a modern "
            "voice would make from this spelling, read by the spelling "
            "rules, and the sound the IPA has, including the substitute "
            "the pair rules name when the target language has no letter "
            "for that sound. No score in this field."
        )
    )
    stress: str = Field(
        description=(
            "Directed reasoning from the sounds of the previous step, "
            "before any later criterion and before any score. Cover, in "
            "order: for each word, the syllable this line marks as "
            "stressed and the syllable the IPA stress mark selects. "
            "No score in this field."
        )
    )
    format: str = Field(
        description=(
            "Directed reasoning from the stress of the previous step, "
            "before any score. Cover, in order: each word unbroken, spaces "
            "between words, an accent on the stressed vowel, lowercase "
            "letters, a capital only at the start of a sentence, and pause "
            "marks; whether the words match the IPA; anything present that "
            "is not the line. No score in this field."
        )
    )
    phonetics_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How close the sounds a modern voice would make from this "
            "line are to the IPA, 0-100, scored after the reasoning above. "
            "100 means every sound matches, including a substitute the "
            "pair rules name. Below 30 means the voice would make a sound "
            "the IPA does not have, or would miss a sound the IPA has."
        ),
    )
    stress_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How far stress falls on the syllable the IPA mark selects, "
            "0-100, scored after the phonetics score. 100 means every "
            "word is marked on the right syllable. Below 30 means the "
            "stress is on the wrong syllable."
        ),
    )
    format_score: int = Field(
        ge=0,
        le=100,
        description=(
            "How closely the line follows the stated shape, 0-100, scored "
            "after the stress score. 100 means the writing matches. Below "
            "30 means a hyphen inside a word, a capital used to mark "
            "stress, a missing word, an added word, leftover IPA, or "
            "something that is not the line."
        ),
    )
    verdict: str = Field(
        description=(
            "One sentence, written after the three scores, on whether this "
            "line should be accepted. No number in this field."
        )
    )


class RespellRevision(BaseModel):
    phonetics: str = Field(
        description=(
            "Directed reasoning before any corrected stress is chosen. "
            "Cover, in order: each phonetic mistake the assessment names; "
            "whether the spelling rules and the pair rules agree that it "
            "is a mistake; for each sound, the spelling that makes the "
            "voice produce the IPA, including the substitute the pair "
            "rules name when the target language has no letter for that "
            "sound; which sounds the assessment leaves standing. Where a "
            "vowel's length or quality depends on stress, use the syllable "
            "the IPA mark selects. No corrected line in this field."
        )
    )
    stress: str = Field(
        description=(
            "Directed reasoning from the spellings of the previous step, "
            "before the line's shape is settled. Cover, in order: for "
            "each word, the syllable the IPA stress mark selects and the "
            "accent the spelling rules place on that vowel, with each "
            "confirmed fault corrected and each stress the assessment "
            "leaves standing kept. No line in this field."
        )
    )
    format: str = Field(
        description=(
            "Directed reasoning from the stress of the previous step, "
            "before the line is written. Cover, in order: each word "
            "unbroken, spaces between words, the accent on the stressed "
            "vowel, lowercase letters, a capital only at the start of a "
            "sentence, and the pause marks the IPA uses, with each "
            "confirmed format fault corrected. No line in this field."
        )
    )
    text: str = Field(
        description=(
            "The corrected line, written after that reasoning, from the "
            "format of the previous step. Ordinary text in the target "
            "spelling: each word is one unbroken word, a space separates "
            "words, stress is an accent on the stressed vowel, and letters "
            "are lowercase except the first letter of a sentence. No "
            "hyphen inside a word. No capital letters used to mark a "
            "stressed syllable. No IPA, no notes, no markdown."
        )
    )

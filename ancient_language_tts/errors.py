"""Errors the pipeline raises. A missing key, a failed model chain, and a
failed ElevenLabs call are all `TTSError`, so a caller can catch one type.
"""


class TTSError(RuntimeError):
    """A transcription, respelling, or ElevenLabs call could not finish."""

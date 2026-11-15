"""Compatibility shim: LLM IPA lives in `ancient_language_tts.llm.ipa`."""

from .llm.ipa import *  # noqa: F403
from .llm.ipa import (  # noqa: F401
    improve_ipa,
    improve_system_prompt,
    improve_user_prompt,
    score_ipa,
    score_system_prompt,
    score_user_prompt,
    system_prompt,
    to_ipa,
    user_prompt,
)

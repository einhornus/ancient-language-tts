"""One helper for every typed LLM call: Pydantic AI over OpenRouter.

`ask()` sends a system prompt and a user prompt, and returns an instance
of the Pydantic model you asked for. The model walks the chain from
`Settings.llm_models(tier)`: if one model fails (content filter, provider
error, unexpected shape), the next is tried. `ask()` raises `TTSError`
when every model failed. There is no spoken fallback: a guessed
pronunciation would be read aloud as if it were the text.

Identical calls (same model, prompts, and output schema) are served from
a local JSON cache under `<cache_dir>/llm/`.

Prompting is guided chain-of-thought (see the README). The system prompt
closes with the agenda: reason about these things, in order, then write
the result. Every output type declares those reasoning fields first and
the answer last. Each reasoning field's description says what to cover,
in that order, and that the step uses the one before it. Provider
thinking tokens are not relied on.

Set the `ancient_language_tts.llm` logger to DEBUG to see every prompt
and answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from typing import Any, TypeVar

from pydantic import BaseModel

from .. import config
from ..config import Settings, Tier
from ..errors import TTSError

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger("ancient_language_tts.llm")


def _cache_key(model: str, system: str, user: str, output_type: type[BaseModel]) -> str:
    """Same model + same prompts + same expected shape -> same entry."""
    request = {
        "model": model,
        "system": system,
        "user": user,
        "schema": output_type.model_json_schema(),
    }
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_load(path: Any, output_type: type[T]) -> T | None:
    if not path.is_file():
        return None
    try:
        return output_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _cache_save(path: Any, result: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _agent(model: str, output_type: type[BaseModel], system: str, settings: Settings) -> Any:
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        app_url=config.APP_URL,
        app_title=config.APP_TITLE,
    )
    return Agent(
        OpenRouterModel(model, provider=provider),
        output_type=output_type,
        system_prompt=system,
        model_settings={"max_tokens": settings.max_tokens},
    )


def _run(agent: Any, prompt: str) -> Any:
    """`agent.run()` to completion, with its HTTP client closed afterwards.

    Inside a running event loop (a notebook), the call runs on a worker
    thread with its own loop, so `speak()` stays a plain function.
    """

    async def go() -> Any:
        async with agent:
            return await agent.run(prompt)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(go())
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["result"] = asyncio.run(go())
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _log_call(
    kind: str,
    model: str,
    output_type: type[BaseModel],
    system: str,
    user: str,
    *,
    output: BaseModel | None = None,
    error: BaseException | None = None,
) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return
    lines = [
        "=" * 72,
        f"LLM {kind}  model={model}  -> {output_type.__name__}",
        "-- system --",
        system,
        "-- user --",
        user,
    ]
    if error is not None:
        lines += ["-- error --", f"{type(error).__name__}: {error}"]
    elif output is not None:
        lines += ["-- answer --", output.model_dump_json(indent=2)]
    lines.append("=" * 72)
    log.debug("\n".join(lines))


def require_key() -> None:
    """Raise when this run has no OpenRouter key."""
    if not config.current().openrouter_api_key:
        raise TTSError("No OpenRouter key. Pass openrouter_api_key= or set OPENROUTER_API_KEY.")


def ask(output_type: type[T], system: str, user: str, tier: Tier = "high") -> T:
    """Ask the LLM for an `output_type` object.

    Raises `TTSError` when no OpenRouter key is set or every model failed.
    """
    require_key()
    settings = config.current()
    names = settings.llm_models(tier)
    if not names:
        raise TTSError("No LLM models configured.")
    last_error: BaseException | None = None
    for name in names:
        cache_path = settings.llm_cache_dir / f"{_cache_key(name, system, user, output_type)}.json"
        if settings.cache:
            cached = _cache_load(cache_path, output_type)
            if cached is not None:
                _log_call("CACHE", name, output_type, system, user, output=cached)
                log.info("LLM %s cache hit -> %s", name, output_type.__name__)
                return cached
        try:
            agent = _agent(name, output_type, system, settings)
            output = _run(agent, user).output
        except Exception as exc:
            last_error = exc
            _log_call("CALL", name, output_type, system, user, error=exc)
            log.info("LLM %s failed: %s: %s", name, type(exc).__name__, exc)
            continue
        _log_call("CALL", name, output_type, system, user, output=output)
        log.info("LLM %s answered %s", name, output_type.__name__)
        if settings.cache:
            _cache_save(cache_path, output)
        return output
    raise TTSError(f"Every model failed ({', '.join(names)}). Last error: {last_error}")

"""The backbone of the per-language tests (tests/test_<code>.py).

Each test names an (ancient, modern) pair and its constants, then calls
`run`. For that pair, `run`:

1. loads the first `sentences` lines of data/<source>/texts.json,
2. transcribes each one to IPA and respells it with the LLM pipeline
   (`tier`, `effort`),
3. speaks each respelling with the first `voices_per_gender` male and the
   first `voices_per_gender` female voices of voices/<target>.json (the
   language's built-in voice when there is no voices file), and hears each
   take back with the local phoneme recognizer (ancient_language_tts.check),
4. scores each take against the expected IPA and reports the metrics:
   overall, by voice, every take ranked, and the mistakes behind them.

The metric is the mismatch of check.mismatch: the weighted phoneme edit
distance between the expected and the heard IPA, divided by the number of
expected phonemes. 0 is a perfect match. `pooled` divides the summed
distance by the summed expected phonemes (long sentences weigh more);
`mean`, `median` and `stdev` are over takes; for one voice that is one
take per sentence, so its `stdev` is the spread of its score between
sentences. A mistake's `contribution` is its share
of the pooled mismatch, so the contributions sum to it.

The report is written to out/tests/<source>.<target>.json, and a summary is
printed. Its `ranking` lists every take, worst first: the sentence, the
voice, the scores, and the audio file.

Every take is also saved in out/tests/<source>.<target>/ as
<sentence>.<voice>.mp3 (e.g. 003.Darren.mp3), next to index.html: the same
ranking as a page with a player in each row. Open it in a browser to hear
the worst takes first. That folder is emptied at the start of each run of
the pair, so it only ever holds the latest run.

LLM answers and audio are cached (config.cache), so a rerun after
editing data/<source>/respelling/<target>.txt pays only for what changed.
"""

from __future__ import annotations

import html
import json
import logging
import re
import shutil
import statistics
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import ancient_language_tts as alt
from ancient_language_tts import config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ancient_language_tts" / "data"
VOICES = ROOT / "voices"
OUT = ROOT / "out" / "tests"

# LLM calls run this many at a time. Recognition runs one at a time.
WORKERS = 8

# ElevenLabs calls run this many at a time. Keep it at or below your plan's
# concurrency limit (5 on yours); above it, ElevenLabs answers 429.
TTS_WORKERS = 4

# A take rejected with 429 (rate or concurrency limit) is retried this many
# times, waiting TTS_RETRY_SECONDS, then twice that, and so on.
TTS_RETRIES = 4
TTS_RETRY_SECONDS = 5

# ElevenLabs seed for every take, so a rerun hears the same take (and hits the cache).
SEED = 0

# A silence at least this long between two recognized phonemes is a space in the heard IPA.
PAUSE_MS = 300

# How much the printed summary lists (the JSON keeps everything). Every
# mistake is printed; only the sentences and each mistake's words are cut.
TOP_SENTENCES = 10
TOP_WORDS = 3

GENDERS = ("male", "female")

log = logging.getLogger("tests.harness")


def run(
    source: str,
    target: str,
    *,
    sentences: int,
    voices_per_gender: int,
    tier: str,
    effort: str,
    max_mismatch: float | None = None,
) -> dict:
    """Run the pair end to end, write and print its report, and return it."""
    _logging()
    texts = load_texts(source, sentences)
    chosen = pick_voices(target, voices_per_gender)
    started = time.time()

    folder = _fresh_folder(source, target)
    _file_names(chosen)

    log.info("%s → %s: %s sentences, %s voices, tier %s, effort %s",
             source, target, len(texts), len(chosen), tier, effort)
    rows = _parallel(lambda item: _transcribe(item, source, target, tier, effort), enumerate(texts))

    jobs = [(row, voice) for row in rows if row["respelling"] for voice in chosen]
    audio = _parallel(lambda job: _speak(job[0], job[1], target), jobs, TTS_WORKERS)

    for (row, voice), data in zip(jobs, audio):
        name = voice["name"]
        if isinstance(data, str):
            row["takes"][name] = {"error": data}
            continue
        audio_file = f"{row['index'] + 1:03d}.{voice['file_name']}.mp3"
        (folder / audio_file).write_bytes(data)
        try:
            row["takes"][name] = _hear(data, source, row["expected"])
        except Exception as exc:  # one bad take should not sink the run
            row["takes"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        row["takes"][name]["audio"] = f"{folder.name}/{audio_file}"
    for row in rows:
        row.pop("expected", None)

    report = {
        "settings": _settings(source, target, len(texts), chosen, tier, effort),
        "overall": _metrics(rows, None),
        "by_voice": [{**voice, **_metrics(rows, voice["name"])} for voice in chosen],
        "ranking": _ranking(rows, chosen),
        "worst_sentences": _worst(rows),
        "sentences": rows,
        "seconds": round(time.time() - started, 1),
    }
    path = _write(source, target, report)
    page = _write_page(folder, report)
    print_report(report)
    print(f"\nreport: {path}")
    print(f"listen: {page}")

    overall = report["overall"]
    assert overall["takes"], f"{source} → {target}: no take was heard; see {path}"
    if max_mismatch is not None:
        assert overall["mismatch_pooled"] <= max_mismatch, (
            f"{source} → {target}: pooled mismatch {overall['mismatch_pooled']:.3f} "
            f"> {max_mismatch}"
        )
    return report


# ---------------------------------------------------------------- inputs


def load_texts(source: str, count: int) -> list[str]:
    """The first `count` sentences of data/<source>/texts.json (a JSON list of strings)."""
    path = DATA / source / "texts.json"
    if not path.is_file():
        pytest.skip(f"no test sentences for {source}: add data/{source}/texts.json")
    texts = [t.strip() for t in json.loads(path.read_text(encoding="utf-8")) if t.strip()]
    if not texts:
        pytest.skip(f"data/{source}/texts.json is empty")
    return texts[:count]


def pick_voices(target: str, per_gender: int) -> list[dict]:
    """The first `per_gender` males, then the first `per_gender` females, of voices/<target>.json.

    With no voices file, the one built-in voice of `target` (id None).
    """
    path = VOICES / f"{target}.json"
    if not path.is_file():
        return [{"name": "built-in", "id": None, "gender": None}]
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")  # some library exports are not UTF-8
    listed = json.loads(text)
    chosen = []
    for wanted in GENDERS:
        matching = [v for v in listed if _gender(v) == wanted][:per_gender]
        chosen += [{"name": v["name"].strip(), "id": v["id"], "gender": wanted} for v in matching]
    return chosen


def _gender(voice: dict) -> str | None:
    if voice.get("gender") in GENDERS:
        return voice["gender"]
    return next((g for g in GENDERS if g in (voice.get("tags") or [])), None)


# ---------------------------------------------------------------- pipeline


def _transcribe(item: tuple[int, str], source: str, target: str, tier: str, effort: str) -> dict:
    index, text = item
    row: dict[str, Any] = {"index": index, "text": text, "ipa": None, "respelling": None, "takes": {}}
    try:
        pieces = alt.to_ipa(text, source, method="llm", tier=tier, effort=effort)
        row["expected"] = alt.chunks.join_ipa(pieces)
        row["ipa"] = alt.ipa_tokens.render(row["expected"])
        row["respelling"] = alt.respell_ipa(pieces, source, target, method="llm", tier=tier, effort=effort)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("sentence %s failed: %s", index, row["error"])
    return row


def _speak(row: dict, voice: dict, target: str) -> bytes | str:
    """The audio of one take, or the error as a string. A 429 is retried with backoff."""
    for attempt in range(TTS_RETRIES + 1):
        try:
            return alt.speak(row["respelling"], target, voice=voice["id"], seed=SEED)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if "(429)" not in error or attempt == TTS_RETRIES:
                return error
            wait = TTS_RETRY_SECONDS * 2 ** attempt
            log.info("  429 from ElevenLabs on %s, retrying in %ss", voice["name"], wait)
            time.sleep(wait)
    return error


def _hear(audio: bytes, source: str, expected: list[str]) -> dict:
    from ancient_language_tts import check

    phonemes = check.recognize_phonemes(audio, source=source)
    tokens = check.tokens_from_phonemes(phonemes, pause_ms=PAUSE_MS)
    comparison = check.compare(expected, tokens)
    return {
        "heard": alt.ipa_tokens.render(tokens),
        "mismatch": comparison.mismatch,
        "distance": comparison.distance,
        "expected_phonemes": len(comparison.expected),
        "heard_phonemes": len(comparison.heard),
        "confidence": (
            sum(p["confidence"] for p in phonemes) / len(phonemes) if phonemes else None
        ),
        "mistakes": [m.to_dict() for m in comparison.breakdown()],
    }


def _parallel(work: Callable, items: Iterable, workers: int = WORKERS) -> list:
    items = list(items)
    done = 0
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(work, items):
            results.append(result)
            done += 1
            if done % 10 == 0 or done == len(items):
                log.info("  %s/%s", done, len(items))
    return results


# ---------------------------------------------------------------- metrics


def _metrics(rows: list[dict], voice: str | None) -> dict:
    """The metrics of every heard take (`voice` None) or of one voice's takes."""
    takes = [
        take
        for row in rows
        for name, take in row["takes"].items()
        if (voice is None or name == voice) and "mismatch" in take
    ]
    failed_takes = sum(
        1
        for row in rows
        for name, take in row["takes"].items()
        if (voice is None or name == voice) and "error" in take
    )
    metrics: dict[str, Any] = {
        "takes": len(takes),
        "failed_takes": failed_takes,
    }
    if voice is None:
        metrics["sentences"] = len(rows)
        metrics["failed_sentences"] = sum(1 for row in rows if "error" in row)
    if not takes:
        return metrics

    scores = [t["mismatch"] for t in takes]
    expected = sum(t["expected_phonemes"] for t in takes)
    confidences = [t["confidence"] for t in takes if t["confidence"] is not None]
    metrics.update({
        "mismatch_pooled": sum(t["distance"] for t in takes) / expected if expected else 0.0,
        "mismatch_mean": statistics.fmean(scores),
        "mismatch_median": statistics.median(scores),
        "mismatch_stdev": statistics.pstdev(scores),
        "mismatch_worst": max(scores),
        "perfect_takes": sum(1 for s in scores if s == 0),
        "length_ratio": sum(t["heard_phonemes"] for t in takes) / expected if expected else None,
        "confidence": statistics.fmean(confidences) if confidences else None,
        "mistakes": _pool_mistakes(takes, expected),
    })
    return metrics


def _pool_mistakes(takes: list[dict], expected: int) -> list[dict]:
    """The mistakes of all `takes` together; contributions sum to the pooled mismatch."""
    pooled: dict[tuple, dict] = {}
    for take in takes:
        for m in take["mistakes"]:
            entry = pooled.setdefault(
                (m["expected"], m["heard"]),
                {"expected": m["expected"], "heard": m["heard"], "penalty": m["penalty"],
                 "times": 0, "words": {}},
            )
            entry["times"] += m["times"]
            for word, times in m["words"]:
                entry["words"][word] = entry["words"].get(word, 0) + times
    mistakes = []
    for entry in pooled.values():
        entry["contribution"] = entry["penalty"] * entry["times"] / expected if expected else 0.0
        entry["words"] = sorted(entry["words"].items(), key=lambda item: (-item[1], item[0]))
        mistakes.append(entry)
    return sorted(mistakes, key=lambda m: (-m["contribution"], -m["times"]))


def _worst(rows: list[dict]) -> list[dict]:
    """Sentences by their mismatch averaged over voices, worst first."""
    ranked = []
    for row in rows:
        scores = [t["mismatch"] for t in row["takes"].values() if "mismatch" in t]
        if scores:
            ranked.append({
                "index": row["index"],
                "mismatch": statistics.fmean(scores),
                "text": row["text"],
                "ipa": row["ipa"],
                "respelling": row["respelling"],
            })
    return sorted(ranked, key=lambda r: -r["mismatch"])


def _ranking(rows: list[dict], voices: list[dict]) -> list[dict]:
    """Every heard take, worst first: one entry per (sentence, voice)."""
    gender = {v["name"]: v["gender"] for v in voices}
    takes = []
    for row in rows:
        for name, take in row["takes"].items():
            if "mismatch" not in take:
                continue
            takes.append({
                "mismatch": take["mismatch"],
                "sentence": row["index"] + 1,
                "voice": name,
                "gender": gender.get(name),
                "audio": take.get("audio"),
                "text": row["text"],
                "respelling": row["respelling"],
                "ipa": row["ipa"],
                "heard": take["heard"],
            })
    takes.sort(key=lambda t: (-t["mismatch"], t["sentence"], t["voice"]))
    return [{"rank": rank, **take} for rank, take in enumerate(takes, start=1)]


# ---------------------------------------------------------------- audio folder


def _fresh_folder(source: str, target: str) -> Path:
    """out/tests/<source>.<target>/, emptied: it holds only this run's audio and page."""
    folder = OUT / f"{source}.{target}"
    if folder.resolve().parent != OUT.resolve():  # never delete outside out/tests
        raise RuntimeError(f"refusing to clean {folder}")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def _file_names(voices: list[dict]) -> None:
    """Give each voice a short, unique, file-safe `file_name`: "Darren", "Laura", ...

    Uses the name before " - " (ElevenLabs names are "Darren - Calm and Deep");
    a clash gets the gender and then a number appended.
    """
    used: set[str] = set()
    for voice in voices:
        short = voice["name"].split(" - ")[0]
        base = re.sub(r"[^\w-]+", "_", short).strip("_") or "voice"
        name = base
        if name in used and voice["gender"]:
            name = f"{base}_{voice['gender']}"
        number = 2
        while name in used:
            name = f"{base}_{number}"
            number += 1
        used.add(name)
        voice["file_name"] = name


def _write_page(folder: Path, report: dict) -> Path:
    """index.html: the ranking as a table, each row with a player for its take."""
    s, o = report["settings"], report["overall"]
    e = html.escape
    rows = []
    for t in report["ranking"]:
        audio = Path(t["audio"]).name if t["audio"] else ""
        player = f'<audio controls preload="none" src="{e(audio)}"></audio>' if audio else "–"
        rows.append(
            f'<tr data-voice="{e(t["voice"])}">'
            f'<td class="n">{t["rank"]}</td>'
            f'<td class="n">{t["mismatch"]:.3f}</td>'
            f'<td class="n">#{t["sentence"]}</td>'
            f'<td>{e(t["voice"])}<br><small>{e(t["gender"] or "")}</small></td>'
            f'<td>{player}<br><small>{e(audio)}</small></td>'
            f'<td><div>{e(t["text"])}</div>'
            f'<div class="k">respelling</div><div>{e(t["respelling"] or "")}</div>'
            f'<div class="k">expected</div><div class="ipa">{e(t["ipa"] or "")}</div>'
            f'<div class="k">heard</div><div class="ipa">{e(t["heard"])}</div></td>'
            "</tr>"
        )
    options = "".join(
        f'<option>{e(v["name"])}</option>' for v in s["voices"]
    )
    pooled = f'{o["mismatch_pooled"]:.3f}' if o.get("takes") else "–"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{e(s["source"])} → {e(s["target"])}: takes ranked</title>
<style>
  body {{ font: 14px/1.4 system-ui, sans-serif; margin: 1.5rem; color: #222; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 .3rem; }}
  .meta {{ color: #666; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: .5rem; vertical-align: top; text-align: left; }}
  th {{ position: sticky; top: 0; background: #f6f6f6; }}
  td.n {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .k {{ color: #888; font-size: 11px; text-transform: uppercase; margin-top: .4rem; }}
  .ipa {{ font-family: "Charis SIL", "Doulos SIL", serif; font-size: 15px; }}
  small {{ color: #888; }}
  audio {{ width: 220px; }}
</style></head><body>
<h1>{e(s["source"])} → {e(s["target"])}: every take, worst first</h1>
<div class="meta">tier {e(s["tier"])}, effort {e(s["effort"])} ·
  {o.get("takes", 0)} takes · pooled mismatch {pooled} (0 is a perfect match) ·
  voice <select id="voice"><option value="">all</option>{options}</select></div>
<table><thead><tr><th>rank</th><th>mismatch</th><th>sentence</th><th>voice</th>
<th>audio</th><th>text</th></tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table>
<script>
  // Show one voice only.
  document.getElementById("voice").onchange = (ev) => {{
    for (const row of document.querySelectorAll("tbody tr"))
      row.hidden = ev.target.value !== "" && row.dataset.voice !== ev.target.value;
  }};
  // One take plays at a time.
  document.addEventListener("play", (ev) => {{
    for (const a of document.querySelectorAll("audio")) if (a !== ev.target) a.pause();
  }}, true);
</script>
</body></html>
"""
    path = folder / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


# ---------------------------------------------------------------- output


def _settings(source: str, target: str, count: int, voices: list[dict], tier: str, effort: str) -> dict:
    settings = config.current()
    return {
        "source": source,
        "target": target,
        "sentences": count,
        "voices": voices,
        "tier": tier,
        "effort": effort,
        "llm_models": settings.llm_models(tier),
        "max_tokens": settings.max_tokens,
        "max_chars": settings.max_chars,
        "cache": settings.cache,
        "tts": {
            "elevenlabs_model": settings.elevenlabs_model,
            "output_format": settings.output_format,
            "stability": settings.stability,
            "similarity_boost": settings.similarity_boost,
            "seed": SEED,
        },
        "check": {"pause_ms": PAUSE_MS},
    }


def _write(source: str, target: str, report: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}.{target}.json"
    path.write_text(json.dumps(_rounded(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


def print_report(report: dict) -> None:
    s, o = report["settings"], report["overall"]
    print(f"\n=== {s['source']} → {s['target']}  (tier {s['tier']}, effort {s['effort']}) ===")
    print(f"sentences {o['sentences']} (failed {o['failed_sentences']}), "
          f"takes {o['takes']} (failed {o['failed_takes']})")
    if not o["takes"]:
        return

    print("\noverall")
    _print_scores("all voices", o)

    print("\nby voice")
    for v in sorted(report["by_voice"], key=lambda v: v.get("mismatch_pooled", float("inf"))):
        label = f"{v['name'][:30]} ({v['gender'] or '?'})"
        if v["takes"]:
            _print_scores(label, v)
        else:
            print(f"  {label:<40} no takes (failed {v['failed_takes']})")

    ranking = report["ranking"]
    print(f"\nworst takes ({min(TOP_SENTENCES, len(ranking))} of {len(ranking)}; "
          f"the report's `ranking` and index.html have all)")
    for t in ranking[:TOP_SENTENCES]:
        audio = Path(t["audio"]).name if t["audio"] else "–"
        print(f"  {t['rank']:>3}. {t['mismatch']:.3f}  #{t['sentence']:<3} {t['voice'][:30]}  [{audio}]")
        print(f"        text   {t['text']}")
        print(f"        resp   {t['respelling']}")
        print(f"        ipa    {t['ipa']}")
        print(f"        heard  {t['heard']}")

    print(f"\nall mistakes, {len(o['mistakes'])} kinds "
          f"(share of pooled mismatch {o['mismatch_pooled']:.3f})")
    for m in o["mistakes"]:
        pair = f"{m['expected'] or '∅'} → {m['heard'] or '∅'}"
        words = ", ".join(f"{w}×{n}" for w, n in m["words"][:TOP_WORDS])
        print(f"  {pair:<12} {m['penalty']:.2f} × {m['times']:<4} = {m['contribution']:.4f}   {words}")


def _print_scores(label: str, m: dict) -> None:
    confidence = f"{m['confidence']:.2f}" if m.get("confidence") is not None else "–"
    print(
        f"  {label:<40} pooled {m['mismatch_pooled']:.3f}  mean {m['mismatch_mean']:.3f}  "
        f"median {m['mismatch_median']:.3f}  std {m['mismatch_stdev']:.3f}  "
        f"worst {m['mismatch_worst']:.3f}  "
        f"len {m['length_ratio']:.2f}  conf {confidence}  takes {m['takes']}"
    )


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "pydantic_ai", "ancient_language_tts"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

# Ancient Language TTS

Text-to-speech for ancient languages. You give it a passage in one of the languages below. It gives back audio of a modern voice reading that passage.

It's BYOK. An [OpenRouter](https://openrouter.ai) key pays for the LLM calls, made through [Pydantic AI](https://ai.pydantic.dev). An [ElevenLabs](https://elevenlabs.io) key pays for the speech.

```python
import ancient_language_tts as alt

text = "Arma virumque canō, Troiae quī prīmus ab ōrīs."
ipa = alt.to_ipa(text, "lat", method="llm", tier="low")       # ancient text → IPA
line = alt.respell_ipa(ipa, "lat", "it", method="llm", text=text)  # IPA → Italian spelling
audio = alt.speak(line, "it", output="out/lat.mp3")             # Italian voice reads it
```

Keys come from `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY` (environment or `.env`), or pass `openrouter_api_key=` / `elevenlabs_api_key=` to any call.

## Install

Python 3.11+.

In another project, install it like any other package:

```bash
pip install ancient-language-tts                                # from PyPI
pip install git+https://github.com/einhornus/ancient-language-tts
pip install G:/path/to/ancient-language-tts                     # from a local checkout
```

To work on it, install it in editable mode, so your edits take effect without reinstalling:

```bash
git clone https://github.com/einhornus/ancient-language-tts
cd ancient-language-tts
pip install -e .
python examples/quickstart.py
```

You can also run it without installing the script, from this folder: `python -m ancient_language_tts`.

## The pipeline

There is no single "do everything" call. Each step is its own function, and `method` picks how it runs:

| Step | Function | `method="llm"` | `method="rules"` |
|---|---|---|---|
| ancient text → IPA | `to_ipa(text, source, method)` | the tier transcribes (step 1 below) | the language's rule engine; does not depend on a modern language |
| ancient text → modern spelling | `respell(text, source, modern_lang, method)` | `to_ipa` then `respell_ipa`, piece by piece | direct spelling rules, no IPA step |
| IPA → modern spelling | `respell_ipa(ipa, source, modern_lang, method)` | the tier respells (step 2 below) | the rule engine's IPA table |
| modern spelling → audio | `speak(respelled_text, modern_lang, voice)` | ElevenLabs (step 3 below) | — |

The LLM pipeline is the same for every language: text → IPA → respelling. Every ancient language also has a rule engine (`ancient_language_tts.rule_based`) with the three rule-based conversions. Only Ecclesiastical Latin → Italian (`lat-ecc` → `it`) has real rules today. The others are **mocks**: small grapheme tables with no stress or syllables, there so the API is uniform and so they can be replaced one at a time. `rule_based.is_mock(source)` tells them apart, and `ancient-language-tts languages` lists them.

## How the LLM steps work

The LLM calls use guided chain-of-thought, described in the next section. The caller receives the answer. The reasoning stays inside the call.

A passage longer than 1000 characters is split before those calls. The limit is `max_chars` (`ANCIENT_TTS_MAX_CHARS`). A sentence ends at `. `, `? `, or `! `. The cut is the sentence boundary closest to the middle, and both sides are split the same way until each piece fits. One sentence is left whole. Each piece is transcribed and then respelled on its own. The pieces are joined, and the voice reads that one line.

1. **IPA.** Every model in the tier transcribes the passage on its own. The system prompt is the phonetic rules of that language (`ancient_language_tts/data/<code>/phonology.txt`) plus worked examples of the same transformation (`ancient_language_tts/data/<code>/examples.txt`): Classical Latin with /w/ for *v*, first-century Koine with ει as /i/ and υ still /y/, Chaucer's vowels before the Great Vowel Shift, and so on. The model is told to learn the decisions those examples make and apply them. That same model scores its line, then rewrites it from the assessment. It scores the rewrite next. The rewrite stands only when that score is higher than the score of the first line; otherwise the first line stands. Each standing line is scored by every model still in the tier. The score of a line is the arithmetic mean of those scores. `to_ipa` returns the line with the highest mean. On a tie, the earlier model in the tier wins. A call is tried three times. If the first transcription still fails, the model is dropped. If the assessment or the rewrite still fails, the first line stands. If a score of a standing line still fails, that model's scores are left out of the mean.
2. **Respelling.** A second LLM turns that IPA into spelling a modern voice will actually say. The system prompt depends on both languages: how the modern spelling is read (`data/<target>/phonology.txt`), which of those spellings stands in for each sound of the source (`data/<source>/respelling/<target>.txt`), and worked examples of that mapping (`data/<source>/respelling/<target>.examples.txt`). English has no letter for a Greek upsilon, and Lithuanian none for a PIE glottal laryngeal; the pair file names the substitute.
3. **Speech.** ElevenLabs speaks the respelling with a voice of the modern language. Text normalization is off, so the service does not rewrite the spelling into ordinary words. `language_code` is sent when the model accepts it. Stability defaults to 1, so the reading stays on one baseline.

## Guided chain of thought

Every LLM call reasons in the reply, then the program reads the answer out of that reply. Provider thinking tokens are not used. The reasoning is a field the call asks for.

The passage goes in the user message. The system prompt is the stable role, the rules, and the agenda. The IPA prompt closes by naming the order:

First, reason carefully about these things, in this order. Then write the result.

The output schema lists one reasoning field per item of that agenda, then the answer last. Each reasoning field is directed: its description says what to cover, in that order, and that the step uses the one before it. Constrained decoding emits the keys in declaration order, so each step is written before the next field is available. The caller reads that last field. `respell` returns the spelling string. The model writes it in the `text` field.

The IPA prompt includes worked examples and tells the model to learn the decisions they make: which reading a letter takes, which length is restored, where a syllable breaks, where the stress falls, and how the line is written. The stage of the language is already stated in the phonology file, so the call does not reason about it again.

The IPA call reasons in this order, then writes the line:

1. **Values.** Each word's consonants and vowels. A letter with two readings is settled by the rule that picks one, and by the choice the examples make for that letter.
2. **Syllables.** Those values, split by the language's syllable rules, the way the examples split them.
3. **Stress.** The syllable that takes ˈ, and the fact about those syllables that puts it there.
4. **IPA.** The line written from those stressed syllables. This is that model's transcription. The model then scores the line and rewrites it. The rewrite stands only when the same model scores it higher than the first line. `to_ipa` returns the standing line with the highest mean score across the tier.

The respelling call reasons in this order, then writes the spelling:

1. **Rules.** Which spelling each sound takes, where stress is marked, and which letters the rules allow. Where the target language has no letter for a sound, this step uses the substitute the pair rules name.
2. **Hard words.** The words those rules are hardest to apply to, and the spelling the rules require for each one.
3. **Text.** The line written from that work, in ordinary spelling: each word unbroken, spaces between words, an accent on the stressed vowel, lowercase letters, and a capital only at the start of a sentence.

## Languages

| code | language | stage |
|---|---|---|
| `lat` | Classical Latin | c. 100 BC–AD 100. ISO 639-3. |
| `lat-ecc` | Ecclesiastical Latin | Italian school, Roman pronunciation. No ISO 639 code; this one is the project's. |
| `grc` | Koine Greek | Roman-era Koine, 1st century AD. ISO 639-3. |
| `eme` | Early Modern English | c. 1600 AD. No ISO 639 code; this one is the project's. |
| `enm` | Middle English | London, c. 1400 AD. ISO 639-3. |
| `ang` | Old English | Late West Saxon, c. 900 AD. ISO 639-3. |
| `gem-pro` | Proto-Germanic | c. 200 AD. Wiktionary's code. |
| `ine-pro` | Proto-Indo-European | c. 3000 BC. Wiktionary's code. |
| `sla-pro` | Proto-Slavic | Early Common Slavic, c. 500 AD. Wiktionary's code. |

The English name works too: `source="Latin"` and `source="Classical Latin"` are `lat`. `source="Ecclesiastical Latin"` is `lat-ecc`. `source="Old English"` and `source="Proto-Indo-European"` work the same way.

`en`, `it`, `nl`, `is`, `el`, `lt`, and `uk` are the bundled modern languages. English uses the premade voice George (`JBFqnCBsd6RMkjVDRZzb`). Italian uses Giovanni (`zcAOhNBS3c14rBihAFp1`). Dutch has no premade voice, so it uses George with `language_code` `nl`; pin a Dutch voice from the voice library for a native accent. Icelandic does the same with `language_code` `is`, Modern Greek with `el`, Lithuanian with `lt`, and Ukrainian with `uk`. ElevenLabs may route an id to a newer voice on their side. `ELEVENLABS_VOICE_ID`, or `voice_id=` on a call, pins another one. The joined respelling still has to fit one ElevenLabs request.

English pair files are bundled for every ancient language except `lat` and `lat-ecc`. Those two ship with Italian. Middle English (`enm`) also ships with Dutch. Old English (`ang`) and Proto-Germanic (`gem-pro`) also ship with Icelandic. Koine Greek (`grc`) ships with Modern Greek. Proto-Indo-European (`ine-pro`) ships with Lithuanian. Proto-Slavic (`sla-pro`) ships with Ukrainian. `respell(text, "lat", "en")` has no pair file yet, and raises.

## Settings: arguments or `.env`

Each setting can be passed as a keyword to any step function. A setting you leave out is read from the environment, then from the nearest `.env` file in the current folder or above it, then the built-in default. Copy [`.env.example`](.env.example).

```python
alt.to_ipa(
    text,
    "grc",
    method="llm",
    tier="medium",
    tier_models={"medium": ["anthropic/claude-sonnet-4.5", "openai/gpt-5.6-sol"]},
)
```

`tier="medium"` is the medium list. The IPA step and the respelling step each ask every model on that list. `ancient_language_tts.TIER_MODELS` holds the built-in lists. `tier` is ignored when `method="rules"`.

## Reruns are cheap

LLM answers are cached under `.cache/ancient-language-tts/llm/`, and the audio under `.cache/ancient-language-tts/audio/`. The cache key includes the prompts, so an edit to a rules file or an examples file is a new call. The same passage, the same rules, and the same voice are not paid for twice.

## CLI

```bash
ancient-language-tts ipa "φίλος" --source grc                                  # LLM (default)
ancient-language-tts ipa "φίλος" --source grc --method rules
ancient-language-tts respell "Caesar" --source lat-ecc --target it --method rules
ancient-language-tts respell-ipa "ˈpʰi.los" --source grc --target el --text "φίλος"
ancient-language-tts speak "Cèssar" --target it --out out/caesar.mp3
ancient-language-tts languages              # codes, LLM pair files, rule engines (real or mock)
ancient-language-tts rules lat              # the IPA system prompt
ancient-language-tts rules lat it           # the respelling system prompt
```

`-v` prints each step. `-vv` prints every prompt and answer. A positional argument that points at a `.txt` or `.md` file is read as UTF-8.

## Python API

```python
alt.to_ipa(text, source, method="llm", *, tier="high", **settings) -> str
alt.respell(text, source, modern_lang="en", method="llm", *, tier="high", **settings) -> str
alt.respell_ipa(ipa, source, modern_lang="en", method="llm", *, text=None, tier="high", **settings) -> str
alt.speak(respelled_text, modern_lang="en", voice=None, *, output=None, seed=None, **settings) -> bytes
```

- `source` is an ancient-language code: `lat`, `lat-ecc`, `grc`, `eme`, `enm`, `ang`, `gem-pro`, `ine-pro`, `sla-pro`.
- `method` is `"llm"` or `"rules"`. `"rules"` makes no OpenRouter call.
- `respell_ipa(..., text=...)` is the original passage, used by the LLM for word boundaries only.
- `speak` returns the audio bytes and writes them to `output` when set. `voice` is an ElevenLabs voice id; it defaults to `ELEVENLABS_VOICE_ID`, then the language's bundled voice.
- `**settings` are the keys of `config`: `openrouter_api_key`, `elevenlabs_api_key`, `models`, `tier_models`, `elevenlabs_model`, `output_format`, `stability`, `similarity_boost`, `cache_dir`, `cache`, `max_tokens`, `max_chars`.

The rule engines can be used directly too:

```python
alt.rule_based.to_ipa(text, source)
alt.rule_based.respell(text, source, target)
alt.rule_based.respell_ipa(ipa, source, target)
alt.rule_based.engine(source)   # RuleEngine(to_ipa, respell={...}, respell_ipa={...}, mock=...)
```

To replace a mock, write a module like `rule_based/lat_ecc_it/` and point that language's `rule_based/<code>.py` `ENGINE` at it with `mock=False`.

Progress goes to the `ancient_language_tts` logger. Set `ancient_language_tts.llm` to `DEBUG` to see every prompt and answer.

## Rules files

`lat` is Classical Latin: c is /k/ and v is /w/. `lat-ecc` is Ecclesiastical Latin in the Italian school: c and g soften before front vowels, v is /v/, ae and oe are /e/, and vowel length is not pronounced. `grc` is Koine Greek of the first century AD: vowel length is gone, ει is /i/, αι is /ɛ/, οι and υ are /y/, β and γ are fricatives, φ θ χ are still aspirated stops, and the accent is a stress. The proto-languages use a speaking convention for the usual reconstruction, with the laryngeals and the jers pronounced.

| file | what the model is told |
|---|---|
| `data/<code>/phonology.txt` | How to turn that ancient language into IPA. For a modern language, how its voice reads spelling. |
| `data/<code>/examples.txt` | Worked passages in that ancient language and the IPA the rules require. The IPA prompt tells the model to learn the decisions they make and apply them. |
| `data/<code>/respelling/<modern>.txt` | Which of the modern spellings stands in for each sound of `<code>`. |
| `data/<code>/respelling/<modern>.examples.txt` | Worked lines: the original text, then the IPA, then the respelling those pair rules require. |

To add a modern language, add a `ModernLanguage` in `ancient_language_tts/languages.py` (its ElevenLabs language code and a voice id), add `data/<code>/phonology.txt`, and add a pair file and an examples file for each ancient language it should read. To add an ancient language, add `data/<code>/phonology.txt` and `data/<code>/examples.txt`, and a pair file plus examples for each modern language that should read it.

## Cost

The IPA step asks every model in the tier to transcribe, score its own line, and rewrite it. The same model scores the rewrite, and the rewrite stands only when that score is higher. Every model still scoring then scores every standing line. The respelling step does the same for that piece, and the speech is one request for the joined line. A passage over `max_chars` repeats the IPA step and the respelling step once per piece. A call that still fails on the third attempt follows the rule for that step: a failed transcription drops the model, a failed rewrite keeps the first line, and a failed score leaves that model out of the mean. A rerun of the same passage is free while the cache is on. `--tier low` is the cheap way to try a rules change: fewer models, fewer calls. After you edit a rules file or an examples file the prompts no longer match, so the next call is paid.

## Environment

| variable | |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter key. |
| `ELEVENLABS_API_KEY` | Your ElevenLabs key. |
| `ELEVENLABS_VOICE_ID` | Replaces every language's built-in voice. |
| `ANCIENT_TTS_ELEVENLABS_MODEL` | Default `eleven_v4`. `eleven_multilingual_v2` ignores `language_code`. |
| `ANCIENT_TTS_OUTPUT_FORMAT` | Default `mp3_44100_128`. |
| `ANCIENT_TTS_STABILITY`, `ANCIENT_TTS_SIMILARITY` | Voice settings, 0 to 1. Defaults are 1 and 0.75. |
| `ANCIENT_TTS_LLM_MODELS` | Comma-separated OpenRouter ids, replacing the tier lists. |
| `ANCIENT_TTS_MODELS_HIGH`, `ANCIENT_TTS_MODELS_MEDIUM`, `ANCIENT_TTS_MODELS_LOW` | Replacing one tier list. |
| `ANCIENT_TTS_CACHE_DIR` | Default `.cache/ancient-language-tts`. |
| `ANCIENT_TTS_CACHE` | `0` turns caching off. |
| `ANCIENT_TTS_MAX_TOKENS` | Per LLM call, default 16384. |
| `ANCIENT_TTS_MAX_CHARS` | Source characters one IPA call may see, default 1000. Longer passages are split on sentence boundaries. |

Arguments passed to a step function win over the environment, and real environment variables win over `.env`.

## Building and publishing

```powershell
.\build.ps1                  # builds dist/ancient-language-tts-<version>.tar.gz and .whl, and checks them
.\build.ps1 -Publish -Test   # uploads to TestPyPI
.\build.ps1 -Publish         # uploads to PyPI
```

Before each release, bump `__version__` in [`ancient_language_tts/__init__.py`](ancient_language_tts/__init__.py). PyPI never accepts the same version twice.

You can also share the wheel without PyPI: `pip install dist/ancient_language_tts-0.1.0-py3-none-any.whl`.

## License

MIT

"""ZIPA multilingual phone-recognition model (Zipformer CR-CTC), run locally on CPU.

ZIPA (Zhu et al., ACL 2025, https://github.com/lingjzhu/zipa) is trained on
IPAPack++, 17k hours of phone-transcribed speech in 88 languages, and emits
IPA characters directly. The ONNX export, its `tokens.txt` and ~1.2 GB of
fp32 weights live in a Hugging Face Hub repo and are cached under
`~/.cache/huggingface/hub` on first use.

* `ZIPA_MODEL_REPO` overrides the repo id (default: the large noisy-student
  CTC model, the most accurate ZIPA checkpoint). It may also be a local
  directory holding `model.onnx` and `tokens.txt`, for offline use. Note the
  default's weights are CC BY-NC-SA 4.0; `anyspeech/zipa-large-crctc-500k`
  is CC BY 4.0 and slightly less accurate.
* `ZIPA_MODEL_FILE` picks the ONNX file: `model.onnx` (fp32, default),
  `model.fp16.onnx` or `model.int8.onnx` (smaller and faster, a little less
  accurate).

The heavy dependencies (`onnxruntime`, `lhotse`, `torch`) are only imported
when the model is first loaded, so importing this module is cheap. Install
them with `pip install ancient-language-tts[check]`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_REPO = os.environ.get("ZIPA_MODEL_REPO", "anyspeech/zipa-large-crctc-ns-800k")
MODEL_FILE = os.environ.get("ZIPA_MODEL_FILE", "model.onnx")
TOKENS_FILE = "tokens.txt"

# Fbank features are 10 ms apart and the CTC head sees them subsampled 2x, so
# one output frame is 20 ms = 320 samples. Long audio is only split on this grid.
TICK_MS = 20
TICK_SAMPLES = SAMPLE_RATE * TICK_MS // 1000  # 320

# Long audio is cut into pieces of at most L samples before inference.
L_MS = 100_000
L = SAMPLE_RATE * L_MS // 1000

# Split-point search: lowest energy in a 100 ms neighbourhood, within 25% of centre.
ENERGY_NEIGHBOURHOOD_MS = 100
ENERGY_NEIGHBOURHOOD_SAMPLES = SAMPLE_RATE * ENERGY_NEIGHBOURHOOD_MS // 1000  # 1600
SPLIT_CENTER_FRAC = 0.25


def _sumsq_strided_windows(audio, first_start, n_windows, step, win):
    last_start = first_start + (n_windows - 1) * step
    windows = sliding_window_view(audio, win)[first_start:last_start + 1:step]
    return np.einsum("ij,ij->i", windows, windows)


def _find_split(audio, lo, hi, tick=TICK_SAMPLES):
    """Index at which to bisect audio[lo:hi]: the quietest tick near the centre."""
    length = hi - lo
    if length < 2 * tick:
        return None

    n = int(audio.shape[0])
    center = lo + (length // 2)
    k_min = lo // tick + 1
    k_max = (hi - 1) // tick
    if k_min > k_max:
        return None

    half = ENERGY_NEIGHBOURHOOD_SAMPLES // 2
    win = half * 2
    if n >= win:
        k_min = max(k_min, (half + tick - 1) // tick)
        k_max = min(k_max, (n - half) // tick)
    if k_min > k_max:
        k_min = lo // tick + 1
        k_max = (hi - 1) // tick
        k = (center + tick // 2) // tick
        return int(min(max(k, k_min), k_max) * tick)

    ks = np.arange(k_min, k_max + 1, dtype=np.int64)
    sites = ks * tick
    ks = ks[np.abs(sites - center) < SPLIT_CENTER_FRAC * length]
    if ks.size == 0:
        k = (center + tick // 2) // tick
        ks = np.array([min(max(k, k_min), k_max)], dtype=np.int64)

    sites = ks * tick
    first_start = int(sites[0]) - half
    energies = _sumsq_strided_windows(audio, first_start, int(sites.size), tick, win)
    best = int(np.lexsort((np.abs(sites - center), energies))[0])
    return int(sites[best])


def _split_range(audio, lo, hi, max_samples, bounds):
    if hi - lo <= max_samples:
        bounds.append((lo, hi))
        return
    split = _find_split(audio, lo, hi)
    if split is None or split <= lo or split >= hi:
        bounds.append((lo, hi))
        return
    _split_range(audio, lo, split, max_samples, bounds)
    _split_range(audio, split, hi, max_samples, bounds)


def split_waveform(audio, max_samples=L):
    """Cut a 16 kHz waveform into views of at most `max_samples`, at 20 ms ticks."""
    audio = np.ascontiguousarray(np.ravel(audio), dtype=np.float32)
    n = int(audio.shape[0])
    if n <= max_samples:
        return [audio]
    bounds = []
    _split_range(audio, 0, n, max_samples, bounds)
    return [audio[s:e] for s, e in bounds]


def load_tokens(path: str | Path) -> dict[int, str]:
    """`tokens.txt` ("<token> <id>" per line) as {id: token}."""
    tokens = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            tokens[int(parts[1]) if len(parts) > 1 else len(tokens)] = parts[0]
    return tokens


def _resolve(model_id: str, filename: str) -> str:
    local = Path(model_id)
    if local.is_dir():
        return str(local / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename)


class PhonemeCTCModel:
    """The ZIPA CTC ONNX graph, its fbank front end and its token table."""

    def __init__(self, model_id: str | None = None, model_file: str | None = None):
        try:
            import onnxruntime as ort
            import torch  # noqa: F401  (lhotse's fbank runs on torch)
            from lhotse.features.kaldi.extractors import Fbank, FbankConfig
        except ImportError as exc:
            raise RuntimeError(
                "Phoneme recognition needs onnxruntime, lhotse and torch: "
                "pip install 'ancient-language-tts[check]'"
            ) from exc

        self.model_id = model_id or MODEL_REPO
        self.model_file = model_file or MODEL_FILE
        t0 = time.perf_counter()
        self.tokens = load_tokens(_resolve(self.model_id, TOKENS_FILE))
        # The same front end ZIPA was trained with (its inference/utils.py).
        self.fbank = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
        # ORT's CPU arena never shrinks, so RSS ratchets up with the longest
        # clip seen so far. Without it memory is freed after each inference.
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            _resolve(self.model_id, self.model_file),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info(
            "Loaded ZIPA model %r (%s) in %.2fs",
            self.model_id, self.model_file, time.perf_counter() - t0,
        )

    def _forward(self, audio: np.ndarray) -> np.ndarray:
        import torch

        features = self.fbank.extract_batch(
            [torch.from_numpy(audio)], sampling_rate=SAMPLE_RATE
        )[0]
        features = np.asarray(features, dtype=np.float32)[None]  # (1, frames, 80)
        lens = np.array([features.shape[1]], dtype=np.int64)
        log_probs = self.session.run(None, {"x": features, "x_lens": lens})[0]
        return np.array(log_probs[0], dtype=np.float32, copy=True)

    def _forward_aligned(self, audio: np.ndarray) -> np.ndarray:
        """`_forward`, padded or trimmed to one row per started 20 ms tick.

        The encoder drops a few edge frames, so without this the timestamps
        of later pieces of a split clip would drift earlier.
        """
        rows = self._forward(audio)
        want = -(-len(audio) // TICK_SAMPLES)
        if len(rows) >= want:
            return rows[:want]
        pad = np.full((want - len(rows), rows.shape[1]), -1e4, dtype=np.float32)
        pad[:, 0] = 0.0  # log P(blank) = 0
        return np.concatenate([rows, pad], axis=0)

    def get_logits(self, audio: np.ndarray) -> np.ndarray:
        """CTC log-probabilities of shape (time, vocab), one row per 20 ms,
        for one mono 16 kHz float32 waveform.

        Audio longer than 20 s is split at quiet ticks, and the per-piece
        rows are concatenated along time.
        """
        pieces = split_waveform(audio, max_samples=L)
        if len(pieces) == 1:
            return self._forward(pieces[0])
        return np.concatenate([self._forward_aligned(p) for p in pieces], axis=0)


_instances: dict[tuple[str, str], PhonemeCTCModel] = {}
_lock = threading.Lock()


def get_phoneme_model(
    model_id: str | None = None, model_file: str | None = None
) -> PhonemeCTCModel:
    """The cached model for `model_id`/`model_file` (default `ZIPA_MODEL_*`), loaded once."""
    key = (model_id or MODEL_REPO, model_file or MODEL_FILE)
    with _lock:
        if key not in _instances:
            _instances[key] = PhonemeCTCModel(*key)
        return _instances[key]

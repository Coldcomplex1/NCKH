#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune vinai/PhoWhisper-large on the ViMD Vietnamese multi-dialect ASR dataset
(nguyendv02/ViMD_Dataset) and report WER on the official test split.

Run with:

    pip install -r requirements.txt
    python3 main.py

The script is fully unattended: it downloads the model and the dataset, builds a
16 kHz cache, fine-tunes with early stopping on validation WER, selects the best
checkpoint, and decodes the test split exactly once. Every stage is idempotent, so
re-running after an interruption resumes instead of starting over.

All tunables are overridable through VIMD_* environment variables (see README.MD).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The environment must be configured BEFORE torch / transformers / huggingface_hub
# are imported, otherwise these settings have no effect.
# ---------------------------------------------------------------------------
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Never let an experiment tracker try to authenticate: it would block on a prompt
# and the lab runs this job unattended.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("COMET_MODE", "DISABLED")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Bound BLAS threads so 8 dataloader workers do not oversubscribe the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import atexit
import csv
import errno
import gc
import fcntl
import hashlib
import io
import itertools
import json
import logging
import math
import platform
import re
import shutil
import socket
import sys
import time
import traceback
import unicodedata
import wave
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

LOGGER = logging.getLogger("vimd")


# ===========================================================================
# 1. Configuration
# ===========================================================================

def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


_TRUE_WORDS = ("1", "true", "yes", "y", "on")
_FALSE_WORDS = ("0", "false", "no", "n", "off")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    text = value.strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    # Silently reading an unrecognised spelling as False would change the recipe
    # without anyone noticing.
    raise ValueError(
        f"{name}={value!r} is not a boolean. Use one of {_TRUE_WORDS + _FALSE_WORDS}."
    )


@dataclass
class Config:
    # --- model / dataset -------------------------------------------------
    model_id: str = field(default_factory=lambda: _env_str("VIMD_MODEL_ID", "vinai/PhoWhisper-large"))
    dataset_id: str = field(default_factory=lambda: _env_str("VIMD_DATASET_ID", "nguyendv02/ViMD_Dataset"))
    # Empty means "resolve once on the first run and pin it from then on".
    dataset_revision: str = field(default_factory=lambda: _env_str("VIMD_DATASET_REVISION", ""))
    model_revision: str = field(default_factory=lambda: _env_str("VIMD_MODEL_REVISION", ""))
    language: str = "vi"
    task: str = "transcribe"
    sampling_rate: int = 16000

    # --- optimisation ----------------------------------------------------
    learning_rate: float = field(default_factory=lambda: _env_float("VIMD_LR", 5e-6))
    weight_decay: float = field(default_factory=lambda: _env_float("VIMD_WEIGHT_DECAY", 0.0))
    max_grad_norm: float = field(default_factory=lambda: _env_float("VIMD_MAX_GRAD_NORM", 1.0))
    warmup_steps: int = field(default_factory=lambda: _env_int("VIMD_WARMUP_STEPS", 500))
    num_train_epochs: float = field(default_factory=lambda: _env_float("VIMD_EPOCHS", 20.0))
    train_batch_size: int = field(default_factory=lambda: _env_int("VIMD_TRAIN_BATCH_SIZE", 16))
    grad_accum: int = field(default_factory=lambda: _env_int("VIMD_GRAD_ACCUM", 4))
    eval_batch_size: int = field(default_factory=lambda: _env_int("VIMD_EVAL_BATCH_SIZE", 16))
    beam_batch_size: int = field(default_factory=lambda: _env_int("VIMD_BEAM_BATCH_SIZE", 16))
    lr_scheduler_type: str = field(default_factory=lambda: _env_str("VIMD_LR_SCHEDULER", "linear"))
    seed: int = field(default_factory=lambda: _env_int("VIMD_SEED", 42))

    # --- evaluation / checkpointing --------------------------------------
    eval_steps: int = field(default_factory=lambda: _env_int("VIMD_EVAL_STEPS", 250))
    logging_steps: int = field(default_factory=lambda: _env_int("VIMD_LOGGING_STEPS", 25))
    save_total_limit: int = field(default_factory=lambda: _env_int("VIMD_SAVE_TOTAL_LIMIT", 3))
    early_stopping_patience: int = field(default_factory=lambda: _env_int("VIMD_EARLY_STOPPING_PATIENCE", 5))
    early_stopping_threshold: float = field(default_factory=lambda: _env_float("VIMD_EARLY_STOPPING_THRESHOLD", 5e-4))
    final_num_beams: int = field(default_factory=lambda: _env_int("VIMD_FINAL_NUM_BEAMS", 5))

    # --- regularisation --------------------------------------------------
    apply_spec_augment: bool = field(default_factory=lambda: _env_bool("VIMD_SPEC_AUGMENT", True))
    mask_time_prob: float = field(default_factory=lambda: _env_float("VIMD_MASK_TIME_PROB", 0.05))
    mask_time_length: int = field(default_factory=lambda: _env_int("VIMD_MASK_TIME_LENGTH", 10))
    mask_feature_prob: float = field(default_factory=lambda: _env_float("VIMD_MASK_FEATURE_PROB", 0.0))
    mask_feature_length: int = field(default_factory=lambda: _env_int("VIMD_MASK_FEATURE_LENGTH", 10))

    # --- runtime ---------------------------------------------------------
    dataloader_workers: int = field(default_factory=lambda: _env_int("VIMD_DATALOADER_WORKERS", 8))
    download_retries: int = field(default_factory=lambda: _env_int("VIMD_DOWNLOAD_RETRIES", 5))
    # Full extra sweeps over any shards that still failed, to ride out a network
    # outage longer than the per-shard retries without needing a re-launch.
    prepare_sweeps: int = field(default_factory=lambda: _env_int("VIMD_PREPARE_SWEEPS", 3))
    oom_retries: int = field(default_factory=lambda: _env_int("VIMD_OOM_RETRIES", 3))
    min_free_gb: float = field(default_factory=lambda: _env_float("VIMD_MIN_FREE_GB", 100.0))
    recommended_free_gb: float = field(default_factory=lambda: _env_float("VIMD_RECOMMENDED_FREE_GB", 200.0))
    expand_numbers: bool = field(default_factory=lambda: _env_bool("VIMD_EXPAND_NUMBERS", True))
    keep_raw_parquet: bool = field(default_factory=lambda: _env_bool("VIMD_KEEP_RAW_PARQUET", False))
    # Training on a silently incomplete dataset would invalidate the reported score,
    # so a train shard that cannot be prepared aborts the run by default.
    allow_incomplete_train: bool = field(default_factory=lambda: _env_bool("VIMD_ALLOW_INCOMPLETE_TRAIN", False))
    min_vram_gb: float = field(default_factory=lambda: _env_float("VIMD_MIN_VRAM_GB", 70.0))
    min_ram_gb: float = field(default_factory=lambda: _env_float("VIMD_MIN_RAM_GB", 24.0))

    # --- paths -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env_str("VIMD_DATA_DIR", str(ROOT / "data" / "vimd_16k"))))
    raw_dir: Path = field(default_factory=lambda: Path(_env_str("VIMD_RAW_DIR", str(ROOT / "data" / "_raw_parquet"))))
    output_dir: Path = field(default_factory=lambda: Path(_env_str("VIMD_OUTPUT_DIR", str(ROOT / "outputs"))))

    # Whisper hard limits, taken from the model config. Not user tunable.
    max_label_tokens: int = 448
    max_audio_seconds: float = 30.0

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def best_model_dir(self) -> Path:
        return self.output_dir / "best_model"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def results_path(self) -> Path:
        return self.output_dir / "final_results.json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self).items()
        }

    def validate(self) -> None:
        """Fail on nonsensical settings now, with a readable message, rather than
        several hours later inside the Trainer."""
        positive_ints = {
            "train_batch_size": self.train_batch_size,
            "grad_accum": self.grad_accum,
            "eval_batch_size": self.eval_batch_size,
            "beam_batch_size": self.beam_batch_size,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "early_stopping_patience": self.early_stopping_patience,
            "final_num_beams": self.final_num_beams,
            "download_retries": self.download_retries,
            "prepare_sweeps": self.prepare_sweeps,
            "sampling_rate": self.sampling_rate,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        non_negative_ints = {"warmup_steps": self.warmup_steps, "oom_retries": self.oom_retries,
                             "dataloader_workers": self.dataloader_workers}
        for name, value in non_negative_ints.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs!r}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay!r}")
        for name in ("mask_time_prob", "mask_feature_prob"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        if self.sampling_rate != 16000:
            raise ValueError(
                f"sampling_rate must be 16000: Whisper's feature extractor and positional "
                f"embeddings assume it. Got {self.sampling_rate}."
            )
        if self.min_free_gb > self.recommended_free_gb:
            raise ValueError("VIMD_MIN_FREE_GB cannot exceed VIMD_RECOMMENDED_FREE_GB")


# The official ViMD split names are train / valid / test. There is no split
# literally called "validation"; `valid` is the validation split.
SPLITS: Tuple[str, ...] = ("train", "valid", "test")
METADATA_COLUMNS: Tuple[str, ...] = (
    "region",
    "province_code",
    "province_name",
    "filename",
    "text",
    "speakerID",
    "gender",
)


# ===========================================================================
# 2. Logging helpers
# ===========================================================================

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("urllib3", "filelock", "huggingface_hub.file_download", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()


def banner(title: str) -> None:
    LOGGER.info("=" * 78)
    LOGGER.info(title)
    LOGGER.info("=" * 78)


# Locks this process currently holds, keyed by resolved path.
_HELD_LOCKS: Dict[str, "DirectoryLock"] = {}


class DirectoryLock:
    """Exclusive advisory lock over one directory, held via fcntl.flock.

    flock is the right primitive here because the kernel drops the lock when the owning
    process dies - including on SIGKILL, an OOM kill or a node reboot. That removes the
    stale-lock problem entirely, and with it the read-then-take-over sequence that a
    PID-file lock needs: two processes racing to reclaim the same dead owner's file can
    both delete and recreate it and both believe they won. Here the kernel arbitrates,
    so exactly one process holds the lock at any instant.

    The file's contents are diagnostics only - who holds it and since when - never the
    locking mechanism."""

    def __init__(self, directory: Path, purpose: str) -> None:
        self.directory = directory
        self.purpose = purpose
        self.path = directory / ".vimd.lock"
        self._fd: Optional[int] = None

    def acquire(self) -> "DirectoryLock":
        if self._fd is not None:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        # flock is per open-file-description, so a second os.open in *this* process would
        # block against our own lock. Hand back the instance we already hold instead.
        key = str(self.path.resolve())
        held = _HELD_LOCKS.get(key)
        if held is not None and held._fd is not None:
            return held
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                raise
            owner = _read_json(self.path)
            os.close(fd)
            detail = ""
            if isinstance(owner, dict):
                detail = (
                    f" (pid {owner.get('pid')} on {owner.get('host')}, "
                    f"started {owner.get('started')})"
                )
            raise RuntimeError(
                f"another run is already using {self.directory} for {self.purpose}{detail}. "
                f"Wait for it to finish, or use a different directory. The lock is held by a "
                f"live process; it is released automatically when that process exits, so there "
                f"is nothing to clean up by hand."
            ) from exc

        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "purpose": self.purpose,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }).encode("utf-8"))
        os.fsync(fd)
        self._fd = fd
        _HELD_LOCKS[key] = self
        atexit.register(self.release)
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        _HELD_LOCKS.pop(str(self.path.resolve()), None)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        # The file itself is left in place: unlinking it would let another process that
        # already opened it lock a now-orphaned inode.

    def __enter__(self) -> "DirectoryLock":
        return self.acquire()

    def __exit__(self, *_exc: Any) -> None:
        self.release()


_TMP_COUNTER = itertools.count()


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write via a temp file and os.replace.

    A plain write_text that is interrupted leaves a truncated file behind, and
    several of these files are read back on the next run to decide what work to
    skip - a half-written one would silently corrupt that decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process: two runs sharing a data directory would otherwise collide on
    # one another's temp file and produce a corrupt result.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{next(_TMP_COUNTER)}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    """Read a JSON file, treating damage as absence rather than crashing."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unnamed"


def _digest(payload: Any) -> str:
    """Stable short hash of a JSON-serialisable structure."""
    blob = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def recipe_fingerprint(
    cfg: Config, dataset_fingerprint: Dict[str, Any], model_revision: str = ""
) -> Dict[str, Any]:
    """Everything that changes what the produced model and score mean.

    Reusing an output directory after changing any of this would otherwise resume
    another experiment's checkpoints, or report its results as this run's. Note that
    per-device batch size and accumulation are folded into the effective batch: the OOM
    back-off changes those two deliberately, mid-run, while keeping the recipe intact."""
    return {
        "model_id": cfg.model_id,
        # Pinned like the dataset: an upstream PhoWhisper update between restarts would
        # otherwise swap the weights or the tokenizer under an unchanged recipe.
        "model_revision": model_revision,
        "dataset": dataset_fingerprint,
        "language": cfg.language,
        "task": cfg.task,
        # Changes eval_wer_norm, therefore early stopping and checkpoint selection,
        # therefore the reported score. (It is deliberately absent from the *decode*
        # signature, where hypotheses genuinely do not depend on it.)
        "expand_numbers": cfg.expand_numbers,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_steps": cfg.warmup_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "effective_batch_size": cfg.train_batch_size * cfg.grad_accum,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "apply_spec_augment": cfg.apply_spec_augment,
        "mask_time_prob": cfg.mask_time_prob,
        "mask_time_length": cfg.mask_time_length,
        "mask_feature_prob": cfg.mask_feature_prob,
        "mask_feature_length": cfg.mask_feature_length,
        "eval_steps": cfg.eval_steps,
        "early_stopping_patience": cfg.early_stopping_patience,
        "early_stopping_threshold": cfg.early_stopping_threshold,
        "final_num_beams": cfg.final_num_beams,
        "max_label_tokens": cfg.max_label_tokens,
        "seed": cfg.seed,
        "pipeline_version": PIPELINE_VERSION,
    }


def _describe_recipe_change(previous: Dict[str, Any], current: Dict[str, Any]) -> str:
    keys = sorted(set(previous) | set(current))
    lines = [
        f"    {key}: {previous.get(key, '<absent>')!r} -> {current.get(key, '<absent>')!r}"
        for key in keys
        if previous.get(key) != current.get(key)
    ]
    return "\n".join(lines) or "    (no field-level difference; the structures differ)"


def _checkpoint_fingerprint(path: Path) -> str:
    """Identify a checkpoint by the size and mtime of its weight files, so a cache
    entry cannot be reused for a different set of weights that happens to sit at the
    same step number."""
    parts = []
    for weight in sorted(list(path.glob("model*.safetensors")) + list(path.glob("pytorch_model.bin"))):
        stat = weight.stat()
        parts.append((weight.name, stat.st_size, int(stat.st_mtime)))
    return _digest(parts)


def _json_safe(value: Any) -> Any:
    """JSON has no NaN/Infinity; emit null instead so the results file stays valid."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


# ===========================================================================
# 3. Vietnamese text handling and WER metrics
# ===========================================================================
#
# Two WER figures are reported for every evaluation:
#
#   raw WER         reference vs hypothesis after Unicode NFC unification and
#                   whitespace collapsing only. Casing, punctuation and diacritics
#                   all count as differences.
#
#   normalized WER  the checkpoint-selection metric. A deterministic pipeline that
#                   removes differences which are orthographic rather than
#                   acoustic, applied identically to reference and hypothesis.
#
# Diacritics are preserved throughout: lowercasing is str.lower() over NFC text,
# never an ASCII fold.

# The five Vietnamese combining tone marks, written as escapes because they are
# zero-width in an editor and trivially mangled by copy-paste.
_TONE_MARKS = "\u0300\u0301\u0303\u0309\u0323"  # grave, acute, tilde, hook above, dot below
# After NFD the only base letters left are ASCII plus Đ / đ, which never decompose.
_VN_BASE_LETTERS = "a-zA-ZĐđ"

# "hoà" -> "hòa", "khoẻ" -> "khỏe", "thuý" -> "thúy". Only fires when the vowel
# cluster ends the syllable (no following base letter), which is exactly where the
# two Vietnamese tone-placement conventions disagree. "uy" after "q" is excluded
# because there the u belongs to the "qu" digraph ("quý", never "qúy").
_TONE_MOVE_RE = re.compile(
    rf"(o[ae]|(?<!q)uy)([{_TONE_MARKS}])(?![{_VN_BASE_LETTERS}])",
    re.IGNORECASE,
)
# The mirror-image typo after "qu": "qủa" -> "quả", "qúy" -> "quý".
_QU_FIX_RE = re.compile(rf"(?<=q)u([{_TONE_MARKS}])([aeiouy])", re.IGNORECASE)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(rf"[^\w\s{_TONE_MARKS}]|_", re.UNICODE)
_NUMBER_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+")

_VN_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_VN_SCALES = ("", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ")


def _read_three_digit_group(value: int, force_hundreds: bool) -> str:
    """Read 0..999 in Vietnamese. force_hundreds is set for non-leading groups,
    where the hundreds digit is spoken even when it is zero ("không trăm")."""
    hundreds, tens, units = value // 100, (value // 10) % 10, value % 10
    parts: List[str] = []
    if hundreds > 0 or force_hundreds:
        parts.append(f"{_VN_DIGITS[hundreds]} trăm")
    if tens == 0:
        if units > 0:
            if hundreds > 0 or force_hundreds:
                parts.append("lẻ")
            parts.append(_VN_DIGITS[units])
    elif tens == 1:
        parts.append("mười")
        if units == 5:
            parts.append("lăm")
        elif units > 0:
            parts.append(_VN_DIGITS[units])
    else:
        parts.append(f"{_VN_DIGITS[tens]} mươi")
        if units == 1:
            parts.append("mốt")
        elif units == 4:
            parts.append("tư")
        elif units == 5:
            parts.append("lăm")
        elif units > 0:
            parts.append(_VN_DIGITS[units])
    return " ".join(parts)


def read_vietnamese_number(value: int) -> str:
    """Spell out a non-negative integer the way it is read aloud in Vietnamese."""
    if value < 0:
        raise ValueError("negative numbers are not handled")
    if value == 0:
        return _VN_DIGITS[0]
    groups: List[int] = []
    while value > 0:
        groups.append(value % 1000)
        value //= 1000
    if len(groups) > len(_VN_SCALES):
        raise ValueError("number too large to read")
    top = len(groups) - 1
    parts: List[str] = []
    for index in range(top, -1, -1):
        group = groups[index]
        if group == 0:
            continue
        parts.append(_read_three_digit_group(group, force_hundreds=index != top))
        if _VN_SCALES[index]:
            parts.append(_VN_SCALES[index])
    return " ".join(parts)


def _expand_numbers(text: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        digits = match.group(0).replace(".", "").replace(",", "")
        if not digits or len(digits) > 15:
            return match.group(0)
        try:
            return " " + read_vietnamese_number(int(digits)) + " "
        except ValueError:
            return match.group(0)

    return _NUMBER_RE.sub(replace, text)


def canonicalize_tone_marks(text: str) -> str:
    """Make the two Vietnamese tone-placement conventions comparable."""
    decomposed = unicodedata.normalize("NFD", text)
    decomposed = _QU_FIX_RE.sub(lambda m: f"u{m.group(2)}{m.group(1)}", decomposed)
    decomposed = _TONE_MOVE_RE.sub(
        lambda m: f"{m.group(1)[0]}{m.group(2)}{m.group(1)[1]}", decomposed
    )
    return unicodedata.normalize("NFC", decomposed)


def normalize_raw(text: str) -> str:
    """Minimal normalisation used for the raw WER: NFC plus whitespace collapsing."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def normalize_for_wer(text: str, expand_numbers: bool = True) -> str:
    """Full Vietnamese normalisation used for the normalized WER."""
    text = unicodedata.normalize("NFC", text or "").lower()
    text = canonicalize_tone_marks(text)
    text = text.replace("%", " phần trăm ")
    if expand_numbers:
        text = _expand_numbers(text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass
class WerReport:
    wer_norm: float
    wer_raw: float
    cer_norm: float
    num_pairs: int
    num_skipped_empty_refs: int
    per_utterance_wer: List[Optional[float]]
    normalized_refs: List[str]
    normalized_hyps: List[str]


def compute_wer_report(
    references: Sequence[str],
    hypotheses: Sequence[str],
    expand_numbers: bool = True,
) -> WerReport:
    """Corpus-level WER/CER (total edits over total reference words).

    jiwer raises on empty references, so those pairs are excluded from the corpus
    figures and counted separately rather than silently scored as zero."""
    import jiwer

    raw_refs = [normalize_raw(r) for r in references]
    raw_hyps = [normalize_raw(h) for h in hypotheses]
    norm_refs = [normalize_for_wer(r, expand_numbers) for r in references]
    norm_hyps = [normalize_for_wer(h, expand_numbers) for h in hypotheses]

    keep_norm = [i for i, ref in enumerate(norm_refs) if ref]
    keep_raw = [i for i, ref in enumerate(raw_refs) if ref]

    def corpus(metric, refs: Sequence[str], hyps: Sequence[str], keep: Sequence[int]) -> float:
        if not keep:
            return float("nan")
        return float(metric([refs[i] for i in keep], [hyps[i] for i in keep]))

    per_utterance: List[Optional[float]] = [
        float(jiwer.wer(ref, hyp)) if ref else None for ref, hyp in zip(norm_refs, norm_hyps)
    ]

    return WerReport(
        wer_norm=corpus(jiwer.wer, norm_refs, norm_hyps, keep_norm),
        wer_raw=corpus(jiwer.wer, raw_refs, raw_hyps, keep_raw),
        cer_norm=corpus(jiwer.cer, norm_refs, norm_hyps, keep_norm),
        num_pairs=len(references),
        num_skipped_empty_refs=len(references) - len(keep_norm),
        per_utterance_wer=per_utterance,
        normalized_refs=norm_refs,
        normalized_hyps=norm_hyps,
    )


def grouped_wer(
    records: Sequence[Dict[str, Any]],
    report: WerReport,
    key: str,
) -> Dict[str, Dict[str, float]]:
    """WER broken down by a metadata field (region, province_name, ...)."""
    import jiwer

    buckets: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        if not report.normalized_refs[index]:
            continue
        buckets.setdefault(str(record.get(key, "unknown")), []).append(index)

    out: Dict[str, Dict[str, float]] = {}
    for name, indices in sorted(buckets.items()):
        out[name] = {
            "wer_norm": float(
                jiwer.wer(
                    [report.normalized_refs[i] for i in indices],
                    [report.normalized_hyps[i] for i in indices],
                )
            ),
            "num_utterances": len(indices),
        }
    return out


# ===========================================================================
# 4. Dataset preparation: Hub parquet -> local 16 kHz mono int16 shards
# ===========================================================================
#
# The published ViMD parquet files hold 44.1 kHz stereo 16-bit WAV and total
# ~60 GB. Shards are downloaded, converted and deleted one at a time so peak raw
# disk stays around one shard, and the resulting cache is ~12 GB.
#
# Layout produced under data/vimd_16k/<split>/:
#   shard_00000.bin    concatenated little-endian int16 PCM, 16 kHz mono
#   shard_00000.jsonl  one record per utterance: byte offset, length, text, metadata
#   shard_00000.done   written last; its absence means the shard must be redone

_SHARD_NAME_RE = re.compile(
    r"^data/(?P<split>train|valid|test)-(?P<index>\d+)-of-\d+\.parquet$"
)


def _decode_wav_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes to float32 samples in [-1, 1] plus the native sample rate."""
    try:
        import soundfile as sf

        data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(rate)
    except Exception:  # pragma: no cover - fallback if libsndfile cannot open it
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise RuntimeError(f"unsupported WAV sample width: {width} bytes")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return samples, rate


def _to_target_mono_int16(samples: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """Downmix to mono and resample to target_rate. The gcd-derived up/down factors
    make this correct for any source rate, not just the 44.1 kHz ViMD ships today."""
    from scipy.signal import resample_poly

    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)
    if int(rate) != int(target_rate):
        divisor = math.gcd(int(rate), int(target_rate))
        samples = resample_poly(
            samples, target_rate // divisor, int(rate) // divisor
        ).astype(np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return np.rint(samples * 32767.0).astype("<i2")


def _audio_bytes_from_column(column: Any, row: int) -> bytes:
    """ViMD stores audio as a HuggingFace Audio feature, i.e. a {bytes, path} struct."""
    import pyarrow as pa

    if pa.types.is_struct(column.type):
        value = column.field("bytes")[row].as_py()
        if value is None:
            path = column.field("path")[row].as_py()
            raise RuntimeError(f"audio row carries no inline bytes (path={path!r})")
        return value
    value = column[row].as_py()
    if value is None:
        raise RuntimeError("audio row is null")
    return value


def _retry(operation, description: str, attempts: int):
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - network failures are broad by nature
            last_error = exc
            wait = min(60.0, 5.0 * (2 ** (attempt - 1)))
            LOGGER.warning(
                "%s failed on attempt %d/%d (%s)%s",
                description, attempt, attempts, exc,
                f"; retrying in {wait:.0f}s" if attempt < attempts else "",
            )
            if attempt < attempts:
                time.sleep(wait)
    raise RuntimeError(f"{description} failed after {attempts} attempts") from last_error


# Bumped when the on-disk shard format changes, so an old cache is not silently reused.
SHARD_FORMAT_VERSION = 2
# Bumped when a change to this script would invalidate existing checkpoints or results.
PIPELINE_VERSION = 1


def _shard_identity(cfg: Config, filename: str, revision: str) -> Dict[str, Any]:
    """What a prepared shard must match to be reusable. Shard number alone is not
    enough: a different dataset id, or an upstream revision with the same file names,
    would otherwise be served from a stale cache."""
    return {
        "format": SHARD_FORMAT_VERSION,
        "dataset_id": cfg.dataset_id,
        "revision": revision,
        "filename": filename,
        "sampling_rate": cfg.sampling_rate,
    }


def _shard_is_current(done_path: Path, identity: Dict[str, Any]) -> bool:
    marker = _read_json(done_path)
    if not isinstance(marker, dict):
        return False
    return all(marker.get(key) == value for key, value in identity.items())


def _convert_one_shard(
    cfg: Config, filename: str, split_dir: Path, shard_index: int, revision: str
) -> Tuple[int, float]:
    """Download one parquet shard, convert it, and write the local shard files."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    bin_path = split_dir / f"shard_{shard_index:05d}.bin"
    jsonl_path = split_dir / f"shard_{shard_index:05d}.jsonl"
    done_path = split_dir / f"shard_{shard_index:05d}.done"
    if done_path.exists():
        done_path.unlink()  # stale identity; the marker must not survive a failed redo

    # A previous run may have died mid-write; always start this shard from scratch.
    for stale in (bin_path, jsonl_path):
        if stale.exists():
            stale.unlink()

    scratch = _scratch_dir(cfg.raw_dir)
    parquet_path = Path(
        _retry(
            lambda: hf_hub_download(
                repo_id=cfg.dataset_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,  # pinned, so an upstream push mid-run cannot mix revisions
                local_dir=str(scratch),
            ),
            f"download of {filename}",
            cfg.download_retries,
        )
    )

    num_rows = 0
    total_seconds = 0.0
    try:
        with pq.ParquetFile(str(parquet_path)) as parquet_file, \
                open(bin_path, "wb") as audio_out, \
                open(jsonl_path, "w", encoding="utf-8") as meta_out:
            offset = 0
            for batch in parquet_file.iter_batches(batch_size=8):
                audio_column = batch.column("audio")
                metadata = {
                    name: batch.column(name).to_pylist()
                    for name in METADATA_COLUMNS
                    if name in batch.schema.names
                }
                for row in range(batch.num_rows):
                    samples, rate = _decode_wav_bytes(_audio_bytes_from_column(audio_column, row))
                    pcm = _to_target_mono_int16(samples, rate, cfg.sampling_rate)
                    audio_out.write(pcm.tobytes())

                    record: Dict[str, Any] = {name: values[row] for name, values in metadata.items()}
                    record["text"] = normalize_raw(record.get("text") or "")
                    record["offset"] = offset
                    record["num_samples"] = int(pcm.shape[0])
                    record["duration"] = float(pcm.shape[0]) / cfg.sampling_rate
                    record["source_sampling_rate"] = int(rate)
                    meta_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                    offset += int(pcm.shape[0]) * 2
                    num_rows += 1
                    total_seconds += record["duration"]
            audio_out.flush()
            os.fsync(audio_out.fileno())
            meta_out.flush()
            os.fsync(meta_out.fileno())
    except BaseException:
        # No .done marker will be written, so leaving these would only waste disk
        # until the next attempt cleans them up.
        for partial in (bin_path, jsonl_path):
            try:
                partial.unlink()
            except OSError:
                pass
        raise
    finally:
        if not cfg.keep_raw_parquet:
            try:
                parquet_path.unlink()
            except OSError:
                pass

    marker = dict(_shard_identity(cfg, filename, revision))
    marker.update({"rows": num_rows, "seconds": total_seconds})
    _write_json_atomic(done_path, marker)
    return num_rows, total_seconds


# Everything downloaded goes inside this subdirectory of VIMD_RAW_DIR, and only this
# subdirectory is ever deleted. Downloading straight into VIMD_RAW_DIR would mean the
# cleanup had to remove <raw_dir>/data and <raw_dir>/.cache - names a user may well
# already be using for their own files.
_SCRATCH_DIRNAME = ".vimd_scratch"
_SCRATCH_MARKER = ".created_by_vimd_main"


def _scratch_dir(raw_dir: Path) -> Path:
    """The disposable directory this script owns outright, created on first use."""
    scratch = raw_dir / _SCRATCH_DIRNAME
    scratch.mkdir(parents=True, exist_ok=True)
    marker = scratch / _SCRATCH_MARKER
    if not marker.exists():
        marker.write_text(
            "Created by main.py. Everything in this directory is disposable scratch "
            "space for ViMD parquet downloads and is deleted automatically.\n",
            encoding="utf-8",
        )
    return scratch


def _cleanup_raw_dir(raw_dir: Path) -> None:
    """Delete the scratch directory, and nothing else.

    The recursive delete is confined to <raw_dir>/.vimd_scratch/, and only runs when
    that directory carries the marker written when we created it. Anything the user
    keeps in VIMD_RAW_DIR - including their own data/ or .cache/ - is untouched."""
    scratch = raw_dir / _SCRATCH_DIRNAME
    if not scratch.is_dir() or scratch.is_symlink():
        return
    if not (scratch / _SCRATCH_MARKER).is_file():
        LOGGER.warning(
            "%s exists but was not created by this script; leaving it alone", scratch
        )
        return
    shutil.rmtree(scratch, ignore_errors=True)


def prepare_dataset(cfg: Config) -> str:
    """Materialise the local 16 kHz cache. Safe to call repeatedly.

    Returns the dataset revision every shard was built from."""
    banner("Stage 1/5 - dataset preparation")
    # Held only while shards are being written. Two experiments with different output
    # directories may legitimately share a prepared cache - but they must not build it
    # at the same time, and the output-directory lock does not cover that.
    with DirectoryLock(cfg.data_dir, "dataset preparation"), \
            DirectoryLock(cfg.raw_dir, "raw shard downloads"):
        return _prepare_dataset_locked(cfg)


def _resolve_dataset_revision(cfg: Config, api) -> str:
    """Pin the dataset revision once and keep it for the life of the cache.

    Re-resolving the current HEAD on every run would mean that an upstream push during a
    multi-day job silently invalidates every prepared shard and every checkpoint, turning
    an ordinary restart into a 60 GB re-download and a recipe-mismatch abort. Pinning also
    makes the result reproducible."""
    pin_path = cfg.data_dir / "dataset_revision.json"
    pinned = _read_json(pin_path)

    if cfg.dataset_revision:
        revision = cfg.dataset_revision
        source = "VIMD_DATASET_REVISION"
    elif isinstance(pinned, dict) and pinned.get("dataset_id") == cfg.dataset_id and pinned.get("revision"):
        revision = str(pinned["revision"])
        source = f"pinned by an earlier run in {pin_path.name}"
    else:
        info = _retry(
            lambda: api.dataset_info(cfg.dataset_id),
            f"resolving {cfg.dataset_id}",
            cfg.download_retries,
        )
        revision = info.sha
        source = "resolved from the Hub"

    LOGGER.info("dataset %s at revision %s (%s)", cfg.dataset_id, revision, source)
    if not (isinstance(pinned, dict)
            and pinned.get("dataset_id") == cfg.dataset_id
            and pinned.get("revision") == revision):
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(pin_path, {"dataset_id": cfg.dataset_id, "revision": revision})
    return revision


def _prepare_dataset_locked(cfg: Config) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    revision = _resolve_dataset_revision(cfg, api)
    files = _retry(
        lambda: list(api.list_repo_files(cfg.dataset_id, repo_type="dataset", revision=revision)),
        f"listing {cfg.dataset_id}@{revision}",
        cfg.download_retries,
    )

    by_split: Dict[str, List[Tuple[int, str]]] = {split: [] for split in SPLITS}
    for name in files:
        match = _SHARD_NAME_RE.match(name)
        if match:
            by_split[match.group("split")].append((int(match.group("index")), name))
    for shards in by_split.values():
        shards.sort()

    for split, shards in by_split.items():
        if not shards:
            raise RuntimeError(
                f"no parquet shards found for split '{split}' in {cfg.dataset_id}; "
                f"the dataset layout on the Hub may have changed"
            )
        LOGGER.info("split %-5s -> %d parquet shards on the Hub", split, len(shards))

    for split in SPLITS:
        split_dir = cfg.data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shards = by_split[split]
        # The shard id is the parquet file's own index, so resumption stays correct
        # even if the Hub listing order changes. A shard is reused only when its .done
        # marker matches the current dataset id, revision, sample rate and format.
        pending = [
            (index, name)
            for index, name in shards
            if not _shard_is_current(
                split_dir / f"shard_{index:05d}.done", _shard_identity(cfg, name, revision)
            )
        ]
        stale = sum(1 for index, _ in pending if (split_dir / f"shard_{index:05d}.done").exists())
        if stale:
            LOGGER.warning(
                "%d %s shard(s) were prepared from a different dataset id, revision or format "
                "and will be rebuilt", stale, split,
            )
        # A revision that removes or renumbers shards would otherwise leave orphans
        # behind, and ViMDDataset loads every shard_*.done it finds - mixing data from
        # two revisions into one split.
        expected = {index for index, _ in shards}
        for stray in sorted(split_dir.glob("shard_*.*")):
            try:
                stray_index = int(stray.stem.split("_")[-1])
            except ValueError:
                continue
            if stray_index not in expected:
                LOGGER.warning("removing orphaned shard file %s (not in revision %s)",
                               stray.name, revision)
                stray.unlink(missing_ok=True)

        if not pending:
            LOGGER.info("split %-5s already prepared (%d shards)", split, len(shards))
            continue

        LOGGER.info("split %-5s: %d/%d shards still to convert", split, len(pending), len(shards))
        # Each shard already retries its download `download_retries` times. Sweeping the
        # whole split again afterwards is what rescues a longer outage - a proxy hiccup or
        # a Hub blip lasting minutes - without a human having to re-launch the job.
        failures: List[Tuple[int, str]] = list(pending)
        for sweep in range(1, cfg.prepare_sweeps + 1):
            if not failures:
                break
            if sweep > 1:
                pause = min(600.0, 60.0 * (2 ** (sweep - 2)))
                LOGGER.warning(
                    "%d %s shard(s) failed on sweep %d/%d; pausing %.0fs then retrying only those",
                    len(failures), split, sweep - 1, cfg.prepare_sweeps, pause,
                )
                time.sleep(pause)
            todo, failures = failures, []
            for position, (index, name) in enumerate(todo, start=1):
                started = time.time()
                try:
                    rows, seconds = _convert_one_shard(cfg, name, split_dir, index, revision)
                except BaseException as exc:  # noqa: BLE001
                    LOGGER.error("shard %s failed: %s\n%s", name, exc, traceback.format_exc())
                    failures.append((index, name))
                    continue
                LOGGER.info(
                    "  [%s %d/%d] %s -> %d utterances, %.1f min audio, %.0fs",
                    split, position, len(todo), name, rows, seconds / 60.0, time.time() - started,
                )

        if failures:
            names = ", ".join(name for _, name in failures[:5])
            if split == "train" and cfg.allow_incomplete_train:
                LOGGER.error(
                    "SKIPPING %d train shard(s) (%s) because VIMD_ALLOW_INCOMPLETE_TRAIN is set - "
                    "the reported result will NOT be for the full ViMD training set",
                    len(failures), names,
                )
            else:
                raise RuntimeError(
                    f"{len(failures)} {split} shard(s) could not be prepared after "
                    f"{cfg.prepare_sweeps} sweeps of {cfg.download_retries} attempts each "
                    f"({names}). Refusing to continue with an incomplete {split} split; re-running "
                    f"retries only these shards"
                    + (
                        ". Set VIMD_ALLOW_INCOMPLETE_TRAIN=1 to train on what was prepared instead"
                        if split == "train"
                        else ""
                    )
                )

    if not cfg.keep_raw_parquet:
        _cleanup_raw_dir(cfg.raw_dir)
    LOGGER.info("dataset preparation complete")
    return revision


# ===========================================================================
# 5. Torch dataset over the prepared shards
# ===========================================================================

class ViMDDataset(Dataset):
    """Memory-mapped access to one prepared split."""

    def __init__(
        self,
        split_dir: Path,
        sampling_rate: int,
        expected_dataset: Optional[str] = None,
        expected_revision: Optional[str] = None,
    ) -> None:
        self.split_dir = split_dir
        self.sampling_rate = sampling_rate
        self.records: List[Dict[str, Any]] = []
        self._shard_paths: List[Path] = []
        self._memmaps: Dict[int, np.memmap] = {}

        done_files = sorted(split_dir.glob("shard_*.done"))
        if not done_files:
            raise RuntimeError(f"no prepared shards under {split_dir}")
        for done in done_files:
            # Independent of the pruning in prepare_dataset: a split must never be
            # assembled from two dataset revisions, so verify at the point of use.
            marker = _read_json(done)
            if not isinstance(marker, dict):
                raise RuntimeError(f"{done} is unreadable; delete it and re-run to rebuild")
            for label, expected in (("dataset_id", expected_dataset), ("revision", expected_revision)):
                if expected is not None and marker.get(label) != expected:
                    raise RuntimeError(
                        f"{done.name} was prepared from {label}={marker.get(label)!r} but this run "
                        f"expects {expected!r}. Delete {split_dir} and re-run to rebuild the split."
                    )
            stem = done.with_suffix("")
            jsonl_path = stem.with_suffix(".jsonl")
            bin_path = stem.with_suffix(".bin")
            if not jsonl_path.exists() or not bin_path.exists():
                raise RuntimeError(f"shard {stem.name} is marked done but its files are missing")
            shard_id = len(self._shard_paths)
            self._shard_paths.append(bin_path)
            with open(jsonl_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    record["shard"] = shard_id
                    self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def _memmap(self, shard: int) -> np.memmap:
        mapped = self._memmaps.get(shard)
        if mapped is None:
            mapped = np.memmap(self._shard_paths[shard], dtype="<i2", mode="r")
            self._memmaps[shard] = mapped
        return mapped

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        mapped = self._memmap(record["shard"])
        start = record["offset"] // 2
        pcm = np.asarray(
            mapped[start:start + record["num_samples"]], dtype=np.float32
        ) / 32768.0
        return {"audio": pcm, "text": record["text"], "index": index}

    def select(self, indices: Sequence[int]) -> "ViMDDataset":
        clone = ViMDDataset.__new__(ViMDDataset)
        clone.split_dir = self.split_dir
        clone.sampling_rate = self.sampling_rate
        clone._shard_paths = self._shard_paths
        clone._memmaps = {}
        clone.records = [self.records[i] for i in indices]
        return clone

    def texts(self) -> List[str]:
        return [record["text"] for record in self.records]


@dataclass
class WhisperCollator:
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int
    sampling_rate: int
    max_label_tokens: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.feature_extractor(
            [feature["audio"] for feature in features],
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,  # required for Whisper's SpecAugment masking
            return_tensors="pt",
        )
        encoded = self.tokenizer([feature["text"] for feature in features])
        padded = self.tokenizer.pad({"input_ids": encoded["input_ids"]}, return_tensors="pt")
        labels = padded["input_ids"].masked_fill(padded["attention_mask"].ne(1), -100)
        # The model prepends decoder_start_token_id itself when shifting right.
        if labels.shape[1] > 0 and bool((labels[:, 0] == self.decoder_start_token_id).all()):
            labels = labels[:, 1:]
        if labels.shape[1] > self.max_label_tokens:
            labels = labels[:, : self.max_label_tokens]
        return {
            "input_features": batch["input_features"],
            "attention_mask": torch.as_tensor(batch["attention_mask"]),
            "labels": labels,
        }


def _dataloader_worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


# ===========================================================================
# 6. Model
# ===========================================================================

# Everything the model and processor need; skips only docs and images.
MODEL_FILE_PATTERNS: Tuple[str, ...] = ("*.json", "*.txt", "*.bin", "*.safetensors", "*.model")
# The tokenizer, feature extractor and configs on their own - a few megabytes rather than
# 6.2 GB. Fetching just these makes the processor cheap enough to build and check before
# the data stage instead of after it.
PROCESSOR_FILE_PATTERNS: Tuple[str, ...] = ("*.json", "*.txt")


def resolve_model_snapshot(
    cfg: Config, allow_patterns: Sequence[str] = MODEL_FILE_PATTERNS
) -> Tuple[Path, str]:
    """Download the pretrained model once, at a pinned revision, with retries.

    Two problems solved together. The 6.2 GB download previously happened inside a bare
    from_pretrained call, so a network interruption killed the run outright rather than
    being retried like the dataset shards. And the revision was never pinned, so an
    upstream PhoWhisper push between restarts could change the weights or the tokenizer
    without the recipe guard noticing.

    `allow_patterns` narrows what is fetched. Calling this twice is deliberate and cheap:
    snapshot_download is incremental, and the second call reads the revision back from
    the pin file rather than asking the Hub again."""
    from huggingface_hub import HfApi, snapshot_download

    pin_path = cfg.output_dir / "model_revision.json"
    pinned = _read_json(pin_path)
    if cfg.model_revision:
        revision, source = cfg.model_revision, "VIMD_MODEL_REVISION"
    elif isinstance(pinned, dict) and pinned.get("model_id") == cfg.model_id and pinned.get("revision"):
        revision, source = str(pinned["revision"]), f"pinned in {pin_path.name}"
    else:
        info = _retry(
            lambda: HfApi().model_info(cfg.model_id),
            f"resolving {cfg.model_id}",
            cfg.download_retries,
        )
        revision, source = info.sha, "resolved from the Hub"
    LOGGER.info("model %s at revision %s (%s)", cfg.model_id, revision, source)

    local = Path(
        _retry(
            lambda: snapshot_download(
                cfg.model_id,
                revision=revision,
                allow_patterns=list(allow_patterns),
            ),
            f"downloading {cfg.model_id}@{revision}",
            cfg.download_retries,
        )
    )
    if not (isinstance(pinned, dict)
            and pinned.get("model_id") == cfg.model_id
            and pinned.get("revision") == revision):
        _write_json_atomic(pin_path, {"model_id": cfg.model_id, "revision": revision})
    return local, revision


def _processor_tokenizer_classes() -> List[type]:
    """The tokenizer classes the installed WhisperProcessor will actually accept.

    ProcessorMixin.__init__ isinstance-checks each argument against whatever
    WhisperProcessor declares in `tokenizer_class`, and that declaration moved between
    releases. Every version up to and including 4.53 declares the plain string
    "WhisperTokenizer", so handing it a WhisperTokenizerFast - which descends from
    PreTrainedTokenizerFast, not from WhisperTokenizer - always raises
    `Received a WhisperTokenizerFast for argument tokenizer, but a WhisperTokenizer was
    expected`. From 4.56 the declaration is the tuple ("WhisperTokenizer",
    "WhisperTokenizerFast") and both are allowed. Asking the class what it accepts,
    rather than naming a tokenizer here, is what survives that change and the next one.

    The declared order is kept, so the slow tokenizer wins wherever both are allowed: it
    is the one class every version accepts, which keeps the label ids identical across
    upgrades. Nothing here is tokenizer-bound - one cached pass over the corpus in
    _label_lengths, a handful of short strings per batch in the collator, and
    batch_decode at evaluation - so the fast tokenizer buys no measurable wall-clock."""
    declared = getattr(WhisperProcessor, "tokenizer_class", "WhisperTokenizer")
    names = list(declared) if isinstance(declared, (tuple, list)) else [declared]
    classes: List[type] = []
    for name in names:
        candidate = getattr(transformers, name, None) if isinstance(name, str) else name
        if isinstance(candidate, type) and candidate not in classes:
            classes.append(candidate)
    if not classes:
        raise RuntimeError(
            f"transformers {transformers.__version__} declares "
            f"WhisperProcessor.tokenizer_class={declared!r}, none of which resolves to a "
            f"class in the transformers package. Install the versions pinned in "
            f"requirements.txt."
        )
    return classes


# Short, deliberately accented: it exercises the byte-level BPE path that Vietnamese
# diacritics take through the vocabulary.
TOKENIZER_PROBE = "xin chào các bạn"


def _verify_tokenizer(tokenizer: Any, cfg: Config) -> None:
    """Prove the tokenizer is the one this recipe assumes, in a millisecond.

    Every training label carries the four-token prefix checked below. A tokenizer that
    quietly ignored `language=` or `task=` would teach the model the wrong language tag
    and would only show up days later as a mediocre WER, so it is worth being explicit
    here. The round-trip and the pad() call exercise exactly what WhisperCollator and
    make_compute_metrics do with this object."""
    name = type(tokenizer).__name__
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            f"{name} loaded from {tokenizer.name_or_path} has no pad token, which the "
            f"collator needs to build a rectangular batch of labels"
        )

    unk_id = tokenizer.unk_token_id
    expected_tokens = [
        "<|startoftranscript|>",
        f"<|{cfg.language}|>",
        f"<|{cfg.task}|>",
        "<|notimestamps|>",
    ]
    expected_ids: List[int] = []
    for token in expected_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        # Neither class raises on a token it does not know: both hand back the unk id,
        # which for Whisper is the perfectly valid <|endoftext|>, so compare against it.
        if token_id is None or (unk_id is not None and token_id == unk_id):
            raise RuntimeError(
                f"{name} loaded from {tokenizer.name_or_path} has no {token} token. Either "
                f"Config.language/Config.task ({cfg.language!r}/{cfg.task!r}) name something "
                f"this model was not trained for, or the snapshot's vocabulary is not a "
                f"Whisper one."
            )
        expected_ids.append(int(token_id))

    encoded = list(tokenizer([TOKENIZER_PROBE])["input_ids"][0])
    if encoded[:4] != expected_ids:
        raise RuntimeError(
            f"{name} prefixes labels with "
            f"{tokenizer.convert_ids_to_tokens(encoded[:4])} instead of {expected_tokens}: "
            f"language={cfg.language!r} and task={cfg.task!r} did not take effect. Training "
            f"on these labels would tag every utterance wrongly."
        )
    decoded = tokenizer.batch_decode([encoded], skip_special_tokens=True)[0].strip()
    if decoded != TOKENIZER_PROBE:
        raise RuntimeError(
            f"{name} does not round-trip Vietnamese text: {TOKENIZER_PROBE!r} came back as "
            f"{decoded!r}. The vocabulary files in {tokenizer.name_or_path} look wrong."
        )

    # The exact call WhisperCollator makes on every batch.
    padded = tokenizer.pad({"input_ids": [encoded, encoded[:2]]}, return_tensors="pt")
    expected_shape = (2, len(encoded))
    if tuple(padded["input_ids"].shape) != expected_shape or tuple(
        padded["attention_mask"].shape
    ) != expected_shape:
        raise RuntimeError(
            f"{name}.pad returned {tuple(padded['input_ids'].shape)} for a batch that should "
            f"pad to {expected_shape}; the collator cannot build labels from it"
        )


def load_processor(cfg: Config, model_path: Path) -> WhisperProcessor:
    """Build the processor with a tokenizer this transformers version accepts, and check it.

    Hard-coding WhisperTokenizerFast here is what used to kill the run six hours in, on
    the pinned transformers 4.51.3, which accepts only the slow class. Every accepted
    class is now tried in turn and whatever survives is verified before it can reach the
    trainer."""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))
    failures: List[str] = []
    last_error: Optional[BaseException] = None
    for tokenizer_class in _processor_tokenizer_classes():
        try:
            tokenizer = tokenizer_class.from_pretrained(
                str(model_path), language=cfg.language, task=cfg.task, predict_timestamps=False
            )
            # Not redundant. A fast tokenizer loaded from a tokenizer.json encodes with the
            # post-processor template baked into that file, so `language=` above sets the
            # attribute - tokenizer.language really does read "vi" - while every label
            # still comes out prefixed with whatever was baked, typically a bare
            # <|startoftranscript|><|notimestamps|>. set_prefix_tokens rewrites the
            # template; on the slow tokenizer, which builds its prefix on each call, it is
            # a no-op. Verified against 4.56.2.
            set_prefix_tokens = getattr(tokenizer, "set_prefix_tokens", None)
            if callable(set_prefix_tokens):
                set_prefix_tokens(
                    language=cfg.language, task=cfg.task, predict_timestamps=False
                )
            processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        # Deliberately broad. A missing vocab.json alone gets you a TypeError on 4.51.3
        # that the library then re-raises as an ImportError about protobuf; other versions
        # and other missing files produce OSError, ValueError, KeyError or a JSON decode
        # error. There is another candidate to try and every reason is repeated in the
        # error below, so nothing is swallowed - only postponed.
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{tokenizer_class.__name__}: {type(exc).__name__}: {exc}")
            last_error = exc
            LOGGER.warning(
                "%s cannot be used here (%s: %s); trying the next tokenizer class",
                tokenizer_class.__name__, type(exc).__name__, exc,
            )
            continue
        # Deliberately outside the try: a tokenizer that loads but is wrong is a reason to
        # stop, not a reason to quietly try another class against the same files.
        _verify_tokenizer(processor.tokenizer, cfg)
        LOGGER.info(
            "tokenizer %s (%d tokens), language=%s task=%s, no timestamps",
            type(processor.tokenizer).__name__, len(processor.tokenizer), cfg.language, cfg.task,
        )
        return processor
    raise RuntimeError(
        "could not build a WhisperProcessor from {path} with transformers {version}:\n"
        "  {attempts}\n"
        "The slow WhisperTokenizer is built from vocab.json and merges.txt, the fast one "
        "from tokenizer.json, and WhisperProcessor only accepts the fast class from "
        "transformers 4.56 onwards. Check which of those files the snapshot actually "
        "contains.".format(
            path=model_path,
            version=transformers.__version__,
            attempts="\n  ".join(failures) or "(no tokenizer class was even attempted)",
        )
    ) from last_error


def _convert_bin_to_safetensors(cfg: Config, source: Path) -> Path:
    """Fallback for transformers >= 4.52, which refuses to torch.load a .bin unless
    torch >= 2.6. PhoWhisper publishes only pytorch_model.bin, so convert it once."""
    from safetensors.torch import save_file

    # Keyed by model id: a shared directory would hand back another model's weights
    # if VIMD_MODEL_ID were ever changed.
    target = Path(os.environ["HF_HOME"]) / "safetensors_conversions" / _slugify(cfg.model_id)
    # The marker is written last, so a conversion killed while writing the 6 GB weights
    # is redone rather than reused as a truncated file.
    marker = target / ".conversion_complete"
    if marker.exists() and (target / "model.safetensors").exists():
        return target
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    LOGGER.warning("converting %s weights to safetensors under %s", cfg.model_id, target)
    target.mkdir(parents=True, exist_ok=True)
    state = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # clone() breaks the tied embedding/output weights, which safetensors rejects.
    tensors = {
        key: value.contiguous().clone()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    del state
    for pattern in ("*.json", "*.txt"):
        for extra in source.glob(pattern):
            shutil.copy2(extra, target / extra.name)
    weights_tmp = target / "model.safetensors.tmp"
    save_file(tensors, str(weights_tmp), metadata={"format": "pt"})
    del tensors
    gc.collect()
    os.replace(weights_tmp, target / "model.safetensors")
    _write_json_atomic(marker, {"model_id": cfg.model_id, "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    return target


def load_model(
    cfg: Config, model_path: Path, source: Optional[str] = None
) -> WhisperForConditionalGeneration:
    """Load either the pretrained snapshot (source=None) or a local checkpoint."""
    path = source or str(model_path)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            path, low_cpu_mem_usage=True, torch_dtype=torch.float32
        )
    except ValueError as exc:
        if source is not None or "torch" not in str(exc).lower():
            raise
        LOGGER.warning("direct load of %s failed (%s); converting to safetensors", path, exc)
        model = WhisperForConditionalGeneration.from_pretrained(
            str(_convert_bin_to_safetensors(cfg, model_path)),
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
    return configure_model(model, cfg)


def configure_model(
    model: WhisperForConditionalGeneration, cfg: Config
) -> WhisperForConditionalGeneration:
    """Pin the decoding task and enable SpecAugment.

    PhoWhisper's generation_config carries forced_decoder_ids [[1, None], [2, 50359]],
    which leaves the language slot to be auto-detected and conflicts with the modern
    language/task API. Both the config and the generation config copies must be cleared.
    """
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task
    model.generation_config.return_timestamps = False
    model.generation_config.num_beams = 1
    model.generation_config.use_cache = True
    model.generation_config.max_length = cfg.max_label_tokens

    model.config.apply_spec_augment = cfg.apply_spec_augment
    model.config.mask_time_prob = cfg.mask_time_prob
    model.config.mask_time_length = cfg.mask_time_length
    model.config.mask_feature_prob = cfg.mask_feature_prob
    model.config.mask_feature_length = cfg.mask_feature_length

    model.config.use_cache = False  # required alongside gradient checkpointing
    return model


# ===========================================================================
# 7. Trainer plumbing
# ===========================================================================

class AutocastSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer.prediction_step calls model.generate() outside any autocast
    context, so generation-based evaluation would otherwise run the fp32 weights.
    Wrapping it roughly halves eval wall-clock and KV-cache memory."""

    amp_dtype = torch.bfloat16

    def prediction_step(self, *args, **kwargs):
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                return super().prediction_step(*args, **kwargs)
        return super().prediction_step(*args, **kwargs)


class MetricsHistoryCallback(TrainerCallback):
    """Persist the WER at every evaluation so an interrupted run keeps its history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.history: List[Dict[str, Any]] = []
        loaded = _read_json(path)
        if isinstance(loaded, list):
            self.history = loaded

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D102
        if not metrics:
            return
        entry: Dict[str, Any] = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
        }
        entry.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        self.history.append(entry)
        _write_json_atomic(self.path, self.history)
        LOGGER.info(
            "eval @ step %d: wer_norm=%.4f wer_raw=%.4f cer_norm=%.4f",
            entry["step"],
            metrics.get("eval_wer_norm", float("nan")),
            metrics.get("eval_wer_raw", float("nan")),
            metrics.get("eval_cer_norm", float("nan")),
        )


def make_compute_metrics(tokenizer, pad_token_id: int, expand_numbers: bool):
    def compute_metrics(eval_prediction) -> Dict[str, float]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        labels = np.asarray(eval_prediction.label_ids)
        # Trainer pads both generated ids and labels with -100 when concatenating batches.
        predictions = np.where(predictions < 0, pad_token_id, predictions)
        labels = np.where(labels < 0, pad_token_id, labels)

        hyps = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        report = compute_wer_report(refs, hyps, expand_numbers)
        return {
            "wer_norm": report.wer_norm,
            "wer_raw": report.wer_raw,
            "cer_norm": report.cer_norm,
            "skipped_empty_refs": float(report.num_skipped_empty_refs),
        }

    return compute_metrics


def build_seq2seq_trainer(
    cfg: Config,
    model,
    processor,
    train_dataset,
    eval_dataset,
    collator,
    compute_metrics,
    callbacks,
    train_batch_size: int,
    grad_accum: int,
    eval_batch_size: int,
    generation_max_length: int,
    use_bf16: bool,
    tf32: bool,
) -> AutocastSeq2SeqTrainer:
    args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.checkpoint_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=tf32,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,  # must equal eval_steps for load_best_model_at_end
        save_total_limit=cfg.save_total_limit,
        save_safetensors=True,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        logging_dir=str(cfg.log_dir / "hf"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer_norm",
        greater_is_better=False,
        # Defaults to False, which would reset the early-stopping patience counter on
        # every resume and let a plateaued run keep training.
        restore_callback_states_from_checkpoint=True,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_num_beams=1,  # greedy during training keeps the eval loop affordable
        dataloader_num_workers=cfg.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.dataloader_workers > 0,
        dataloader_prefetch_factor=2 if cfg.dataloader_workers > 0 else None,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],  # the default "all" would start wandb and block on a login prompt
        disable_tqdm=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        push_to_hub=False,
    )

    return AutocastSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )


def _is_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ===========================================================================
# 8. Standalone decoding (checkpoint re-ranking and the final test pass)
# ===========================================================================

def transcribe_dataset(
    model,
    dataset: ViMDDataset,
    collator: WhisperCollator,
    tokenizer,
    cfg: Config,
    num_beams: int,
    max_length: int,
    batch_size: int,
    use_bf16: bool,
    description: str,
    on_progress: Optional[Any] = None,
    resume_from: Optional[List[str]] = None,
) -> List[str]:
    """Decode a split in dataset order, halving the batch size on CUDA OOM.

    Written explicitly rather than via Trainer.predict so the mapping from hypothesis to
    metadata row is index-for-index and obvious.

    `resume_from` supplies hypotheses already decoded for the leading utterances, and
    `on_progress` is called periodically with the full prefix so far, which together let
    a crash part-way through a long beam-search pass pick up where it stopped instead of
    starting the split again."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model.to(device)
    model.eval()

    done: List[str] = list(resume_from or [])
    if len(done) > len(dataset):
        done = []
    if done:
        LOGGER.info("  %s: resuming after %d/%d already decoded", description, len(done), len(dataset))
    if len(done) == len(dataset):
        return done
    remaining = dataset.select(range(len(done), len(dataset)))

    attempt_batch = max(1, batch_size)
    for attempt in range(cfg.oom_retries + 1):
        loader = DataLoader(
            remaining,
            batch_size=attempt_batch,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.dataloader_workers,
            pin_memory=True,
            worker_init_fn=_dataloader_worker_init,
        )
        hypotheses: List[str] = []
        started = time.time()
        try:
            with torch.no_grad():
                for step, batch in enumerate(loader, start=1):
                    features = batch["input_features"].to(device, non_blocking=True)
                    attention = batch["attention_mask"].to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        generated = model.generate(
                            input_features=features,
                            attention_mask=attention,
                            num_beams=num_beams,
                            max_length=max_length,
                            language=cfg.language,
                            task=cfg.task,
                            return_timestamps=False,
                        )
                    hypotheses.extend(
                        text.strip()
                        for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
                    )
                    if step % 20 == 0:
                        rate = len(hypotheses) / max(time.time() - started, 1e-6)
                        LOGGER.info(
                            "  %s: %d/%d utterances (%.1f/s)",
                            description, len(done) + len(hypotheses), len(dataset), rate,
                        )
                        if on_progress is not None:
                            on_progress(done + hypotheses)
            if len(hypotheses) != len(remaining):
                raise RuntimeError(
                    f"decoded {len(hypotheses)} hypotheses for {len(remaining)} utterances"
                )
            LOGGER.info(
                "  %s: done, %d utterances in %.0fs", description, len(hypotheses),
                time.time() - started,
            )
            return done + hypotheses
        except BaseException as exc:  # noqa: BLE001
            if not _is_oom(exc) or attempt >= cfg.oom_retries or attempt_batch == 1:
                raise
            _free_cuda()
            attempt_batch = max(1, attempt_batch // 2)
            LOGGER.warning("OOM while decoding; retrying with batch size %d", attempt_batch)
    raise RuntimeError("decoding failed")


# ===========================================================================
# 9. Output writers
# ===========================================================================

CSV_COLUMNS = (
    "index",
    "filename",
    "region",
    "province_code",
    "province_name",
    "speakerID",
    "gender",
    "duration",
    "reference",
    "prediction",
    "reference_normalized",
    "prediction_normalized",
    "wer_norm",
)


def write_predictions_csv(
    path: Path,
    dataset: ViMDDataset,
    hypotheses: Sequence[str],
    report: WerReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Vietnamese diacritics survive a double-click into Excel.
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for index, record in enumerate(dataset.records):
            per_utterance = report.per_utterance_wer[index]
            writer.writerow(
                {
                    "index": index,
                    "filename": record.get("filename", ""),
                    "region": record.get("region", ""),
                    "province_code": record.get("province_code", ""),
                    "province_name": record.get("province_name", ""),
                    "speakerID": record.get("speakerID", ""),
                    "gender": record.get("gender", ""),
                    "duration": f"{float(record.get('duration', 0.0)):.3f}",
                    "reference": record.get("text", ""),
                    "prediction": hypotheses[index],
                    "reference_normalized": report.normalized_refs[index],
                    "prediction_normalized": report.normalized_hyps[index],
                    "wer_norm": "" if per_utterance is None else f"{per_utterance:.4f}",
                }
            )
    LOGGER.info("wrote %s (%d rows)", path, len(dataset.records))


def split_summary(dataset: ViMDDataset, cfg: Config) -> Dict[str, Any]:
    durations = [float(record["duration"]) for record in dataset.records]
    return {
        "num_utterances": len(dataset),
        "hours": round(sum(durations) / 3600.0, 3),
        "min_duration_s": round(min(durations), 3) if durations else 0.0,
        "max_duration_s": round(max(durations), 3) if durations else 0.0,
        "num_utterances_over_30s_truncated_by_whisper": sum(
            1 for d in durations if d > cfg.max_audio_seconds
        ),
        "num_speakers": len({record.get("speakerID") for record in dataset.records}),
        "num_provinces": len({record.get("province_name") for record in dataset.records}),
    }


# ===========================================================================
# 10. Preflight
# ===========================================================================

def _total_ram_gb() -> Optional[float]:
    """Total system RAM, or None where it cannot be determined. Uses only the stdlib,
    so this adds no dependency."""
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except (AttributeError, ValueError, OSError):
        return None


def _tf32_available() -> bool:
    """TrainingArguments(tf32=True) raises outright on pre-Ampere hardware, which would
    defeat the fp16 fallback, so the flag is only ever set when it is legal."""
    try:
        from transformers.utils import is_torch_tf32_available

        return bool(is_torch_tf32_available())
    except Exception:  # pragma: no cover - helper moved between versions
        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)


def _writable(directory: Path) -> bool:
    """os.access lies on NFS and read-only bind mounts; actually write something."""
    probe = directory / f".write_probe_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _free_gb_by_volume(paths: Sequence[Path]) -> Dict[Path, float]:
    """Free space per distinct filesystem, so paths on separate volumes are all checked
    and paths sharing one volume are not counted twice."""
    seen: Dict[int, Tuple[Path, float]] = {}
    for path in paths:
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            device = os.stat(probe).st_dev
            free = shutil.disk_usage(str(probe)).free / (1024 ** 3)
        except OSError:
            continue
        seen.setdefault(device, (path, free))
    return {path: free for path, free in seen.values()}


def preflight(cfg: Config) -> Dict[str, Any]:
    banner("PhoWhisper x ViMD - Vietnamese multi-dialect ASR fine-tuning")

    if sys.version_info < (3, 9):
        raise RuntimeError(f"Python 3.9+ is required, found {platform.python_version()}")

    cfg.validate()

    hf_home = Path(os.environ.get("HF_HOME", str(ROOT / ".hf_cache")))
    for directory in (cfg.data_dir, cfg.raw_dir, cfg.output_dir, cfg.checkpoint_dir,
                      cfg.log_dir, hf_home):
        directory.mkdir(parents=True, exist_ok=True)
        if not _writable(directory):
            raise RuntimeError(f"{directory} is not writable by this process")

    # These four may live on different filesystems; every distinct volume is checked.
    volumes = _free_gb_by_volume([cfg.output_dir, cfg.data_dir, cfg.raw_dir, hf_home])
    if not volumes:
        raise RuntimeError(
            f"could not determine free space for any of {cfg.output_dir}, {cfg.data_dir}, "
            f"{cfg.raw_dir}, {hf_home}"
        )
    free_gb = min(volumes.values())
    for path, free in sorted(volumes.items()):
        LOGGER.info("%-22s %.1f GB free", f"disk[{path.name or path}]", free)
    if free_gb < cfg.min_free_gb:
        tight = min(volumes, key=lambda p: volumes[p])
        raise RuntimeError(
            f"only {free_gb:.1f} GB free on the volume holding {tight}. This job needs roughly "
            f"12 GB for the 16 kHz audio cache, ~19 GB per retained checkpoint "
            f"({cfg.save_total_limit} retained), ~7 GB for the model and ~1 GB of transient raw "
            f"shards. Free space, lower VIMD_SAVE_TOTAL_LIMIT, or point VIMD_OUTPUT_DIR / "
            f"VIMD_DATA_DIR / VIMD_RAW_DIR / HF_HOME at a larger volume."
        )
    if free_gb < cfg.recommended_free_gb:
        LOGGER.warning(
            "only %.1f GB free; %.0f GB is recommended for a comfortable run",
            free_gb, cfg.recommended_free_gb,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device visible. Fine-tuning PhoWhisper-large requires a GPU; check "
            "nvidia-smi and that torch was installed from the CUDA wheel index."
        )
    try:
        # Surfaces a driver/runtime mismatch here rather than hours into the run.
        torch.zeros(1, device="cuda").add_(1).cpu()
    except BaseException as exc:  # noqa: BLE001
        raise RuntimeError(
            f"CUDA is visible but unusable ({exc}). This is normally a driver/runtime mismatch: "
            f"torch was built for CUDA {torch.version.cuda}, so the NVIDIA driver must support "
            f"at least that major version. Check nvidia-smi."
        ) from exc

    capability = torch.cuda.get_device_capability(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_vram < cfg.min_vram_gb:
        LOGGER.warning(
            "GPU has %.1f GB VRAM, below the %.0f GB this recipe is tuned for. Training will "
            "still start and back off on OOM, but consider VIMD_TRAIN_BATCH_SIZE=8 "
            "VIMD_GRAD_ACCUM=8 to keep the effective batch at 64.",
            total_vram, cfg.min_vram_gb,
        )

    ram_gb = _total_ram_gb()
    if ram_gb is not None and ram_gb < cfg.min_ram_gb:
        LOGGER.warning(
            "only %.1f GB of system RAM detected; loading the 6.2 GB fp32 checkpoint needs "
            "roughly 13 GB of headroom. Lower VIMD_DATALOADER_WORKERS if the run is killed.",
            ram_gb,
        )

    use_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
        ),
        "bf16_supported": use_bf16,
        "tf32_available": _tf32_available(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "free_disk_gb": round(free_gb, 1),
        "hf_home": os.environ.get("HF_HOME", ""),
    }
    for key, value in environment.items():
        LOGGER.info("%-22s %s", key, value)
    if not use_bf16:
        LOGGER.warning("bf16 unsupported on this GPU; falling back to fp16 mixed precision")

    if environment["tf32_available"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return environment


# ===========================================================================
# 11. Pipeline
# ===========================================================================

def _label_lengths(tokenizer, texts: Sequence[str], cache_path: Path) -> List[int]:
    """Tokenised length of every reference, cached.

    Keyed on a digest of the texts and the tokenizer identity, not just the utterance
    count: these lengths decide which training utterances are dropped and how long the
    decode budget is, so a same-sized but different dataset - or the same dataset under
    a different tokenizer - must not be answered from the old cache."""
    key = {"texts": _digest(list(texts)), "tokenizer": f"{tokenizer.name_or_path}:{len(tokenizer)}"}
    cached = _read_json(cache_path)
    if isinstance(cached, dict) and cached.get("key") == key:
        lengths = cached.get("lengths")
        if isinstance(lengths, list) and len(lengths) == len(texts):
            return lengths
    lengths = []
    for start in range(0, len(texts), 512):
        chunk = list(texts[start:start + 512])
        lengths.extend(len(ids) for ids in tokenizer(chunk)["input_ids"])
    _write_json_atomic(cache_path, {"key": key, "lengths": lengths})
    return lengths


def _checkpoint_has_weights(path: Path) -> bool:
    return any(path.glob("model*.safetensors")) or (path / "pytorch_model.bin").exists()


def _checkpoint_is_complete(path: Path) -> bool:
    """Usable for *scoring*: weights present and a trainer_state.json that actually parses.

    Trainer writes the weights first and trainer_state.json last, so weights without a
    state file mean a save killed halfway through. Existence alone is not enough either -
    the state file itself can be truncated by the same kill - so it is parsed and checked
    for the field Trainer always writes."""
    if not path.is_dir() or not _checkpoint_has_weights(path):
        return False
    state = _read_json(path / "trainer_state.json")
    return isinstance(state, dict) and isinstance(state.get("global_step"), int)


def _checkpoint_is_resumable(path: Path) -> bool:
    """Usable for *resuming*: also needs the optimiser and scheduler state.

    Resuming without them would silently restart Adam's moments and the LR schedule
    mid-run, which is a different (and worse) recipe than the one being reported."""
    if not _checkpoint_is_complete(path):
        return False
    missing = [name for name in ("optimizer.pt", "scheduler.pt") if not (path / name).exists()]
    if missing:
        LOGGER.warning("checkpoint %s is missing %s; not resumable", path.name, ", ".join(missing))
        return False
    return True


def _sorted_checkpoints(checkpoint_dir: Path, newest_first: bool = False) -> List[Path]:
    if not checkpoint_dir.exists():
        return []
    return sorted(
        (p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),  # numeric, so 1000 sorts after 500
        reverse=newest_first,
    )


def _validated_last_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """Newest resumable checkpoint, stepping back past any interrupted save."""
    for candidate in _sorted_checkpoints(checkpoint_dir, newest_first=True):
        if _checkpoint_is_resumable(candidate):
            return str(candidate)
        LOGGER.warning("ignoring unusable checkpoint %s", candidate)
    return None


# --- decode caches ---------------------------------------------------------
# Beam-decoding a split costs the better part of an hour, so hypotheses are cached.
# A cache is only ever reused when the full decoding configuration matches, and each
# entry additionally carries the fingerprint of the exact weights that produced it:
# "checkpoint-1500" in a reused output directory can otherwise mean a different model,
# a different dataset, or a different beam count.

def _load_decode_cache(path: Path, signature: Dict[str, Any]) -> Dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {}
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        LOGGER.warning(
            "discarding %s: it was written for a different model, dataset or decoding config",
            path.name,
        )
        return {}
    entries = payload.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_decode_cache(path: Path, signature: Dict[str, Any], entries: Dict[str, Any]) -> None:
    _write_json_atomic(path, {"signature": signature, "entries": entries})


def _cached_prefix(entries: Dict[str, Any], key: str, fingerprint: str, expected: int) -> List[str]:
    """Hypotheses already decoded for the leading utterances, complete or not.

    Progress is checkpointed during a decode, so a crash part-way through a long beam
    pass resumes from here instead of repeating the whole split."""
    entry = entries.get(key)
    if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
        return []
    hypotheses = entry.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) > expected:
        return []
    if not all(isinstance(item, str) for item in hypotheses):
        return []
    return hypotheses


def _cached_hypotheses(
    entries: Dict[str, Any], key: str, fingerprint: str, expected: int
) -> Optional[List[str]]:
    entry = entries.get(key)
    if not isinstance(entry, dict) or not entry.get("complete"):
        return None
    prefix = _cached_prefix(entries, key, fingerprint, expected)
    return prefix if len(prefix) == expected else None


def _decode_with_cache(
    cache: Dict[str, Any],
    cache_path: Path,
    signature: Dict[str, Any],
    key: str,
    fingerprint: str,
    dataset: "ViMDDataset",
    decode: Any,
) -> List[str]:
    """Return cached hypotheses when they are complete, otherwise decode the remainder
    and keep the cache up to date as it goes."""
    complete = _cached_hypotheses(cache, key, fingerprint, len(dataset))
    if complete is not None:
        LOGGER.info("  %s: reusing cached decode", key)
        return complete

    def store(prefix: List[str], done: bool = False) -> None:
        cache[key] = {"fingerprint": fingerprint, "hypotheses": prefix, "complete": done}
        _save_decode_cache(cache_path, signature, cache)

    hypotheses = decode(
        on_progress=store,
        resume_from=_cached_prefix(cache, key, fingerprint, len(dataset)),
    )
    store(hypotheses, done=True)
    return hypotheses


def _checkpoint_candidates(checkpoint_dir: Path, best: Optional[str]) -> List[Path]:
    found: Dict[str, Path] = {}
    if best and _checkpoint_is_complete(Path(best)):
        found[Path(best).name] = Path(best)
    for candidate in _sorted_checkpoints(checkpoint_dir):
        if _checkpoint_is_complete(candidate):
            found[candidate.name] = candidate
        else:
            LOGGER.warning("excluding incomplete checkpoint %s from selection", candidate)
    return list(found.values())


def run(cfg: Config) -> int:
    environment = preflight(cfg)
    use_bf16 = bool(environment["bf16_supported"])
    if not use_bf16:
        AutocastSeq2SeqTrainer.amp_dtype = torch.float16

    set_seed(cfg.seed)

    # The processor is built here rather than beside the weights in stage 2. The files it
    # needs are a few megabytes and checking it takes a millisecond, but getting it wrong
    # used to surface only once the six-hour audio conversion had finished. Same bargain
    # as the CUDA smoke test in preflight: pay a second now, not a night.
    processor_path, _ = resolve_model_snapshot(cfg, PROCESSOR_FILE_PATTERNS)
    processor = load_processor(cfg, processor_path)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor

    # ---- stage 1: data --------------------------------------------------
    dataset_revision = prepare_dataset(cfg)

    # ---- stage 2: index -------------------------------------------------
    banner("Stage 2/5 - model download, indexing and label checks")
    # The processor pulled the small files already; this adds the weights beside them.
    model_path, model_revision = resolve_model_snapshot(cfg)

    splits: Dict[str, ViMDDataset] = {
        split: ViMDDataset(
            cfg.data_dir / split, cfg.sampling_rate, cfg.dataset_id, dataset_revision
        )
        for split in SPLITS
    }
    summaries = {split: split_summary(dataset, cfg) for split, dataset in splits.items()}
    for split, summary in summaries.items():
        LOGGER.info("%-5s %s", split, json.dumps(summary, ensure_ascii=False))

    label_lengths: Dict[str, List[int]] = {}
    for split, dataset in splits.items():
        lengths = _label_lengths(
            tokenizer, dataset.texts(), cfg.data_dir / f"{split}_label_lengths.json"
        )
        label_lengths[split] = lengths
        too_long = sum(1 for length in lengths if length > cfg.max_label_tokens)
        LOGGER.info(
            "%-5s label tokens: max=%d mean=%.1f; %d over the %d-token decoder limit",
            split,
            max(lengths) if lengths else 0,
            (sum(lengths) / len(lengths)) if lengths else 0.0,
            too_long,
            cfg.max_label_tokens,
        )
        if too_long and split != "train":
            LOGGER.warning(
                "%d %s references exceed %d tokens; they are truncated in the teacher-forced "
                "loss only - the reported WER always uses the full reference text",
                too_long, split, cfg.max_label_tokens,
            )

    # Filtering applies to the training split only. Dropping an evaluation utterance
    # would quietly change what the reported WER measures.
    keep = [
        index
        for index, length in enumerate(label_lengths["train"])
        if length <= cfg.max_label_tokens and splits["train"].records[index]["text"].strip()
    ]
    dropped = len(splits["train"]) - len(keep)
    if dropped:
        LOGGER.warning(
            "dropping %d training utterances (empty text or label above %d tokens)",
            dropped, cfg.max_label_tokens,
        )
    train_dataset = splits["train"].select(keep)
    valid_dataset = splits["valid"]
    test_dataset = splits["test"]

    # A train shard that failed on an earlier run and succeeded on this one changes
    # the number of optimiser steps per epoch, which no longer matches the LR schedule
    # stored in an existing checkpoint. Silent, so say it out loud.
    fingerprint_path = cfg.output_dir / "dataset_fingerprint.json"
    fingerprint = {
        "dataset_id": cfg.dataset_id,
        "revision": dataset_revision,
        "sampling_rate": cfg.sampling_rate,
        "shard_format": SHARD_FORMAT_VERSION,
        "utterances": {split: summary["num_utterances"] for split, summary in summaries.items()},
        "train_used": len(train_dataset),
        "text_digest": {
            split: _digest(dataset.texts()) for split, dataset in splits.items()
        },
    }
    previous = _read_json(fingerprint_path)
    if isinstance(previous, dict) and previous != fingerprint:
        LOGGER.warning(
            "the dataset changed since the last run in this output directory:\n  before: %s\n"
            "  now:    %s\nExisting checkpoints were trained on different data or a different "
            "number of steps per epoch. For a clean run, delete %s.",
            json.dumps(previous, ensure_ascii=False),
            json.dumps(fingerprint, ensure_ascii=False),
            cfg.checkpoint_dir,
        )
    _write_json_atomic(fingerprint_path, fingerprint)

    # The recipe is only fully known once the data is indexed, which is why the
    # completion check lives here rather than at the top of the run: a finished run
    # spends a couple of minutes re-verifying shards before exiting, and in exchange a
    # changed configuration can never be answered with a previous experiment's result.
    recipe = recipe_fingerprint(cfg, fingerprint, model_revision)
    _write_json_atomic(cfg.output_dir / "run_recipe.json", recipe)

    if cfg.results_path.exists():
        existing = _read_json(cfg.results_path)
        test_wer = (existing or {}).get("test", {}).get("wer_normalized") if existing else None
        if isinstance(test_wer, (int, float)):
            previous_recipe = (existing or {}).get("recipe")
            if previous_recipe == recipe:
                LOGGER.info(
                    "%s already holds a complete result for this exact recipe "
                    "(test wer_normalized=%.4f) - nothing to do", cfg.results_path, test_wer,
                )
                return 0
            raise RuntimeError(
                f"{cfg.results_path} holds a finished result produced by a DIFFERENT "
                f"configuration:\n"
                f"{_describe_recipe_change(previous_recipe or {}, recipe)}\n"
                f"Returning that number as this run's result would be wrong, and overwriting it "
                f"would destroy the earlier experiment. Point VIMD_OUTPUT_DIR at a new directory "
                f"for this configuration, or move the existing outputs aside."
            )
        damaged = cfg.results_path.with_name(cfg.results_path.name + ".damaged")
        LOGGER.warning(
            "%s exists but is empty, truncated or missing a test score; moving it to %s and "
            "re-running", cfg.results_path, damaged,
        )
        os.replace(cfg.results_path, damaged)

    # Checkpoints carry optimiser state and weights for one specific recipe. Resuming
    # them under a changed model, learning rate, schedule or seed would silently produce
    # a model that matches neither configuration.
    recipe_marker = cfg.checkpoint_dir / "recipe.json"
    stored_recipe = _read_json(recipe_marker)
    existing_checkpoints = [
        path for path in _sorted_checkpoints(cfg.checkpoint_dir) if _checkpoint_has_weights(path)
    ]
    if isinstance(stored_recipe, dict):
        if stored_recipe != recipe:
            if _validated_last_checkpoint(cfg.checkpoint_dir):
                raise RuntimeError(
                    f"{cfg.checkpoint_dir} holds checkpoints trained with a different recipe:\n"
                    f"{_describe_recipe_change(stored_recipe, recipe)}\n"
                    f"Resuming them would mix two experiments. Use a fresh VIMD_OUTPUT_DIR, or "
                    f"delete {cfg.checkpoint_dir} to start this configuration from scratch."
                )
            LOGGER.warning("recipe changed since the last run, but no checkpoint survives it; "
                           "starting fresh")
    elif existing_checkpoints:
        # Checkpoints with no readable recipe marker cannot be shown to belong to this
        # configuration. Writing the current recipe over the gap would launder someone
        # else's weights into this experiment.
        raise RuntimeError(
            f"{cfg.checkpoint_dir} holds {len(existing_checkpoints)} checkpoint(s) but "
            f"{recipe_marker.name} is missing or unreadable, so they cannot be shown to belong "
            f"to this configuration. Delete {cfg.checkpoint_dir} to train this recipe from "
            f"scratch, or restore the file if you know the checkpoints match."
        )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(recipe_marker, recipe)

    # Two decode budgets. The periodic evaluations use a train-derived cap purely for
    # speed; deriving it from train only keeps validation and test out of the decision.
    # The scores that get reported come from the final passes, which use the model's
    # own 448-token limit so a long valid/test hypothesis is never cut short.
    train_eval_max_length = int(
        max(64, min(cfg.max_label_tokens, (max(label_lengths["train"]) if label_lengths["train"] else 0) + 32))
    )
    final_max_length = cfg.max_label_tokens
    LOGGER.info(
        "generation max length: %d during training evals, %d for selection and test",
        train_eval_max_length, final_max_length,
    )

    decoder_start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if decoder_start_token_id is None:
        raise RuntimeError("tokenizer has no <|startoftranscript|> token")
    collator = WhisperCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder_start_token_id=int(decoder_start_token_id),
        sampling_rate=cfg.sampling_rate,
        max_label_tokens=cfg.max_label_tokens,
    )
    compute_metrics = make_compute_metrics(tokenizer, int(tokenizer.pad_token_id), cfg.expand_numbers)

    # ---- stage 3: training ----------------------------------------------
    banner("Stage 3/5 - fine-tuning")
    train_batch_size = cfg.train_batch_size
    grad_accum = cfg.grad_accum
    eval_batch_size = cfg.eval_batch_size
    effective_batch = train_batch_size * grad_accum
    trainer: Optional[AutocastSeq2SeqTrainer] = None
    model: Optional[WhisperForConditionalGeneration] = None

    for attempt in range(cfg.oom_retries + 1):
        model = load_model(cfg, model_path)
        trainer = build_seq2seq_trainer(
            cfg=cfg,
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[
                MetricsHistoryCallback(cfg.output_dir / "metrics_history.json"),
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.early_stopping_patience,
                    early_stopping_threshold=cfg.early_stopping_threshold,
                ),
            ],
            train_batch_size=train_batch_size,
            grad_accum=grad_accum,
            eval_batch_size=eval_batch_size,
            generation_max_length=train_eval_max_length,
            use_bf16=use_bf16,
            tf32=bool(environment["tf32_available"]),
        )
        resume_from = _validated_last_checkpoint(cfg.checkpoint_dir)
        LOGGER.info(
            "batch %d x accum %d (effective %d), eval batch %d, resume from %s",
            train_batch_size, grad_accum, train_batch_size * grad_accum, eval_batch_size,
            resume_from or "scratch",
        )
        try:
            trainer.train(resume_from_checkpoint=resume_from)
            break
        except BaseException as exc:  # noqa: BLE001
            if not _is_oom(exc) or attempt >= cfg.oom_retries or train_batch_size == 1:
                raise
            LOGGER.error("CUDA OOM during training: %s", exc)
            del trainer, model
            trainer, model = None, None
            _free_cuda()
            train_batch_size = max(1, train_batch_size // 2)
            # Keep the effective batch constant so the recipe itself does not change.
            grad_accum = max(1, effective_batch // train_batch_size)
            eval_batch_size = max(1, eval_batch_size // 2)
            LOGGER.warning(
                "retrying with batch %d x accum %d, eval batch %d",
                train_batch_size, grad_accum, eval_batch_size,
            )

    if trainer is None:
        raise RuntimeError("training did not run")

    best_checkpoint = trainer.state.best_model_checkpoint
    best_greedy_wer = trainer.state.best_metric
    LOGGER.info(
        "training finished at step %d; best greedy validation wer_norm=%.4f (%s)",
        trainer.state.global_step,
        best_greedy_wer if best_greedy_wer is not None else float("nan"),
        best_checkpoint,
    )
    trainer.save_state()
    candidates = _checkpoint_candidates(cfg.checkpoint_dir, best_checkpoint)
    log_history = list(trainer.state.log_history)

    # Release optimiser, gradients and the training model before beam search.
    try:
        trainer.accelerator.free_memory()
    except Exception:  # pragma: no cover - accelerate internals differ across versions
        pass
    trainer.optimizer = None
    trainer.lr_scheduler = None
    del trainer, model
    _free_cuda()

    # ---- stage 4: checkpoint selection ----------------------------------
    banner("Stage 4/5 - checkpoint selection on validation (beam search)")
    if not candidates:
        raise RuntimeError(
            f"no usable checkpoints under {cfg.checkpoint_dir}; training produced nothing to score"
        )
    LOGGER.info("re-ranking %d checkpoint(s) with num_beams=%d", len(candidates), cfg.final_num_beams)

    # Everything that changes what a decode produces. expand_numbers is deliberately
    # absent: it affects scoring only, and scores are always recomputed from the cached
    # hypotheses, so toggling it must not throw away an hour of valid decoding.
    decode_signature = {
        "model_id": cfg.model_id,
        "model_revision": model_revision,
        "dataset": fingerprint,
        "language": cfg.language,
        "task": cfg.task,
        "num_beams": cfg.final_num_beams,
        "max_length": final_max_length,
        "tokenizer": f"{tokenizer.name_or_path}:{len(tokenizer)}",
        "bf16": use_bf16,
    }
    cache_path = cfg.output_dir / "valid_rerank_cache.json"
    cache = _load_decode_cache(cache_path, decode_signature)
    if cache:
        LOGGER.info("found cached validation decodes for %d checkpoint(s)", len(cache))

    reranking: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_entry: Optional[Dict[str, Any]] = None
    best_report: Optional[WerReport] = None
    best_hyps: Optional[List[str]] = None

    for candidate in candidates:
        weights_fingerprint = _checkpoint_fingerprint(candidate)
        scored_model: Optional[WhisperForConditionalGeneration] = None

        def decode(on_progress, resume_from, _candidate=candidate):
            nonlocal scored_model
            if scored_model is None:
                scored_model = load_model(cfg, model_path, source=str(_candidate))
            return transcribe_dataset(
                model=scored_model,
                dataset=valid_dataset,
                collator=collator,
                tokenizer=tokenizer,
                cfg=cfg,
                num_beams=cfg.final_num_beams,
                max_length=final_max_length,
                batch_size=cfg.beam_batch_size,
                use_bf16=use_bf16,
                description=f"valid/{_candidate.name}",
                on_progress=on_progress,
                resume_from=resume_from,
            )

        hypotheses = _decode_with_cache(
            cache, cache_path, decode_signature, candidate.name, weights_fingerprint,
            valid_dataset, decode,
        )
        if scored_model is not None:
            del scored_model
            _free_cuda()

        report = compute_wer_report(valid_dataset.texts(), hypotheses, cfg.expand_numbers)
        entry = {
            "checkpoint": str(candidate),
            "valid_wer_norm": report.wer_norm,
            "valid_wer_raw": report.wer_raw,
            "valid_cer_norm": report.cer_norm,
        }
        reranking.append(entry)
        LOGGER.info(
            "  %-28s wer_norm=%.4f  wer_raw=%.4f  cer_norm=%.4f",
            candidate.name, report.wer_norm, report.wer_raw, report.cer_norm,
        )
        # An unscorable checkpoint sorts last rather than winning by accident.
        score = float("inf") if math.isnan(report.wer_norm) else report.wer_norm
        if best_score is None or score < best_score:
            best_score, best_entry, best_report, best_hyps = score, entry, report, hypotheses

    if best_entry is None or best_report is None or best_hyps is None:
        raise RuntimeError("checkpoint re-ranking produced no result")
    if best_score == float("inf"):
        raise RuntimeError(
            "every checkpoint scored an undefined WER; the validation references appear to be empty"
        )
    LOGGER.info(
        "selected %s with validation wer_norm=%.4f",
        best_entry["checkpoint"], best_entry["valid_wer_norm"],
    )
    write_predictions_csv(
        cfg.output_dir / "valid_predictions.csv", valid_dataset, best_hyps, best_report
    )

    best_model = load_model(cfg, model_path, source=best_entry["checkpoint"])
    best_model.config.use_cache = True  # the saved artefact is for inference
    cfg.best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(str(cfg.best_model_dir), safe_serialization=True)
    processor.save_pretrained(str(cfg.best_model_dir))
    LOGGER.info("saved the selected model to %s", cfg.best_model_dir)

    # ---- stage 5: the single test pass -----------------------------------
    banner("Stage 5/5 - final test decoding")
    # The test split is decoded once per selected model. If the run is interrupted
    # between decoding and writing the results, the cached hypotheses are reused rather
    # than decoded again - and because the cache is keyed on the winning checkpoint's
    # weights, a different selection always forces a fresh decode.
    test_cache_path = cfg.output_dir / "test_decode_cache.json"
    test_cache = _load_decode_cache(test_cache_path, decode_signature)
    selected_fingerprint = _checkpoint_fingerprint(Path(best_entry["checkpoint"]))

    def decode_test(on_progress, resume_from):
        return transcribe_dataset(
            model=best_model,
            dataset=test_dataset,
            collator=collator,
            tokenizer=tokenizer,
            cfg=cfg,
            num_beams=cfg.final_num_beams,
            max_length=final_max_length,
            batch_size=cfg.beam_batch_size,
            use_bf16=use_bf16,
            description="test",
            on_progress=on_progress,
            resume_from=resume_from,
        )

    test_hyps = _decode_with_cache(
        test_cache, test_cache_path, decode_signature, "selected", selected_fingerprint,
        test_dataset, decode_test,
    )

    test_report = compute_wer_report(test_dataset.texts(), test_hyps, cfg.expand_numbers)
    write_predictions_csv(
        cfg.output_dir / "test_predictions.csv", test_dataset, test_hyps, test_report
    )

    results = {
        "model": cfg.model_id,
        "dataset": cfg.dataset_id,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "recipe": recipe,
        "selected_checkpoint": best_entry["checkpoint"],
        "validation": {
            "wer_normalized": best_report.wer_norm,
            "wer_raw": best_report.wer_raw,
            "cer_normalized": best_report.cer_norm,
            "num_utterances": best_report.num_pairs,
            "num_skipped_empty_references": best_report.num_skipped_empty_refs,
            "num_beams": cfg.final_num_beams,
            "best_greedy_wer_normalized_during_training": best_greedy_wer,
        },
        "test": {
            "wer_normalized": test_report.wer_norm,
            "wer_raw": test_report.wer_raw,
            "cer_normalized": test_report.cer_norm,
            "num_utterances": test_report.num_pairs,
            "num_skipped_empty_references": test_report.num_skipped_empty_refs,
            "num_beams": cfg.final_num_beams,
            "wer_by_region": grouped_wer(test_dataset.records, test_report, "region"),
            "wer_by_province": grouped_wer(test_dataset.records, test_report, "province_name"),
        },
        "checkpoint_reranking": reranking,
        "splits": summaries,
        "training": {
            "num_train_utterances_used": len(train_dataset),
            "num_train_utterances_dropped": dropped,
            "per_device_train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": train_batch_size * grad_accum,
            "train_eval_generation_max_length": train_eval_max_length,
            "final_generation_max_length": final_max_length,
        },
        "normalization": {
            "raw_wer": "Unicode NFC plus whitespace collapsing only",
            "normalized_wer": [
                "Unicode NFC",
                "lowercase (diacritics preserved)",
                "canonical Vietnamese tone placement (hoà = hòa, thuý = thúy)",
                "% expanded to 'phan tram'",
                "digit sequences expanded to Vietnamese number words"
                if cfg.expand_numbers
                else "digit expansion disabled",
                "punctuation and symbols replaced by spaces",
                "whitespace collapsed",
            ],
        },
        "config": cfg.to_dict(),
        "environment": environment,
    }
    # log history first: final_results.json is the completion marker, so it must be
    # the last thing written, and atomically.
    _write_json_atomic(cfg.output_dir / "trainer_log_history.json", log_history)
    _write_json_atomic(cfg.results_path, results)

    banner("RESULTS")
    LOGGER.info(
        "validation  wer_normalized=%.4f  wer_raw=%.4f", best_report.wer_norm, best_report.wer_raw
    )
    LOGGER.info(
        "test        wer_normalized=%.4f  wer_raw=%.4f", test_report.wer_norm, test_report.wer_raw
    )
    LOGGER.info("results written to %s", cfg.results_path)
    return 0


def main() -> int:
    cfg = Config()
    setup_logging(cfg.log_dir)
    try:
        # Held for the whole run: checkpoints, caches and results are all written here.
        with DirectoryLock(cfg.output_dir, "training outputs").acquire():
            return run(cfg)
    except KeyboardInterrupt:
        LOGGER.warning("interrupted; re-run the same command to resume")
        return 130
    except BaseException:  # noqa: BLE001 - always leave a full trace in the log file
        # Logged once, then reported through the exit status. Re-raising here would print
        # the same traceback a second time as an uncaught exception.
        LOGGER.error("run failed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

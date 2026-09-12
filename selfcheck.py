#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify this machine, this install and the W&B setup before committing to a
12-hour training run.

    python3 selfcheck.py

Downloads nothing: no model, no dataset, no Hugging Face Hub access. It builds a
~10k-parameter Whisper from a config, writes a handful of synthetic 16 kHz shards
in the documented cache format, and drives the real functions from main.py over
them - the same tracking code paths the real run uses, on the real GPU.

It writes only into a scratch directory (outputs_selfcheck/ by default) and never
touches data/, outputs/ or any checkpoint. Safe to run at any time, including
while a real run is in progress, as long as VIMD_OUTPUT_DIR differs.

Exit status is 0 only when every check passed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("VIMD_SELFCHECK_DIR", str(ROOT / "outputs_selfcheck")))

# Tracking is the thing under test, so it is on unless explicitly disabled.
os.environ.setdefault("VIMD_WANDB", "1")
os.environ["VIMD_OUTPUT_DIR"] = str(SCRATCH)
os.environ["VIMD_DATA_DIR"] = str(SCRATCH / "data")
os.environ["VIMD_RAW_DIR"] = str(SCRATCH / "raw")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import main  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str):
    """Run a step, record pass/fail, never let one failure hide the rest."""
    def wrap(fn):
        print(f"\n--- {label}")
        try:
            detail = fn() or ""
            RESULTS.append((label, True, detail))
            print(f"    PASS  {detail}")
        except BaseException as exc:  # noqa: BLE001
            detail = f"{exc.__class__.__name__}: {exc}"
            RESULTS.append((label, False, detail))
            print(f"    FAIL  {detail}")
            if os.environ.get("VIMD_SELFCHECK_TRACE"):
                traceback.print_exc()
        return fn
    return wrap


# ---------------------------------------------------------------------------
# 1. machine
# ---------------------------------------------------------------------------
@check("GPU is visible and usable")
def _gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device. The real run aborts here too; check nvidia-smi and that "
            "torch came from the CUDA wheel index."
        )
    torch.zeros(1, device="cuda").add_(1).cpu()          # a real kernel, not just a query
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    note = "" if vram >= 70 else f" [below the 80 GB this recipe is tuned for]"
    return f"{name}, {vram:.0f} GB, bf16={'yes' if bf16 else 'no (will use fp16)'}{note}"


@check("pinned dependencies import")
def _deps():
    import accelerate, jiwer, pyarrow, scipy, soundfile, transformers  # noqa: F401
    return (f"torch {torch.__version__}, transformers {transformers.__version__}, "
            f"soundfile {soundfile.__version__}")


# ---------------------------------------------------------------------------
# 2. tracking
# ---------------------------------------------------------------------------
cfg = main.Config()
# A few seconds of training rather than twenty epochs. These touch only this
# scratch run; the recipe on disk is untouched because output_dir is scratch too.
cfg.eval_steps, cfg.logging_steps, cfg.warmup_steps = 2, 1, 0
cfg.num_train_epochs, cfg.save_total_limit = 3.0, 1
cfg.dataloader_workers, cfg.early_stopping_patience = 0, 99
cfg.wandb_log_samples, cfg.wandb_log_audio = 12, 4


@check("W&B credentials resolve without a prompt")
def _creds():
    if not cfg.wandb_enabled:
        raise RuntimeError("VIMD_WANDB is not 1, so there is nothing to check")
    if main.WandbRun._has_credentials():
        return f"key found for {main.WandbRun._api_host()} - this run will log live"
    raise RuntimeError(
        "no WANDB_API_KEY and no netrc entry. The run would record offline instead of "
        "live. Fix with `wandb login`, or put the key in wandb.key next to run.sh."
    )


@check("a W&B run opens")
def _start():
    main.TRACKER.start(cfg, {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                             else "none", "selfcheck": True, "bf16_supported": True})
    if not main.TRACKER.enabled:
        raise RuntimeError("TRACKER.start() did not enable tracking; see the log above")
    main.TRACKER.config_update({"SELFCHECK": True, "note": "synthetic data, not a result"})
    url = getattr(main.TRACKER.run, "url", None)
    return f"run {main.TRACKER.run_id} -> {url or 'offline'}"


# ---------------------------------------------------------------------------
# 3. the real dataset layer, over synthetic shards
# ---------------------------------------------------------------------------
SENTENCES = [
    "hôm nay trời nắng đẹp nên cả nhà đi chơi",
    "chị cho tôi hỏi đường ra bến xe đi lối nào",
    "bà con ở đây sống chủ yếu bằng nghề đánh cá",
    "mùa mưa năm nay đến sớm hơn mọi năm",
]
REGIONS = [("Ha Noi", "North"), ("Nghe An", "Central"), ("Can Tho", "South"),
           ("Hai Phong", "North"), ("Da Nang", "Central"), ("Ca Mau", "South")]
splits: dict = {}


@check("ViMDDataset loads shards written in the documented format")
def _dataset():
    rng = np.random.default_rng(0)
    for split, count in (("train", 24), ("valid", 18), ("test", 18)):
        d = Path(cfg.data_dir) / split
        d.mkdir(parents=True, exist_ok=True)
        offset, seconds = 0, 0.0
        with open(d / "shard_00000.bin", "wb") as audio, \
             open(d / "shard_00000.jsonl", "w", encoding="utf-8") as meta:
            for k in range(count):
                province, region = REGIONS[k % len(REGIONS)]
                n = int(rng.uniform(1.5, 4.0) * cfg.sampling_rate)
                audio.write((np.sin(np.arange(n) * 0.03) * 8000).astype("<i2").tobytes())
                meta.write(json.dumps({
                    "region": region, "province_code": f"{k % len(REGIONS):02d}",
                    "province_name": province, "filename": f"{split}_{k:04d}.wav",
                    "text": SENTENCES[k % len(SENTENCES)], "speakerID": f"spk{k % 5}",
                    "gender": ["male", "female"][k % 2], "offset": offset,
                    "num_samples": n, "duration": n / cfg.sampling_rate,
                    "source_sampling_rate": 44100,
                }, ensure_ascii=False) + "\n")
                offset += n * 2
                seconds += n / cfg.sampling_rate
        (d / "shard_00000.done").write_text(json.dumps({
            "format": main.SHARD_FORMAT_VERSION, "dataset_id": cfg.dataset_id,
            "revision": "selfcheck", "filename": f"data/{split}-00000-of-00001.parquet",
            "sampling_rate": cfg.sampling_rate, "rows": count, "seconds": seconds,
        }), encoding="utf-8")
        splits[split] = main.ViMDDataset(d, cfg.sampling_rate, cfg.dataset_id, "selfcheck")
    sizes = ", ".join(f"{s}={len(d)}" for s, d in splits.items())
    if splits["test"][0]["audio"].dtype != np.float32:
        raise RuntimeError("memmap decode returned the wrong dtype")
    return f"{sizes}; audio decodes to float32 at {cfg.sampling_rate} Hz"


# ---------------------------------------------------------------------------
# 4. the Trainer -> W&B seam, on the GPU
# ---------------------------------------------------------------------------
@check("Seq2SeqTrainer trains on the GPU and logs into that same run")
def _trainer():
    import wandb
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    before = wandb.run
    conf = WhisperConfig(
        vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16,
        decoder_ffn_dim=16, max_source_positions=50, max_target_positions=16,
        decoder_start_token_id=1, pad_token_id=0, eos_token_id=2, bos_token_id=1,
    )
    model = WhisperForConditionalGeneration(conf)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"input_features": torch.randn(8, 100, generator=g),
                    "labels": torch.randint(3, 63, (5,), generator=g)}

    def collate(batch):
        return {"input_features": torch.stack([b["input_features"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch])}

    def metrics(pred):
        # Shape-compatible with make_compute_metrics: the keys the recipe selects on.
        return {"wer_norm": 0.5, "wer_raw": 0.55, "cer_norm": 0.2, "skipped_empty_refs": 0.0}

    trainer = main.build_seq2seq_trainer(
        cfg=cfg, model=model, processor=None, train_dataset=DS(), eval_dataset=DS(),
        collator=collate, compute_metrics=metrics, callbacks=[],
        train_batch_size=4, grad_accum=1, eval_batch_size=4, generation_max_length=8,
        use_bf16=bool(torch.cuda.is_bf16_supported()), tf32=main._tf32_available(),
    )
    names = [c.__class__.__name__ for c in trainer.callback_handler.callbacks]
    if "WandbCallback" not in names:
        raise RuntimeError(f"WandbCallback did not attach; callbacks are {names}")
    trainer.train()

    if wandb.run is not before:
        raise RuntimeError("a SECOND W&B run was created; the Trainer did not reuse ours")
    if trainer.args.device.type != "cuda":
        raise RuntimeError(f"the Trainer ran on {trainer.args.device}, not the GPU")
    logged = {k for k in wandb.run.summary.keys() if k.startswith(("train/", "eval/"))}
    missing = {"train/loss", "eval/wer_norm"} - logged
    if missing:
        raise RuntimeError(f"these never reached W&B: {sorted(missing)}")
    return f"one run, on {trainer.args.device}, {len(logged)} train/eval series logged"


# ---------------------------------------------------------------------------
# 5. the stage 4/5 logging
# ---------------------------------------------------------------------------
@check("tables, dialect charts, prediction samples and audio are logged")
def _stage45():
    hyps = [" ".join(r["text"].split()[:-1]) or r["text"] for r in splits["test"].records]
    report = main.compute_wer_report(splits["test"].texts(), hyps, cfg.expand_numbers)
    by_region = main.grouped_wer(splits["test"].records, report, "region")
    by_prov = main.grouped_wer(splits["test"].records, report, "province_name")

    main.TRACKER.table("data/splits", ["split", "utterances", "hours"],
                       [[s, len(d), main.split_summary(d, cfg)["hours"]] for s, d in splits.items()])
    main.TRACKER.table("select/checkpoints", ["checkpoint", "valid_wer_norm"],
                       [["checkpoint-2", 0.31], ["checkpoint-4", 0.27]],
                       chart=("checkpoint", "valid_wer_norm", "Validation WER"))
    for key, buckets in (("region", by_region), ("province", by_prov)):
        main.TRACKER.table(f"test/wer_by_{key}", [key, "wer_norm", "num_utterances"],
                           [[n, v["wer_norm"], v["num_utterances"]] for n, v in buckets.items()],
                           chart=(key, "wer_norm", f"Test WER by {key}"))
    main.TRACKER.summary({"test/wer_norm": report.wer_norm, "status": "complete (SELFCHECK)"})
    main.log_predictions_table("test/samples", splits["test"], hyps, report, cfg)

    run_dir = Path(getattr(main.TRACKER.run, "dir", SCRATCH))
    wavs = list((run_dir / "media" / "audio").rglob("*.wav"))
    if len(wavs) < cfg.wandb_log_audio:
        raise RuntimeError(
            f"expected {cfg.wandb_log_audio} audio clips, found {len(wavs)} - "
            f"audio logging is broken again"
        )
    tables = list((run_dir / "media" / "table").rglob("*.json"))
    return (f"WER {report.wer_norm:.3f} over {report.num_pairs} utterances, "
            f"{len(by_region)} regions / {len(by_prov)} provinces, "
            f"{len(tables)} tables, {len(wavs)} audio clips")


@check("result files upload as an artifact")
def _artifact():
    payload = SCRATCH / "final_results.json"
    payload.write_text(json.dumps({"SELFCHECK": True}), encoding="utf-8")
    main.TRACKER.artifact(f"selfcheck-{main.TRACKER.run_id}", "results", [payload],
                          metadata={"selfcheck": True})
    if not main.TRACKER.enabled:
        raise RuntimeError("tracking switched itself off while uploading")
    return "queued"


@check("restarting reuses the same run instead of opening a second")
def _resume():
    first = main.TRACKER.run_id
    main.TRACKER.finish(0)
    main.TRACKER.start(cfg, {"gpu": "resume probe", "selfcheck": True})
    second = main.TRACKER.run_id
    url = getattr(main.TRACKER.run, "url", None)
    main.TRACKER.finish(0)
    if first != second:
        raise RuntimeError(f"run id changed across a restart: {first} -> {second}")
    return f"both launches attached to {first}; view it at {url or 'offline record'}"


# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
passed = sum(1 for _, ok, _ in RESULTS if ok)
for label, ok, detail in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        {detail}")
print("=" * 72)
print(f"  {passed}/{len(RESULTS)} checks passed")

if os.environ.get("VIMD_SELFCHECK_KEEP") != "1":
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"  scratch directory removed (VIMD_SELFCHECK_KEEP=1 to keep it)")

if passed == len(RESULTS):
    print("\n  Everything works. The machine is ready for the real run.")
    sys.exit(0)
print("\n  Fix the failures above before starting a real run.")
print("  Re-run with VIMD_SELFCHECK_TRACE=1 for full tracebacks.")
sys.exit(1)

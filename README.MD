# PhoWhisper-large × ViMD — Vietnamese multi-dialect ASR fine-tuning

Fine-tunes [`vinai/PhoWhisper-large`](https://huggingface.co/vinai/PhoWhisper-large) on the full
[ViMD dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) (Vietnamese Multi-Dialect,
102.6 hours across 63 provincial dialects) and reports WER on the official test split.

Optimised for one objective: the lowest achievable WER. Training runs unattended from start to
finish, needs no interaction, and resumes automatically if it is interrupted.

---

## 1. How to run

```bash
pip install -r requirements.txt
```

```bash
python3 main.py
```

That is the whole procedure. The script downloads the model and the dataset itself; nothing has to
be placed on disk beforehand and no prompt is ever shown.

Expected wall-clock on one 80 GB GPU: **~2–3 h** of one-time data preparation (network bound), then
**~8–12 h** of training and evaluation. Re-running `python3 main.py` after a crash, a timeout or a
reboot resumes rather than starting over — see §8 for exactly how much work is repeated.

Only one run may use an output directory at a time; a second launch aborts immediately with a clear
message rather than interleaving writes.

---

## 2. Environment

| | |
|---|---|
| OS / arch | Linux x86_64 |
| Python | **3.10 – 3.12** recommended (3.9 and 3.13 also resolve) |
| GPU | one NVIDIA GPU with ≥ 80 GB VRAM (A100 80 GB / H100) |
| CUDA | driver reporting CUDA 12.2; `torch==2.5.1+cu121` |
| CPU RAM | ≥ 32 GB (peak use is well under the lab's 64 GB) |
| Free disk | **≥ 150 GB** recommended; the run aborts immediately below 100 GB |
| Network | required — the model (6.2 GB) and dataset (59.8 GB) are pulled from the Hugging Face Hub |

Disk budget: ~12 GB for the 16 kHz audio cache, ~19 GB per retained training checkpoint (3 retained
by default), 6.2 GB for the selected model, 6.2 GB for the Hub model cache. The raw 59.8 GB of
parquet is **never** all on disk at once — shards are downloaded, converted and deleted one at a
time, so peak raw usage is about one shard (~0.5 GB).

Both `pip` steps must come from the pinned `requirements.txt`. In particular `transformers` is held
at **4.51.3**: from 4.52 onward transformers refuses to `torch.load` a `pytorch_model.bin` unless
torch ≥ 2.6, and PhoWhisper-large publishes its weights *only* as `pytorch_model.bin`. (`main.py`
also carries a runtime `.bin` → safetensors conversion fallback, so a newer transformers will still
work, just with an extra conversion step on the first run.)

---

## 3. Dataset

`nguyendv02/ViMD_Dataset`, 130 parquet shards, 59.8 GB, licence CC-BY-NC-ND-4.0.

| Split | Utterances | Used for |
|---|---|---|
| `train` | ~15.2k | training only |
| `valid` | ~1.84k | checkpoint selection and early stopping |
| `test`  | ~1.97k | decoded **exactly once**, at the very end, with the selected checkpoint |

> The official split is named **`valid`**, not `validation`. It is the validation split.

Columns: `region` (North/Central/South), `province_code`, `province_name`, `filename`, `text`,
`speakerID`, `gender`, `audio`. The source audio is **44.1 kHz stereo 16-bit WAV**; Whisper needs
16 kHz mono, so preparation downmixes to mono and resamples with a polyphase filter
(`scipy.signal.resample_poly`, up=160 / down=441, derived from `gcd` so any source rate works).

### Local cache layout

`main.py` writes its own compact cache instead of using the `datasets` library, which keeps the
pipeline independent of that library's Audio decoding backend (soundfile in 3.x, torchcodec in 4.x)
and cuts disk from ~150 GB to ~12 GB:

```
data/vimd_16k/<split>/shard_00000.bin      concatenated little-endian int16 PCM, 16 kHz mono
data/vimd_16k/<split>/shard_00000.jsonl    one record per utterance: byte offset, length, text, metadata
data/vimd_16k/<split>/shard_00000.done     written last; its absence means that shard is redone
```

The dataset revision is resolved **once**, on the first run, and recorded in
`data/vimd_16k/dataset_revision.json`; every later run reuses that pin rather than following upstream.
This is deliberate — otherwise a push to the ViMD repo during a multi-day job would invalidate every
prepared shard and every checkpoint, turning an ordinary restart into a 60 GB re-download and a
recipe-mismatch abort. It also makes the result reproducible. `VIMD_DATASET_REVISION` overrides the
pin explicitly. Each `.done` marker records the dataset id, that revision, the sample
rate and a format version; a shard is reused only when all of them match, so pointing
`VIMD_DATASET_ID` somewhere else — or an upstream change that keeps the same shard filenames — forces
a rebuild instead of silently serving stale audio. Shards that no longer exist in the current
revision are deleted rather than left to be loaded as orphans, and `ViMDDataset` independently
re-checks every marker at load time, so a split can never be assembled from two revisions. Audio is
read at training time through `np.memmap`, so the OS page cache is shared across all dataloader
workers.

**Failure policy.** Each shard retries its download 5 times with exponential backoff; the whole split
is then swept up to 3 times, retrying only the shards that are still missing, with a growing pause
between sweeps. That is what rides out a network outage lasting minutes without anyone re-launching
the job. A shard that survives all of it **aborts the run** — for *any* split.
Training on a silently incomplete dataset would invalidate the reported score, and because
preparation runs first, an abort happens in the first few hours rather than after days of training.
Re-running retries only the failed shard. Set `VIMD_ALLOW_INCOMPLETE_TRAIN=1` to downgrade this to a
skip **for the training split only**; the log then says explicitly that the result is not for the
full ViMD training set. Evaluation splits can never be skipped.

---

## 4. Model and training setup

`vinai/PhoWhisper-large` is a Whisper-large-v2 architecture (1280-dim, 32 encoder + 32 decoder
layers, 80 mel bins, 51865-token multilingual vocabulary, ~1.55 B parameters) already adapted to
Vietnamese on 844 hours of speech. This is a **full fine-tune** — no adapters, no frozen layers.

Three things are forced at load time, because PhoWhisper's shipped `generation_config.json` carries
`forced_decoder_ids: [[1, null], [2, 50359]]`, which leaves the language slot to be auto-detected
and conflicts with the modern language/task API:

- `config.forced_decoder_ids` and `generation_config.forced_decoder_ids` → `None`
- `generation_config.language = "vi"`, `task = "transcribe"`, `return_timestamps = False`
- `config.use_cache = False` (for gradient checkpointing), `generation_config.use_cache = True`

### Hyperparameters

| Parameter | Value | Why this value |
|---|---|---|
| learning rate | `5e-6` | The Whisper-large band. PhoWhisper-large is *already* Vietnamese-adapted, so a medium-sized LR risks eroding that adaptation on only 100 h of new data. |
| schedule | linear decay, 500 warmup steps | ≈2 epochs of warmup at the effective batch size below. |
| per-device batch | 16 | fp32 weights + grads + Adam states are ~25 GB on 1.55 B params; 16 leaves comfortable activation headroom on 80 GB. |
| gradient accumulation | 4 | **effective batch = 64**, the standard Whisper fine-tuning batch. |
| max epochs | 20 | ≈238 optimiser steps/epoch. Early stopping decides the real length; this is only a ceiling. |
| optimiser | `adamw_torch_fused`, wd `0.0`, grad clip `1.0` | SpecAugment and early stopping carry the regularisation. |
| precision | bf16 autocast (fp16 fallback), TF32 matmuls | `torch.cuda.is_bf16_supported()` decides at runtime. |
| gradient checkpointing | on, `use_reentrant=False` | required at this parameter count. |
| SpecAugment | `mask_time_prob=0.05`, `mask_time_length=10`, `mask_feature_prob=0.0` | time masking on the log-mel is the well-validated regulariser for Whisper fine-tuning. Requires the collator to emit `attention_mask`, which it does. |
| eval / save | every 250 steps (they must match for `load_best_model_at_end`) | ≈once per epoch. |
| selection metric | `eval_wer_norm`, lower is better | see §5. |
| early stopping | patience 5, threshold 5e-4 | ≈5 epochs without a 0.05-point WER gain. |
| checkpoints retained | 3 | ~19 GB each. |
| decoding during training | greedy | keeps the ~15 min validation loop affordable. |
| decoding for selection & test | beam 5 | |
| decode length cap | training evals: `min(448, max_train_label_tokens + 32)`; selection and test: **448** | a hard-coded 225 (a common default) would truncate long Vietnamese references. The training-eval cap is derived from the *training* labels only, so validation and test never influence the schedule; the passes that produce the reported scores use the model's own 448-token limit so no hypothesis is ever cut short. |
| dataloader | 8 workers, pinned, persistent | log-mel extraction runs in the workers via torch STFT. |
| seed | 42 | |

Training-set filtering (applied to **train only** — never to valid or test): utterances with empty
text, or whose tokenised label exceeds Whisper's 448-token decoder limit, are dropped and counted.

Utterances longer than 30 s (ViMD's maximum is 30.8 s) are truncated to Whisper's 30 s window by the
feature extractor, never dropped. The count is reported in `final_results.json` under
`num_utterances_over_30s_truncated_by_whisper`.

### No data leakage

The three official splits are used strictly for their stated purposes. Nothing is fitted on `valid`
beyond checkpoint selection, and `test` is loaded only in the final stage, after the winning
checkpoint has already been chosen and saved.

---

## 5. WER definitions

Two figures are reported for every evaluation, both computed with `jiwer` at corpus level (total
edits ÷ total reference words). `jiwer` is called directly rather than through `evaluate`, which
would download a metric script from the Hub mid-training.

**Raw WER** — reference vs hypothesis after Unicode NFC unification and whitespace collapsing only.
Casing, punctuation and diacritics all count as differences. This is the strictest reading.

**Normalized WER** — the checkpoint-selection metric. A deterministic pipeline applied identically
to reference and hypothesis, so it can only remove differences that are orthographic rather than
acoustic:

1. Unicode **NFC**
2. lowercase — `str.lower()` over NFC text, **never** an ASCII fold, so every diacritic survives
3. **canonical Vietnamese tone placement**, so the two spelling conventions compare equal:
   `hoà` ≡ `hòa`, `khoẻ` ≡ `khỏe`, `thuý` ≡ `thúy`. Applied only where the vowel cluster (`oa`,
   `oe`, `uy`) ends the syllable, which is exactly where the conventions disagree; `uy` after `q`
   is excluded because there the `u` belongs to the `qu` digraph (`quý`, never `qúy`). The mirror
   typo is repaired too: `qủa` → `quả`.
4. `%` → `phần trăm`
5. **digit expansion**: digit runs are spelled out as Vietnamese number words
   (`2024` → `hai nghìn không trăm hai mươi tư`, `1.000` → `một nghìn`). Variants chosen: `nghìn`,
   `tỷ`, `lẻ`, `mốt`, `tư`, `lăm`. Set `VIMD_EXPAND_NUMBERS=0` to disable.
6. punctuation and symbols → spaces
7. whitespace collapsed

Normalized **CER** is reported alongside as a diacritic-sensitive secondary signal.

Pairs whose reference normalises to empty are excluded from the corpus figure (jiwer raises on empty
references) and counted in `num_skipped_empty_references` rather than being silently scored as zero.

Training targets are the **original** ViMD text — NFC and whitespace-collapsed only, with casing,
punctuation and diacritics intact. Normalisation is a scoring step, not a training step.

---

## 6. Pipeline stages

| Stage | What it does |
|---|---|
| 1. Data preparation | Downloads, converts and deletes the 130 parquet shards one at a time into `data/vimd_16k/`. Resumable per shard. |
| 2. Indexing | Memory-maps the shards, tokenises all references to find the label-length distribution and derive `generation_max_length`, filters the training split. |
| 3. Fine-tuning | `Seq2SeqTrainer` with generation-based validation WER every 250 steps and early stopping. |
| 4. Checkpoint selection | Re-scores **every retained checkpoint** on validation with beam 5 — the final decoding config — rather than trusting the greedy ranking, and picks the lowest normalized WER. |
| 5. Test | Decodes the test split once with the selected model and writes the results. |

---

## 7. Outputs

```
outputs/
├── best_model/                 selected model + processor + generation config,
│                               loadable directly with from_pretrained()
├── final_results.json          best validation WER and final test WER (normalized + raw),
│                               CER, per-region and per-province test WER, the checkpoint
│                               re-ranking table, split statistics, the full hyperparameter
│                               record and the resolved environment
├── test_predictions.csv        one row per test utterance
├── valid_predictions.csv       same shape, for error analysis
├── metrics_history.json        WER at every evaluation step
├── trainer_log_history.json    the Trainer's full log history
├── dataset_fingerprint.json    guards against a silently changed dataset between runs
├── valid_rerank_cache.json     cached beam decodes, so an interruption during checkpoint
│                               selection does not repeat an hour of decoding
├── test_decode_cache.json      the same, for the final test pass
├── run_recipe.json             the recipe fingerprint this output directory belongs to
├── model_revision.json         the pinned PhoWhisper revision
├── .vimd.lock                  PID/host lock; blocks a concurrent second launch
├── checkpoints/                HF Trainer checkpoints (used for resume)
└── logs/train.log              complete run log
```

Both CSVs carry: `index`, `filename`, `region`, `province_code`, `province_name`, `speakerID`,
`gender`, `duration`, `reference`, `prediction`, `reference_normalized`, `prediction_normalized`,
`wer_norm` (per utterance). They are written as `utf-8-sig` so Vietnamese renders correctly if
opened directly in Excel.

`final_results.json` is the single file to read for the headline numbers.

---

## 8. Robustness

**Resume.** Every stage is idempotent, though resumption is not free — concretely, a restart repeats:

- the shard that was being converted when the run died (one shard, ~0.5 GB);
- training back to the last checkpoint, so **up to `VIMD_EVAL_STEPS` (250) optimiser steps**;
- at most the last ~20 batches of whichever beam decode was in flight — validation
  re-ranking and the test decode checkpoint their progress as they go, so an interruption
  halfway through a split resumes there rather than restarting it.

Training resumes from the newest *validated* checkpoint. Validation is by content, not by filename:
`trainer_state.json` must parse and carry a `global_step`, the weights must be present, and for
resuming (not merely scoring) `optimizer.pt` and `scheduler.pt` must be there too — resuming without
them would silently reset Adam's moments and the LR schedule mid-run. Trainer writes the weights
first and the state file last, so a save killed halfway is detected and skipped rather than selected
and then failed on. Callback state is restored too
(`restore_callback_states_from_checkpoint=True`) — without it the early-stopping patience counter
would silently reset on every restart and let a plateaued run keep training.

**Completion and recipe identity.** The run computes a *recipe fingerprint* covering everything that
changes what the result means: model id **and revision**, dataset id **and revision**, per-split
content digests, learning rate, schedule, warmup, epochs, effective batch, SpecAugment, early
stopping, beam count, the WER normalization setting, seed and a pipeline version. Per-device batch and accumulation are folded into the *effective* batch,
because the OOM back-off changes those two deliberately mid-run.

- `outputs/final_results.json` is the completion marker, written atomically and last. A restart exits
  0 only if it parses, contains a test WER, **and** its stored recipe matches the current one. A
  truncated file is moved to `final_results.json.damaged`; a file from a *different* configuration is
  an error telling you to use a new `VIMD_OUTPUT_DIR` — it is neither reported as this run's result
  nor silently overwritten.
- `outputs/checkpoints/recipe.json` guards resume the same way. Changing the model, learning rate,
  schedule or seed and restarting into an existing output directory aborts with a field-by-field diff
  instead of resuming another experiment's weights. If checkpoints exist but that file is missing or
  unreadable, the run aborts rather than writing the current recipe over the gap — weights that
  cannot be shown to belong to this configuration are never laundered into it.

Every file the next run reads back to decide what to skip is written temp-file → `fsync` →
`os.replace`, with a temp name unique per process so two runs sharing a data directory cannot
collide. Because the recipe is only fully known after indexing, this check runs after stage 2:
a finished run spends a couple of minutes re-verifying shards before exiting.

**Cache identity.** Decode caches are keyed on the full configuration — model id, dataset id *and
revision*, split contents, language, task, beam count, decode length, tokenizer, precision — plus,
per entry, a fingerprint of the exact weight files that produced it. A cache written by a different
experiment in a reused output directory is discarded rather than mistaken for this one's.
(`VIMD_EXPAND_NUMBERS` is deliberately *not* part of the key: it affects scoring only, and scores are
always recomputed from cached hypotheses, so toggling it never throws away valid decoding.)

**Test decoding.** The test split is decoded once per selected checkpoint, with progress
checkpointed as it goes. A crash — whether after the decode or halfway through it — resumes rather
than repeating the split, and because the cache is keyed on the winning checkpoint's weights, a
different selection always forces a fresh decode. Test data never influences checkpoint selection.

**Concurrency.** Locks are held with `fcntl.flock`, so the *kernel* arbitrates: exactly one process
holds a lock at any instant even when several start in the same millisecond, and the lock is dropped
automatically when the owner dies — including SIGKILL, an OOM kill or a node reboot. That removes
stale locks as a concept, and with them the read-then-take-over sequence a PID file needs (two
processes racing to reclaim the same dead owner can both delete and recreate it and both think they
won). Verified with eight processes released simultaneously through a barrier: exactly one acquires,
with and without a dead owner's file left behind. The lock file's contents are diagnostics only.

The output directory is locked for the whole run; the data and raw directories only during
preparation, so two experiments with different output directories may still share a prepared audio
cache — they just cannot build it at the same time.

**Scratch cleanup.** Everything downloaded goes inside `<raw_dir>/.vimd_scratch/`, which this script
creates and marks as its own, and that subdirectory is the only thing cleanup ever deletes.
`VIMD_RAW_DIR` itself is never touched — including any `data/` or `.cache/` you keep there. A
`.vimd_scratch` that exists without the marker (or is a symlink) is left alone and reported.

**GPU OOM.** A CUDA OOM during training is caught: the cache is emptied, the per-device batch is
halved and gradient accumulation doubled — keeping the effective batch of 64 unchanged, so the
recipe itself does not change — and training resumes from the last checkpoint. Up to 3 retries.
Decoding backs off the same way.

**Network.** Every Hub interaction retries 5 times with exponential backoff — dataset listing, each
shard download, the model-revision lookup and the 6.2 GB model snapshot itself. The model is fetched
once by `snapshot_download` at a pinned revision and loaded from that local path, so a network
interruption during the big download is retried rather than killing the run, and an upstream
PhoWhisper push between restarts cannot swap the weights or tokenizer under an unchanged recipe.
`VIMD_MODEL_REVISION` overrides the pin.

**Unattended safety.** `report_to=[]` is set explicitly on `TrainingArguments`; the default `"all"`
would try to initialise Weights & Biases and **block on a login prompt**. `WANDB_DISABLED`,
`WANDB_MODE=offline` and `COMET_MODE=DISABLED` are set before any import, progress bars are off, and
all output goes to both stdout and `outputs/logs/train.log`.

**Preflight.** Before any work starts: Python version; every `VIMD_*` value validated (a
non-positive batch size or an unrecognised boolean spelling fails here with a readable message rather
than silently becoming `False` hours later); every directory created and probed by actually writing
to it; free disk on *each distinct volume* backing the output, data, raw and `HF_HOME` paths; CUDA
present *and usable* (a real allocation is attempted, so a driver/runtime mismatch surfaces
immediately); bf16 support; and TF32, which is enabled only where the hardware allows it — passing
`tf32=True` on a pre-Ampere GPU raises outright and would otherwise defeat the fp16 fallback. VRAM
below 70 GB and low system RAM produce warnings with the exact override to use, not a hard failure:
the OOM back-off already handles a smaller card.

---

## 9. Environment overrides

Every tunable is settable without editing code. Defaults in brackets.

| Variable | |
|---|---|
| `VIMD_MODEL_ID` | pretrained checkpoint [`vinai/PhoWhisper-large`] |
| `VIMD_DATASET_ID` | dataset repo [`nguyendv02/ViMD_Dataset`] |
| `VIMD_DATASET_REVISION` | pin a specific revision [empty = pin whatever the first run resolved] |
| `VIMD_MODEL_REVISION` | pin a specific model revision [empty = pin whatever the first run resolved] |
| `VIMD_LR` | peak learning rate [`5e-6`] |
| `VIMD_LR_SCHEDULER` | [`linear`] |
| `VIMD_EPOCHS` | max epochs [`20`] |
| `VIMD_WARMUP_STEPS` | [`500`] |
| `VIMD_WEIGHT_DECAY` / `VIMD_MAX_GRAD_NORM` | [`0.0`] / [`1.0`] |
| `VIMD_LOGGING_STEPS` | [`25`] |
| `VIMD_TRAIN_BATCH_SIZE` / `VIMD_GRAD_ACCUM` | per-device batch [`16`] / accumulation [`4`] |
| `VIMD_EVAL_BATCH_SIZE` / `VIMD_BEAM_BATCH_SIZE` | greedy eval [`16`] / beam decode [`16`] |
| `VIMD_EVAL_STEPS` | eval and save interval [`250`] |
| `VIMD_EARLY_STOPPING_PATIENCE` / `VIMD_EARLY_STOPPING_THRESHOLD` | [`5`] / [`5e-4`] |
| `VIMD_SAVE_TOTAL_LIMIT` | checkpoints retained [`3`] |
| `VIMD_FINAL_NUM_BEAMS` | beams for selection and test [`5`] |
| `VIMD_SPEC_AUGMENT` | SpecAugment on/off [`1`] |
| `VIMD_MASK_TIME_PROB` / `VIMD_MASK_TIME_LENGTH` | [`0.05`] / [`10`] |
| `VIMD_MASK_FEATURE_PROB` / `VIMD_MASK_FEATURE_LENGTH` | [`0.0`] / [`10`] |
| `VIMD_EXPAND_NUMBERS` | digit expansion in the normalizer [`1`] |
| `VIMD_DATALOADER_WORKERS` | [`8`] |
| `VIMD_DOWNLOAD_RETRIES` / `VIMD_OOM_RETRIES` | [`5`] / [`3`] |
| `VIMD_PREPARE_SWEEPS` | full retry sweeps over shards that still failed [`3`] |
| `VIMD_MIN_FREE_GB` / `VIMD_RECOMMENDED_FREE_GB` | abort [`100`] / warn [`200`] |
| `VIMD_KEEP_RAW_PARQUET` | keep the raw shards instead of deleting them [`0`] |
| `VIMD_ALLOW_INCOMPLETE_TRAIN` | skip an unpreparable *training* shard instead of aborting [`0`] |
| `VIMD_MIN_VRAM_GB` / `VIMD_MIN_RAM_GB` | warn-only thresholds [`70`] / [`24`] |
| `VIMD_DATA_DIR` / `VIMD_RAW_DIR` / `VIMD_OUTPUT_DIR` | paths, all default under the repo |
| `VIMD_SEED` | [`42`] |
| `HF_HOME` | Hub cache [`<repo>/.hf_cache`] — kept inside the repo so a small `$HOME` quota cannot break the 6.2 GB model download |

Example — smaller GPU, same effective batch:

```bash
VIMD_TRAIN_BATCH_SIZE=8 VIMD_GRAD_ACCUM=8 python3 main.py
```

---

## 10. Listening to individual utterances

Error analysis usually ends with wanting to *hear* a specific utterance. `download_samples.py`
fetches named utterances on their own, without preparing the whole 60 GB dataset:

```bash
python3 download_samples.py                          # the five outliers from the error analysis
python3 download_samples.py 73_0332 81_0303.wav      # any utterance, .wav optional
python3 download_samples.py --also-16k               # plus the 16 kHz mono the model actually hears
python3 download_samples.py --source local           # read the prepared cache, no network
```

Audio lands in `outputs/audio_samples/`, one `.wav` per utterance plus `samples.json` carrying the
reference transcript, region, province, speaker and duration for each. The utterance name is the
`filename` column of the prediction CSVs, so any row of `test_predictions.csv` can be listened to
directly.

`download_samples_colab.ipynb` does the same on Google Colab — nothing to install locally, and the
utterances play inline in the browser with their reference and predicted text side by side. Open it
in Colab and `Runtime → Run all`. The notebook is deliberately self-contained: it embeds the same
fetch logic rather than importing the script, so opening the link is enough.

How it avoids the 60 GB: parquet is columnar and split into row groups, so the script opens each
shard **over the network** and reads only the `filename` column to locate the utterance, then pulls
just the one row group holding it — tens of MB instead of tens of GB. The filename listing is
cached in `outputs/audio_samples/_filename_index.json`, so looking up further utterances later is
instant. If `data/vimd_16k/` already exists it is used first and no network is touched at all.
Original WAV bytes are written through verbatim, so the audio is bit-identical to the published
dataset. If range requests are unavailable the script falls back to downloading whole shards,
scanning and deleting them one at a time.

---

## 11. Repository layout

```
main.py                        the entire pipeline, self-contained
download_samples.py            standalone utility: fetch named utterances to listen to
download_samples_colab.ipynb   the same, as a Google Colab notebook with inline players
requirements.txt               pinned dependencies
README.MD                      this file
data/                          created at runtime: the 16 kHz cache
outputs/                       created at runtime: models, metrics, predictions, logs
```

`main.py` is deliberately a single file with no local imports: nothing can break from a partial
upload or a Python path surprise on the training server. `download_samples.py` follows the same
rule — it does not import `main.py`, so it can be copied to a laptop and run on its own.

---

## 12. Citation

ViMD — Nguyen et al., *Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges*,
EMNLP 2024. PhoWhisper — Le et al., *PhoWhisper: Automatic Speech Recognition for Vietnamese*,
ICLR 2024 Tiny Papers.

ViMD is licensed **CC-BY-NC-ND-4.0** — non-commercial research use.

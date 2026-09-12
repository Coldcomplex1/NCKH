# PhoWhisper-large × ViMD — Vietnamese multi-dialect ASR fine-tuning

Fine-tunes [`vinai/PhoWhisper-large`](https://huggingface.co/vinai/PhoWhisper-large) on the full
[ViMD dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) (Vietnamese Multi-Dialect,
102.6 hours across 63 provincial dialects) and reports WER on the official test split.

Optimised for one objective: the lowest achievable WER. Training runs unattended from start to
finish, needs no interaction, and resumes automatically if it is interrupted.

---

## 1. How to run

```bash
bash run.sh
```

That is the whole procedure. `run.sh` installs the pinned dependencies and starts the run; the
script downloads the model and the dataset itself, nothing has to be placed on disk beforehand,
and no prompt is ever shown.

Because the run outlives most SSH sessions, start it detached:

```bash
tmux new -s vimd 'bash run.sh'      # detach with Ctrl-B then D
```

The two steps separately, if you prefer them explicit — identical except that experiment
tracking is then off unless you ask for it (§10):

```bash
pip install -r requirements.txt
python3 main.py
```

Expected wall-clock on one 80 GB GPU: **~2–3 h** of one-time data preparation (network bound), then
**~8–12 h** of training and evaluation. Re-running `bash run.sh` after a crash, a timeout or a
reboot resumes rather than starting over — see §8 for exactly how much work is repeated.

Only one run may use an output directory at a time; a second launch aborts immediately with a clear
message rather than interleaving writes.

---

## 2. Environment

| | |
|---|---|
| OS / arch | Linux x86_64 for training. `selfcheck.py` also runs on native Windows — only the output-directory lock needs `fcntl`, and it is the one non-portable thing here, so a Windows workstation can verify a GPU and the W&B wiring without WSL. |
| Python | **3.10 – 3.12** recommended; 3.9 and 3.13 also resolve. **3.14+ does not**: torch 2.5.1 publishes no wheel above cp313, so `requirements.txt` cannot be installed on it. `run.sh` and `selfcheck.py` check this and say so. |
| GPU | one NVIDIA GPU with ≥ 80 GB VRAM (A100 80 GB / H100) |
| CUDA | driver reporting CUDA 12.2; `torch==2.5.1+cu121` |
| CPU RAM | ≥ 32 GB (peak use is well under the lab's 64 GB) |
| Free disk | **≥ 150 GB** recommended; the run aborts immediately below 100 GB |
| Network | required — the model (6.2 GB) and dataset (59.8 GB) are pulled from the Hugging Face Hub |

Disk budget: ~12 GB for the 16 kHz audio cache, ~19 GB per retained training checkpoint (3 retained
by default), 6.2 GB for the selected model, 6.2 GB for the Hub model cache. The raw 59.8 GB of
parquet is **never** all on disk at once — shards are downloaded, converted and deleted one at a
time, so peak raw usage is about one shard (~0.5 GB).

### Windows

`selfcheck.py` runs on native Windows, in PowerShell, against the local GPU — useful for
confirming the machine and the W&B setup before handing the code to a Linux host:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python selfcheck.py
```

`requirements.txt` resolves on both platforms: the `+cu121` local version is published for Linux
only, so Windows takes the plain `torch==2.5.1` wheel, which bundles CUDA already.

Training itself still needs Linux. `run.sh` is a shell script, and `DirectoryLock` needs
`fcntl.flock` — without it two runs could write one output directory and corrupt each other's
checkpoints, so `main.py` stops there rather than proceeding unlocked. Under WSL2 the Windows
NVIDIA driver provides CUDA, so the GPU is still used; clone into the Linux filesystem rather
than `/mnt/c`, which is slow enough to matter for a 12 GB cache.

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
| degenerate-hypothesis repair | detect, then re-decode only what is flagged | A handful of decoder repetition loops carried 4% of the test error budget. Detection is reference-free and the base decode is untouched, so every other hypothesis is bit-identical to a run without it. See §5. |
| decode length cap | training evals: `min(448, max_train_label_tokens + 32)`; selection and test: **448** | a hard-coded 225 (a common default) would truncate long Vietnamese references. The training-eval cap is derived from the *training* labels only, so validation and test never influence the schedule; the passes that produce the reported scores use the model's own 448-token limit so no hypothesis is ever cut short. |
| dataloader | 8 workers, pinned, persistent | log-mel extraction runs in the workers via torch STFT. |
| seed | 42 | |

Training-set filtering (applied to **train only** — never to valid or test): utterances with empty
text, or whose tokenised label exceeds Whisper's 448-token decoder limit, are dropped and counted.

**Named exclusions** are the one filter that does touch an evaluation split, and the bar is
deliberately high: a row qualifies only when its audio and its reference do not describe the same
speech, which is a property of the corpus that no decoder change can fix. `VIMD_EXCLUDE_UTTERANCES`
lists them by filename; they are removed from **every** split before anything is summarised, decoded
or scored, each removal is logged, and `final_results.json` names every one with its reason. The
default list holds a single utterance:

| Utterance | Split | Why |
|---|---|---|
| `77_0282.wav` | test, Bình Định, 21.8 s | The audio opens with roughly ten seconds of speech the ViMD reference does not contain, so the hypothesis can only match its tail. It scored 95% WER against a model that had transcribed it correctly. |

So the reported test WER is a figure over **2,025** utterances, not 2,026 — which is why the count
is in the results file beside the score. Set `VIMD_EXCLUDE_UTTERANCES=` (empty) to exclude nothing
and score the corpus exactly as published. A name that matches no utterance aborts the run rather
than being ignored: excluding nothing while the configuration says otherwise would misreport what
the score is a score over.

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

### Degenerate hypotheses

Whisper's decoder occasionally latches onto a phrase and emits it until the 448-token cap. On the
first full run three test utterances did this — `73_0332.wav` ended `"hội" ×110`, `81_0303.wav`
`"được trồng cây này" ×34` — and between them they carried **4.0% of the entire test error budget**
at per-utterance WER above 400%. Two more did it on validation. Nothing in beam search stops a beam
re-entering the loop it just left, so these have to be noticed and re-decoded rather than hoped away.

**Detection** reads the hypothesis and the audio duration and *nothing else*. It must never look at
the reference: the repair below re-runs generation on whatever it flags, and a test that consulted
the reference would be fitting the decoder to the test set. Two independent rules, each a
conjunction so that neither fires on thin evidence:

- **loop** — at least 48 words, at least half of them inside immediately-repeated word n-grams
  (n ≤ 6) that repeat at least 4 times, and one run repeating at least 6 times. Half a transcript
  inside a sixfold repetition is not speech at any rate, so this needs no corroboration. Coverage is
  the *union* of every qualifying run, not the longest one: `73_0332.wav` is two consecutive runs,
  and a loop that drifts fragments into several that a longest-run measure would let through.
- **rate** — at least 48 words, an impossible word rate, **and** one of: weaker repetition, a zlib
  compression ratio past 2.4 (Whisper's own loop threshold), or generation still running when it hit
  the token cap. An implausible rate alone could be a dense transcript. This rule is the backstop for
  a loop that drifts rather than repeating verbatim.

Words are counted after the normalizer above (with digit expansion pinned off, so the verdict does
not move with `VIMD_EXPAND_NUMBERS`). Normalising first is load-bearing rather than tidy: the loops
emit `ủng hộ, ủng hộ`, and a raw `split()` compares `hộ,` against `hộ` and finds no repetition at
all. The rate denominator is clamped to 30 s because that is all Whisper ever sees.

The word-rate limit is derived from the **training split only**, as `max(6.0, 1.25 × p99.9)` of its
words per second — the same discipline as the training-eval decode budget, so validation and test
stay out of every decision made about them, and a percentile rather than a maximum so one misaligned
training row cannot raise the bar above every loop.

The thresholds are calibrated against the case that must *not* fire: `38_0284.wav` genuinely says
`khi khi khi khi khi` — five unigram repeats at 12% coverage and 3.1 words/s — and is clean under
both rules at every limit.

**Repair** re-decodes only the flagged utterances, one at a time, up a ladder of escalating
constraints. Everything else is returned untouched, so the reported score moves where the decoder
had actually failed rather than because a blanket penalty was applied to two thousand utterances
that did not need one.

| Rung | Constraint |
|---|---|
| 1 | `no_repeat_ngram_size=6` — bans only long verbatim loops, so it is almost free of collateral damage |
| 2 | `no_repeat_ngram_size=4` |
| 3 | `no_repeat_ngram_size=3`, `repetition_penalty=1.15` |
| 4 | + `length_penalty=0.8` — beam search then prefers finishing over filling the budget |
| 5 | + the token budget itself cut to what the audio can hold, from train tokens-per-second |

A rung is accepted the moment the detector clears its output, and not otherwise — the accept test
never looks at the reference, so a repair cannot be rejected for being worse in WER and must not be.
Nothing is sampled: a temperature fallback would make the reported score depend on an RNG. If no
rung clears it, the **original** hypothesis is kept and the failure is recorded, because a
hypothesis that was never validated must not reach the score. Repairs decode at batch size 1, so a
repair does not depend on how many other rows happened to be flagged alongside it.

`VIMD_REPAIR_DEGENERATE=0` is the dry run: the detector still flags, logs and reports, and every
hypothesis is left exactly as it was decoded. `wer_normalized_before_repair` sits beside
`wer_normalized` in `final_results.json` as the standing ablation.

---

## 6. Pipeline stages

| Stage | What it does |
|---|---|
| 1. Data preparation | Downloads, converts and deletes the 130 parquet shards one at a time into `data/vimd_16k/`. Resumable per shard. |
| 2. Indexing | Memory-maps the shards, applies any named exclusions, tokenises all references to find the label-length distribution, and derives from the *training* split alone the `generation_max_length`, the word-rate limit the degeneracy check uses and the token budget its last repair rung uses. Filters the training split. |
| 3. Fine-tuning | `Seq2SeqTrainer` with generation-based validation WER every 250 steps and early stopping. |
| 4. Checkpoint selection | Re-scores **every retained checkpoint** on validation with beam 5 — the final decoding config — rather than trusting the greedy ranking, and picks the lowest normalized WER. Degenerate hypotheses are counted for every candidate but repaired only for the winner: selection deliberately runs on the pre-repair metric, so adding the repair pass can neither change which checkpoint wins nor invalidate the test decode cache. |
| 5. Test | Decodes the test split once with the selected model, repairs any degenerate hypotheses, and writes the results. |

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
├── repair_cache.json           the re-decodes of degenerate hypotheses, keyed separately so a
│                               change to the ladder or the thresholds costs nothing else
├── training_complete.json      written when stage 3 finishes; a re-run re-scores the existing
│                               checkpoints instead of resuming training into them
├── run_recipe.json             the recipe fingerprint this output directory belongs to
├── model_revision.json         the pinned PhoWhisper revision
├── wandb_run.json              the W&B run id this directory owns, so a restart resumes
│                               one run instead of starting a second (only with VIMD_WANDB=1)
├── wandb/                      W&B run files, cache and artifacts, kept here rather than
│                               under $HOME (only with VIMD_WANDB=1) — see §10
├── .vimd.lock                  PID/host lock; blocks a concurrent second launch
├── checkpoints/                HF Trainer checkpoints (used for resume)
└── logs/train.log              complete run log
```

Both CSVs carry: `index`, `filename`, `region`, `province_code`, `province_name`, `speakerID`,
`gender`, `duration`, `reference`, `prediction`, `reference_normalized`, `prediction_normalized`,
`wer_norm` (per utterance), then `repetition_rule`, `repair_rung` and
`prediction_before_repair` — empty for every row the repair pass did not touch, so the
before/after diff is one spreadsheet filter away. The three are **appended**, so anything that
reads the original columns by position still works. Excluded utterances are absent from both
files: they were removed before the split was decoded. The files are written as `utf-8-sig` so
Vietnamese renders correctly if opened directly in Excel.

`final_results.json` is the single file to read for the headline numbers. Per split it carries
`wer_normalized` (the headline: repaired, and over the rows that remained after any exclusion),
`wer_normalized_before_repair` as the ablation, `num_utterances` for what the figure is a figure
over, and a `degenerate_repair` block naming every flagged utterance, the rung that fixed it or
that none did, and the share of the error budget they carried. At the top level,
`excluded_utterances` names every removed row with its split, duration, province and reason, and
`degenerate_repair_spec` records the thresholds and the ladder the run used.

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

**A finished stage 3 stays finished.** `outputs/training_complete.json` is written once
`trainer.train()` returns normally, and a later run that matches its training recipe skips stage 3
entirely and re-scores the checkpoints already on disk. Without it, re-running a *finished*
experiment — which is exactly what an evaluation-side change asks you to do — resumes from the last
checkpoint and keeps training: early stopping does fire again, but only at the next evaluation,
because `TrainerControl.should_training_stop` is the one piece of state `trainer_state.json` does
not carry. That costs an `VIMD_EVAL_STEPS`-worth of GPU time and, worse, writes a checkpoint the
original run never produced, which can then win stage 4 and change which model the experiment
reports. The marker is checked against the newest checkpoint's `global_step` as well as the recipe,
so one left by a shorter run cannot shortcut a longer one, and against the named best checkpoint
still being on disk.

An output directory produced before this marker existed does not have one. Write it by hand from
values already on disk — `run_recipe.json` projected onto the training keys, the finished
`global_step`, the epoch from `trainer_log_history.json`, the selected checkpoint path, the
`best_greedy_wer_normalized_during_training` from `final_results.json`, and the batch size and
accumulation — or accept the extra `VIMD_EVAL_STEPS` of training on the next run.

**Completion and recipe identity.** The run computes a *recipe fingerprint* covering everything that
changes what the result means: model id **and revision**, dataset id **and revision**, per-split
content digests, learning rate, schedule, warmup, epochs, effective batch, SpecAugment, early
stopping, beam count, the WER normalization setting, the exclusion list, the repair settings, seed
and a pipeline version. Per-device batch and accumulation are folded into the *effective* batch,
because the OOM back-off changes those two deliberately mid-run.

That fingerprint answers two different questions, so it is compared two different ways.

- `outputs/final_results.json` is the completion marker, written atomically and last. A restart exits
  0 only if it parses, contains a test WER, **and** its stored **full** recipe matches the current
  one. A truncated file is moved to `final_results.json.damaged`; a file from a *different*
  configuration is an error — it is neither reported as this run's result nor silently overwritten.
  The error says which kind of difference it is: when every changed field is evaluation-side it
  tells you to move the earlier result aside and re-run, because the checkpoints are still valid
  and nothing needs retraining; otherwise it tells you to use a new `VIMD_OUTPUT_DIR`.
- `outputs/checkpoints/recipe.json` guards resume, and holds only the **training projection** of the
  recipe: the part that determines the weights. Changing the model, learning rate, schedule or seed
  and restarting into an existing output directory aborts with a field-by-field diff instead of
  resuming another experiment's weights. If checkpoints exist but that file is missing or
  unreadable, the run aborts rather than writing the current recipe over the gap — weights that
  cannot be shown to belong to this configuration are never laundered into it.

The projection is what makes an evaluation-side change affordable. Left out of it are the beam
count, the exclusion list and the repair settings — stages 4 and 5 read those, and training
evaluations are hardcoded greedy at `generation_num_beams=1` — along with the **test** split's
content digest and utterance count, since nothing reads the test split before stage 5. The valid
split stays in, because the greedy validation WER drives early stopping and therefore which
checkpoints exist at all. Excluding a *train* or *valid* row needs no separate key: it already
shows up through `train_used` and those splits' digests, which are in the projection.

Both sides of the comparison are projected, and every key the projection keeps already existed, so
a `recipe.json` written before the split — holding the whole recipe — still validates. Changing how
a finished run is scored therefore re-decodes it; it never asks for the training to be repeated.

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

The repair pass keeps its **own** cache under its own key, which is the decode key plus the ladder,
the detector thresholds and the derived limits. A base decode does not depend on any of those — the
repair is a second pass over a handful of utterances — so retuning the detector invalidates the
repairs and nothing else. Within that cache each utterance is written as it is finished, because
each repair is independent; an interruption resumes at the next one. Repairs are folded into the
base hypotheses they were computed from, so a changed base decode can never be answered with
repairs belonging to a different one.

Alongside each cached hypothesis sits a flag for whether generation was still running when it hit
the token cap, which the detector uses as corroborating evidence. It is absent from a cache written
before the flag existed, where it reads as *unknown* rather than as false: a cached hypothesis is
never re-decoded merely to learn how it ended, and since the cap is only ever corroborating, an
older cache still catches every loop through the repetition rule.

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
| `VIMD_EXCLUDE_UTTERANCES` | utterances removed from every split before decoding and scoring [`77_0282.wav`] — bare ViMD filenames, comma- or whitespace-separated. An explicitly empty value excludes nothing; a name matching no utterance aborts the run. |
| `VIMD_REPAIR_DEGENERATE` | re-decode degenerate hypotheses [`1`]. `0` still flags, logs and reports them, but leaves every hypothesis as decoded — the dry run. |
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
| `VIMD_WANDB` | experiment tracking on/off [`0`] — the single switch, see §10 |
| `WANDB_PROJECT` / `WANDB_ENTITY` | W&B project [`phowhisper-vimd`] / account [empty = default] |
| `WANDB_MODE` | `online` / `offline` / `disabled` [`online` when tracking is on] |
| `VIMD_WANDB_RUN_NAME` / `VIMD_WANDB_TAGS` / `VIMD_WANDB_NOTES` | run label, comma-separated tags, free-text note |
| `VIMD_WANDB_LOG_SAMPLES` | decoded utterances in the predictions table [`64`], `0` disables |
| `VIMD_WANDB_LOG_AUDIO` | audio clips attached to that table [`0`] |
| `VIMD_WANDB_LOG_MODEL` | upload the selected model (6.2 GB) as an artifact [`0`] |
| `HF_HOME` | Hub cache [`<repo>/.hf_cache`] — kept inside the repo so a small `$HOME` quota cannot break the 6.2 GB model download |

Example — smaller GPU, same effective batch:

```bash
VIMD_TRAIN_BATCH_SIZE=8 VIMD_GRAD_ACCUM=8 bash run.sh
```

---

## 10. Experiment tracking (optional)

Off by default. Without `VIMD_WANDB=1` nothing changes: no `wandb` import, no network
call, no login prompt — which matters because this job runs unattended for hours and a
tracker sitting on a prompt would waste a night.

`bash run.sh` turns it on; a bare `python3 main.py` leaves it off. To log to wandb.ai live,
authenticate once on the machine first:

```bash
wandb login                              # once per machine; the key lands in ~/.netrc
bash run.sh
```

### Checking the setup before a real run

`selfcheck.py` verifies the machine, the install and the W&B wiring in about a minute, and
**downloads nothing** — no model, no dataset, no Hub access. It builds a ~10k-parameter Whisper
from a config, writes a few synthetic shards in the cache format of §3, and drives the real
functions from `main.py` over them on the real GPU:

```bash
python3 selfcheck.py
```

It checks, and names the fix for whichever fails: the GPU is visible and runs a kernel; the pinned
dependencies import; W&B credentials resolve without a prompt; a run opens; `ViMDDataset` reads the
shard format; `Seq2SeqTrainer` trains **on the GPU** and logs into that same run rather than opening
a second; the tables, dialect charts, prediction samples and audio clips are all written; result
files upload as an artifact; and a restart reattaches to the same run instead of starting another.

Exit status is 0 only when every check passes. It writes to `outputs_selfcheck/` and removes it
afterwards (`VIMD_SELFCHECK_KEEP=1` keeps it), so it never touches `data/`, `outputs/` or any
checkpoint, and is safe to run while a real job is in progress.

It is not a substitute for the smoke test below: nothing here exercises the Hub download, the
PhoWhisper tokenizer, shard conversion, beam search or the repair pass.

### Handing the code to someone else to run

When the machine that runs this is not the machine that owns the W&B account, put the key in
`wandb.key` next to `run.sh` **before** you archive the tree. `run.sh` exports it, so whoever
runs the job types nothing and never sees a prompt:

```bash
printf '%s' 'YOUR_KEY' > wandb.key && chmod 600 wandb.key
tar czf vimd.tar.gz --exclude=data --exclude=outputs .
```

The file holds the key and nothing else. wandb accepts a key of **at least** 40 characters,
optionally with a variable-length prefix and a dash before them, so do not treat a longer key as
malformed — the one constraint wandb enforces is that it must not begin or end with whitespace,
which is why `run.sh` strips any and why `printf`/`-NoNewline` matter more than the length.

`wandb.key` is in `.gitignore` and must stay uncommitted. That is enforced, not merely advised:
GitHub enables push protection for Weights & Biases keys by default, so a commit carrying one is
rejected at push time, and a key that does reach a public repository is reported to W&B and
revoked — breaking tracking on the training host rather than protecting anything. A key that has
been committed is not recoverable by deleting the commit; rotate it at
[wandb.ai/authorize](https://wandb.ai/authorize).

Shipping no key at all is also a complete answer: the run records offline, and whoever owns the
account syncs it afterwards from their own machine, where they are already logged in.

```bash
wandb sync outputs/wandb/offline-run-*   # from the account owner's machine
```

**Without credentials the run still records — it does not stop to ask.** `main.py` checks for a
key the same way wandb does (`WANDB_API_KEY`, then the netrc entry for the API host) *before*
calling `wandb.init`, and falls back to offline recording when it finds none, printing the
`wandb sync` command to upload afterwards. This matters more than it sounds: wandb asks for a
missing key through `getpass()`, which returns without asking only when there is no tty — and
under tmux there is one, so an unconfigured host would otherwise hold the GPU at a password
prompt all night.

```bash
VIMD_WANDB=0 bash run.sh                 # or skip tracking altogether
```

**Credentials never live in this repository.** There is no config field for an API key
and none is ever read from a file in the tree. `wandb` picks up `wandb login` or, where
`$HOME` is quota-bound, `WANDB_API_KEY` from the environment. This repository is public;
a committed key would be scraped within minutes and cannot be un-published by deleting
the commit.

### What gets recorded

| Stage | Recorded |
|---|---|
| 2 — indexing | full recipe and dataset fingerprint as run config; per-split utterances, hours, speakers, provinces; label-token statistics |
| 3 — training | loss, learning rate, gradient norm, `eval_wer_norm` / `eval_wer_raw` / `eval_cer_norm` at every evaluation, plus GPU / VRAM / CPU / disk / network from wandb's own system monitor. Any OOM back-off records the batch sizes it fell back to, so the discontinuity it leaves in the loss curve has an explanation |
| 4 — selection | one row per candidate checkpoint with its beam-search validation WER, and the greedy score the Trainer selected on beside the beam score that actually decided |
| 5 — test | final WER/CER, WER broken down by region and by province, a sample of decoded utterances (worst half plus a seeded random half) optionally with audio, and `final_results.json` + both prediction CSVs as a versioned artifact |

The Trainer covers stage 3 only; everything else is logged explicitly by `main.py`.

### Resuming, and interruptions

The run id is stored in `<output_dir>/wandb_run.json`, so one output directory is one
W&B run no matter how many times the job is restarted — an interrupted run continues its
own curves instead of starting a second run beside them. Every exit path, including a
crash or `Ctrl-C`, closes the run with its exit status, so nothing is left showing as
*running* after the process is gone.

Everything wandb writes (run files, cache, artifacts, config) is kept under
`VIMD_OUTPUT_DIR`, for the same reason `HF_HOME` is kept inside the repo: a small `$HOME`
quota must not be able to kill an eleven-hour job.

**Tracking never ends a run.** Every call into wandb is guarded; the first failure logs a
warning and switches tracking off for the rest of the run. Losing the network at hour
nine costs the graphs, not the training.

### No network on the training host

```bash
WANDB_MODE=offline bash run.sh          # force offline even where a key exists
wandb sync outputs/wandb/offline-run-*     # later, from a machine with network
```

Each restart in offline mode writes its *own* `offline-run-*` directory even though the
run id is unchanged, so sync them in chronological order or the history interleaves.
Online mode resumes seamlessly and is preferable wherever the host can reach wandb.ai —
it already reaches the Hugging Face Hub.

### Hyperparameter sweeps

A sweep must give every trial its own `VIMD_OUTPUT_DIR`. Changing `VIMD_LR` while reusing
one directory is caught by the recipe guard in §8 and the trial aborts immediately —
correctly, since those checkpoints belong to a different experiment. Budget disk
accordingly: each trial retains up to `VIMD_SAVE_TOTAL_LIMIT` checkpoints at ~19 GB.

---

## 11. Repository layout

```
run.sh             entry point: install, then train
main.py            the entire pipeline, self-contained
selfcheck.py       one-minute GPU + W&B verification; downloads nothing
requirements.txt   pinned dependencies
README.MD          this file
.gitignore         the runtime directories below, which must never be committed
data/              created at runtime: the 16 kHz cache
outputs/           created at runtime: models, metrics, predictions, logs
```

`main.py` is deliberately a single file with no local imports: nothing can break from a partial
upload or a Python path surprise on the training server.

---

## 12. Citation

ViMD — Nguyen et al., *Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges*,
EMNLP 2024. PhoWhisper — Le et al., *PhoWhisper: Automatic Speech Recognition for Vietnamese*,
ICLR 2024 Tiny Papers.

ViMD is licensed **CC-BY-NC-ND-4.0** — non-commercial research use.

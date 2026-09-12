#!/usr/bin/env bash
#
# Single entry point for the training host:
#
#     bash run.sh
#
# Installs the pinned dependencies, then trains, selects a checkpoint and scores
# the test split. Nothing has to be placed on disk beforehand and no prompt is
# ever shown. The whole thing is idempotent: re-running after a crash, a timeout
# or a reboot resumes rather than starting over, so if in doubt, run it again.
#
# Expect ~2-3 h of one-time data preparation, then ~8-12 h of training.
# Because that outlives most SSH sessions, start it under tmux or screen:
#
#     tmux new -s vimd 'bash run.sh'      # detach with Ctrl-B then D
#
# The full log is written to outputs/logs/train.log either way.

set -euo pipefail
cd "$(dirname "$0")"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' 2>/dev/null; then
    echo "run.sh: Python 3.9+ is required, found: $(python3 --version 2>&1)" >&2
    exit 1
fi

# Pinned versions matter here - see the comments in requirements.txt - so the
# install is attempted every time rather than skipped when something is present.
echo "==> installing pinned dependencies"
if ! python3 -m pip install --quiet --requirement requirements.txt; then
    echo "run.sh: pip install failed (no network?); checking the existing environment" >&2
    if ! python3 -c 'import torch, transformers, jiwer, soundfile, pyarrow, scipy, numpy' 2>/dev/null; then
        echo "run.sh: dependencies are incomplete and cannot be installed. Fix the" >&2
        echo "        network or install requirements.txt by hand, then re-run." >&2
        exit 1
    fi
    echo "run.sh: continuing with the already-installed packages" >&2
fi

# Experiment tracking on by default when launched this way. main.py itself still
# defaults to off, so a bare `python3 main.py` keeps its original behaviour.
# Without credentials this records locally instead of prompting - main.py says so
# and prints the `wandb sync` command. Set VIMD_WANDB=0 to skip tracking entirely.
export VIMD_WANDB="${VIMD_WANDB:-1}"

# An API key shipped alongside this copy, so the training host never has to type
# one and nobody has to run `wandb login` there. wandb.key is in .gitignore: it
# travels inside the archive you hand over and never reaches the repository.
#
# That is not only a policy. GitHub enables push protection for Weights & Biases
# keys by default, so a commit carrying one is rejected at push time, and a key
# that does reach a public repository is reported to W&B and revoked - which
# would break tracking on the training host rather than secure anything.
#
# To arm it, on your own machine, before sending the code over:
#     printf '%s' 'YOUR_KEY' > wandb.key && chmod 600 wandb.key
if [ -z "${WANDB_API_KEY:-}" ] && [ -s wandb.key ]; then
    # tr strips the trailing newline a `echo > wandb.key` leaves behind, which
    # would otherwise be sent as part of the key and fail authentication.
    WANDB_API_KEY="$(tr -d '[:space:]' < wandb.key)"
    export WANDB_API_KEY
    echo "==> wandb: using the key shipped in wandb.key"
fi

echo "==> starting; full log also at outputs/logs/train.log"
exec python3 main.py

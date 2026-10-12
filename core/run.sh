#!/usr/bin/env bash
# core/run.sh -- run a command with the project's credentials loaded.
# Secrets live outside the repo in ~/.config/autoresearch/{modal,claude}.env (chmod 600):
#   modal.env:  MODAL_TOKEN_ID=... / MODAL_TOKEN_SECRET=...   (tavya vault item "modal_creds_new")
#   claude.env: CLAUDE_CODE_OAUTH_TOKEN=...                   (tavya vault item "claude")
#   ./core/run.sh python core/loop.py --project eeg_psych --task mdd_mumtaz --seed 1
set -euo pipefail
set -a
. "$HOME/.config/autoresearch/modal.env"
. "$HOME/.config/autoresearch/claude.env"
set +a
export PATH="$HOME/.local/bin:$PATH" CLAUDE_CONFIG_DIR="$HOME/.config/autoresearch/claude-cli"
cd "$(dirname "$0")/.."
exec "$@"

#!/usr/bin/env bash
# Run wrapper for Gleam SFT generation.
#
# Features:
#   - Auto-restarts on transient failures (up to MAX_RESTARTS)
#   - Logs everything to data/sft/run.log
#   - Works inside tmux for SSH resilience
#
# Usage:
#   # Start a tmux session and run:
#   ./scripts/run_generation.sh
#
#   # Or if already in tmux:
#   ./scripts/run_generation.sh --no-tmux
#
#   # Attach to running session:
#   tmux attach -t gleam-gen
#
#   # Check progress:
#   tail -f data/sft/generation.log
#   cat data/sft/generation_state.json | python -m json.tool

set -euo pipefail

# --- Config ---
PROVIDER="${PROVIDER:-neuralwatt}"
MAX_FILES="${MAX_FILES:-4700}"
BALANCE_LIMIT="${BALANCE_LIMIT:-0.50}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
SEED="${SEED:-42}"
NO_TMUX="${NO_TMUX:-false}"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/data/sft"
RUN_LOG="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR"

# --- API key check ---
if [ "$PROVIDER" = "neuralwatt" ] && [ -z "${NEURALWATT_API_KEY:-}" ]; then
    echo "ERROR: NEURALWATT_API_KEY not set"
    echo "  export NEURALWATT_API_KEY=sk-xxx"
    exit 1
fi
if [ "$PROVIDER" = "zai" ] && [ -z "${ZAI_API_KEY:-}" ]; then
    echo "ERROR: ZAI_API_KEY not set"
    echo "  export ZAI_API_KEY=xxx"
    exit 1
fi

# --- Tmux ---
if [ "$NO_TMUX" != "true" ] && [ -z "${TMUX:-}" ]; then
    # Kill existing session if present (stale from previous run)
    if tmux has-session -t gleam-gen 2>/dev/null; then
        echo "Killing stale tmux session 'gleam-gen'..."
        tmux kill-session -t gleam-gen
    fi

    echo "Starting in tmux session 'gleam-gen'..."
    echo "  Attach with: tmux attach -t gleam-gen"
    echo "  Detach with: Ctrl+B then D"
    tmux new-session -d -s gleam-gen -c "$PROJECT_DIR" \
        "NEURALWATT_API_KEY=${NEURALWATT_API_KEY:-} ZAI_API_KEY=${ZAI_API_KEY:-} PROVIDER=$PROVIDER MAX_FILES=$MAX_FILES BALANCE_LIMIT=$BALANCE_LIMIT SEED=$SEED NO_TMUX=true $0; echo '--- Generation finished at \$(date -u) ---'; sleep 86400"
    exit 0
fi

# --- Run loop ---
cd "$PROJECT_DIR"

# Ensure dependencies are installed
python3 -c "import openai, polars, huggingface_hub" 2>/dev/null || {
    echo "Installing dependencies..."
    pip3 install openai polars huggingface_hub -q
}

echo "=========================================="
echo "Gleam SFT Generation Run"
echo "  Provider:      $PROVIDER"
echo "  Max files:     $MAX_FILES"
echo "  Balance limit: \$$BALANCE_LIMIT"
echo "  Seed:          $SEED"
echo "  Max restarts:  $MAX_RESTARTS"
echo "  Started:       $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Log:           $RUN_LOG"
echo "=========================================="

restart=0
while [ $restart -lt $MAX_RESTARTS ]; do
    echo ""
    echo "--- Run attempt $((restart + 1))/$MAX_RESTARTS at $(date -u '+%H:%M:%S') ---" | tee -a "$RUN_LOG"

    python3 "$PROJECT_DIR/src/gleam/generate.py" \
        --provider "$PROVIDER" \
        --max-files "$MAX_FILES" \
        --balance-limit "$BALANCE_LIMIT" \
        --seed "$SEED" \
        2>&1 | tee -a "$RUN_LOG"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "" | tee -a "$RUN_LOG"
        echo "✅ Generation completed successfully at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$RUN_LOG"
        exit 0
    fi

    restart=$((restart + 1))
    if [ $restart -lt $MAX_RESTARTS ]; then
        delay=$((30 * restart))  # 30s, 60s, 90s, 120s
        echo "" | tee -a "$RUN_LOG"
        echo "⚠ Process exited with code $exit_code. Restarting in ${delay}s (attempt $restart/$MAX_RESTARTS)..." | tee -a "$RUN_LOG"
        sleep "$delay"
    fi
done

echo "" | tee -a "$RUN_LOG"
echo "❌ Failed after $MAX_RESTARTS restarts at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$RUN_LOG"
exit 1

#!/bin/bash
# slurm_ollama.sh - Allocate a Slurm node and spin up N Ollama instances with SSH tunnels.
#
# Usage: bash slurm_ollama.sh [N]
#   N = number of Ollama instances (default 2)
#
# Fill in the three USER CONFIG variables below before running.

# ─── USER CONFIG ─────────────────────────────────────────────────────────────

SRUN_CMD="srun --partition=ml --gres=gpu:2 --time=72:00:00 --pty bash"

OLLAMA_CMD="OLLAMA_CONTEXT_LENGTH=72000 OLLAMA_MODELS=/net/nfs6/gmssd/jdialpuri/.ollama OLLAMA_KEEP_ALIVE=24h ollama serve"

BATCH_CMD="python -m tooling.batch coot --agent --mmdb-only --with-gemmi --workers 2"

# ─────────────────────────────────────────────────────────────────────────────

N=${1:-2}
SESH="ollama-cluster"
BASE_PORT=11434
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build port list (space-separated for passing into subshells)
PORTS=()
for i in $(seq 0 $((N - 1))); do
    PORTS+=($((BASE_PORT + i)))
done
PORTS_STR="${PORTS[*]}"

# Build comma-separated host list for OLLAMA_HOSTS env var
HOSTS_CSV=$(for p in "${PORTS[@]}"; do printf "http://127.0.0.1:%s," "$p"; done | sed 's/,$//')

# ── Abort if session already exists ──────────────────────────────────────────
if tmux has-session -t "$SESH" 2>/dev/null; then
    echo "Session '$SESH' already exists. Attaching..."
    exec tmux attach-session -t "$SESH"
fi

# ── Write the orchestration script to a tempfile ─────────────────────────────
SETUP_SCRIPT=$(mktemp /tmp/slurm_ollama_setup_XXXXXX.sh)
trap "rm -f $SETUP_SCRIPT" EXIT

cat > "$SETUP_SCRIPT" <<SETUP_EOF
#!/bin/bash
SESH="$SESH"
N=$N
BASE_PORT=$BASE_PORT
PORTS_STR="$PORTS_STR"
OLLAMA_CMD='$OLLAMA_CMD'

echo "==> Waiting for Slurm node allocation..."
NODE=""
while [ -z "\$NODE" ]; do
    sleep 3
    NODE=\$(squeue --me --noheader -o "%N" 2>/dev/null | grep -v '^\s*\$' | head -1 || true)
done
echo "==> Allocated node: \$NODE"

# Populate ollama window with N panes
PANE=0
for PORT in \$PORTS_STR; do
    if [ "\$PANE" -gt 0 ]; then
        tmux split-window -t "\$SESH:ollama" -v
    fi
    tmux send-keys -t "\$SESH:ollama.\$PANE" "ssh -L 127.0.0.1:\${PORT}:localhost:\${PORT} \$NODE" C-m
    sleep 1
    tmux send-keys -t "\$SESH:ollama.\$PANE" "CUDA_VISIBLE_DEVICES=\$PANE OLLAMA_HOST=0.0.0.0:\$PORT \$OLLAMA_CMD" C-m
    PANE=\$((PANE + 1))
done
tmux select-layout -t "\$SESH:ollama" even-vertical

echo "==> All done. Tunnels established, batch is running."
SETUP_EOF
chmod +x "$SETUP_SCRIPT"

# ── Create all windows up front ───────────────────────────────────────────────
tmux new-session  -d -s "$SESH" -n "setup"
tmux new-window   -t "$SESH" -n "srun"
tmux new-window   -t "$SESH" -n "ollama"
tmux new-window   -t "$SESH" -n "batch"

# ── setup: runs orchestration so you can watch it ─────────────────────────────
tmux send-keys -t "$SESH:setup" "bash $SETUP_SCRIPT" C-m

# ── srun: allocate the node (keeps the job alive) ─────────────────────────────
tmux send-keys -t "$SESH:srun" "$SRUN_CMD" C-m

# ── batch: activate venv and run batch job ────────────────────────────────────
tmux send-keys -t "$SESH:batch" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "$SESH:batch" "source .venv/bin/activate" C-m
tmux send-keys -t "$SESH:batch" "ml compilers/llvm" C-m
tmux send-keys -t "$SESH:batch" "OLLAMA_HOSTS=$HOSTS_CSV $BATCH_CMD" 

# ── Attach immediately so you can watch everything unfold ─────────────────────
tmux select-window -t "$SESH:setup"
exec tmux attach-session -t "$SESH"

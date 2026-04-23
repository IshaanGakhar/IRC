#!/usr/bin/env bash
# Must be run with `source setup.sh` (so the venv stays activated afterwards).
#
# Two intended environments:
#   - EC2 (CPU instance):  runs embed_beir.py against the OpenAI API.
#                          Installs only the lean requirements.txt.
#   - RunPod (GPU pod):    runs embed_beir_minilm.py locally on the GPU.
#                          Installs requirements-minilm.txt (adds torch +
#                          sentence-transformers).
#
# Auto-detection:
#   * If `nvidia-smi` is present we assume GPU box -> install minilm extras.
#   * Otherwise CPU box -> install lean requirements only.
#   * Force either path with EMBED_PROFILE=cpu or EMBED_PROFILE=gpu.

_sourced=0
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    _sourced=1
elif [ -n "${ZSH_EVAL_CONTEXT:-}" ] && [[ "$ZSH_EVAL_CONTEXT" == *:file ]]; then
    _sourced=1
fi

if [ "$_sourced" -eq 0 ]; then
    echo "ERROR: run with 'source setup.sh' (not './setup.sh') so the venv activates."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || return 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. On Ubuntu: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
    return 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "python3-venv not available. Install with: sudo apt-get install -y python3-venv"
    return 1
fi

# Detect a broken/empty .venv from a previous failed run and rebuild.
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
    echo ">>> Found a broken .venv (no bin/activate) -- removing"
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo ">>> Creating virtualenv at .venv"
    python3 -m venv .venv || { echo "venv creation failed"; return 1; }
fi

echo ">>> Activating .venv"
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "activation failed"; return 1; }

echo ">>> Upgrading pip"
python -m pip install --upgrade pip >/dev/null

# ---- Profile selection ----
PROFILE="${EMBED_PROFILE:-auto}"
if [ "$PROFILE" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        PROFILE="gpu"
    else
        PROFILE="cpu"
    fi
fi

echo ">>> Profile: $PROFILE   (override with EMBED_PROFILE=cpu|gpu)"

if [ "$PROFILE" = "gpu" ]; then
    echo ">>> Installing requirements-minilm.txt (torch + sentence-transformers)"
    echo "    NOTE: on a RunPod PyTorch template, torch is preinstalled and pip will"
    echo "    leave it alone since requirements only pin torch>=2.2."
    pip install -r requirements-minilm.txt
else
    echo ">>> Installing requirements.txt (lean -- no torch / sentence-transformers)"
    pip install -r requirements.txt
fi

# ---- .env bootstrap ----
if [ ! -f ".env" ]; then
    echo "OPENAI_API_KEY=" > .env
fi

if ! grep -q '^OPENAI_API_KEY=.\+' .env 2>/dev/null; then
    echo ""
    echo "WARNING: .env has no OPENAI_API_KEY set. Required for embed_beir.py and the geo*.py scripts."
fi

cat <<'EOF'

==============================================================
  Setup complete. Virtualenv is active.
--------------------------------------------------------------
  geometry.py          -> uniform scalar quantization of hyperspherical angles with Jacobian-aware bit allocation vs float32/int8 baselines
  geo2.py              -> dynamic per-angle precision tiers (f32/f16/u8/u4/u2/clipped) assigned by greedy knapsack under a bit budget
  geo3.py              -> geo2 tier quantization plus a stacked int8 residual stage for higher fidelity at extra storage
  embed_beir.py        -> EC2 / any CPU box: embed BEIR corpora via OpenAI text-embedding-3-small (resumable, tqdm, run log)
  embed_beir_vllm.py   -> RunPod GPU pod: embed BEIR via a local vLLM Docker server (resumable, tqdm, run log)
  embed_beir_minilm.py -> RunPod GPU pod: same but model loaded in-process via sentence-transformers (alternative, no Docker)
  evaluate_quantization.py -> evaluate all quantization schemes on real BEIR embeddings; outputs table, CSV, tradeoff plot
==============================================================
  RUN PLAN
==============================================================

  --- A) On the EC2 (CPU) box: OpenAI embeddings ---
        # OPENAI_API_KEY in .env
        source setup.sh                          # auto-detects CPU profile
        python embed_beir.py \
            --data-dir ~/bier-data \
            --output-dir ./embeddings \
            --datasets fever
        # outputs:  ./embeddings/<dataset>/{corpus,queries}/chunk_*.npy

  --- B) On the RunPod (GPU) pod: MiniLM embeddings via vLLM Docker ---
        # On RunPod, use a custom Docker image:
        #   Image:  ai/all-minilm-l6-v2-vllm
        #   Expose: port 8000
        #   Volume: Network Volume at /workspace
        #
        # Once the pod is running, clone the repo and run:
        cd /workspace
        git clone <your-repo>
        cd <repo>
        pip install -r requirements.txt      # lean install, no torch needed

        unzip /workspace/fever.zip -d /workspace/bier-data/

        python embed_beir_vllm.py \
            --data-dir /workspace/bier-data \
            --output-dir /workspace/embeddings_vllm \
            --datasets fever --batch-size 512

        # outputs: /workspace/embeddings_vllm/<dataset>/{corpus,queries}/chunk_*.minilm-l6-vllm.npy
        #
        # The script waits for the server to be ready automatically.
        # If you need to check the server manually:
        #   curl http://localhost:8000/v1/models

  --- After both runs, collect them in one place ---
        # From the RunPod pod, push back to EC2 (or to your laptop):
        rsync -avP /workspace/embeddings_vllm/ ubuntu@<ec2-ip>:~/IRC/embeddings_vllm/
        # Now ~/IRC/embeddings/ (OpenAI, 1536-d) and ~/IRC/embeddings_vllm/ (MiniLM, 384-d)
        # live side-by-side and the geo*.py / downstream code can consume both.

==============================================================
  THROUGHPUT EXPECTATIONS
==============================================================
  embed_beir.py (OpenAI, network-bound):
        any CPU box (incl. t3.medium)  ~200-400 docs/s
        FEVER (~5.5M docs)             ~5-7 hours, ~$108 in API at default model

  embed_beir_vllm.py (vLLM Docker server on GPU):
        RTX 4090   ~5000-8000 docs/s   ->  12-18 min on FEVER
        A40 / A100 ~8000-20000 docs/s  ->   5-12 min on FEVER
        H100      ~20000-35000 docs/s  ->   3-5 min on FEVER

==============================================================
  RESUME + STORAGE
==============================================================
  - All scripts are resumable. Kill any time; rerun the same command and
    they pick up at the last flushed chunk (atomic tmp+rename writes).
  - Chunk filenames carry a model tag (".minilm-l6-vllm.") so EC2 (1536-d) and
    RunPod (384-d) outputs cannot be mixed even if dirs are merged.
  - Disk needs:
        FEVER raw zip            ~1.2 GB
        FEVER OpenAI embeddings ~32 GB   (5.5M * 1536 * 4)
        FEVER MiniLM embeddings  ~8 GB   (5.5M *  384 * 4)
==============================================================
EOF

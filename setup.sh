#!/usr/bin/env bash
# Must be run with `source setup.sh` (so the venv stays activated afterwards).

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

if [ ! -d ".venv" ]; then
    echo ">>> Creating virtualenv at .venv"
    python3 -m venv .venv || { echo "venv creation failed"; return 1; }
fi

echo ">>> Activating .venv"
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "activation failed"; return 1; }

echo ">>> Upgrading pip"
python -m pip install --upgrade pip >/dev/null

echo ">>> Installing requirements"
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    echo "OPENAI_API_KEY=" > .env
fi

if ! grep -q '^OPENAI_API_KEY=.\+' .env 2>/dev/null; then
    echo ""
    echo "WARNING: .env has no OPENAI_API_KEY set. Edit .env and put your key there before running the scripts."
fi

cat <<'EOF'

==============================================================
  Setup complete. Virtualenv is active.
--------------------------------------------------------------
  geometry.py          -> uniform scalar quantization of hyperspherical angles with Jacobian-aware bit allocation vs float32/int8 baselines
  geo2.py              -> dynamic per-angle precision tiers (f32/f16/u8/u4/u2/clipped) assigned by greedy knapsack under a bit budget
  geo3.py              -> geo2 tier quantization plus a stacked int8 residual stage for higher fidelity at extra storage
  embed_beir.py        -> embed every BEIR dataset's corpus+queries with OpenAI text-embedding-3-small (resumable, tqdm, run log)
  embed_beir_minilm.py -> OPTIONAL: same pipeline but with a local sentence-transformers model (default: all-MiniLM-L6-v2)

  Run any of them with:
      python geometry.py
      python geo2.py
      python geo3.py
      python embed_beir.py        --data-dir /path/to/bier-data --output-dir ./embeddings
      python embed_beir_minilm.py --data-dir /path/to/bier-data --output-dir ./embeddings_minilm

  (make sure OPENAI_API_KEY is set in .env for embed_beir.py)
--------------------------------------------------------------
  Notes on embed_beir_minilm.py (local model -- runs ON THIS MACHINE):

  - The model runs locally; no API. Speed is purely a function of the CPU/GPU
    you have. It auto-detects CUDA: if present it uses GPU, else CPU.

  - On a CPU-only instance (e.g. t3.medium, 2 vCPU) it will be very slow on
    big corpora -- expect ~30-80 docs/s before t3 burst credits drain, then
    ~10-20 docs/s sustained. Fine for small datasets (nfcorpus, scifact, fiqa)
    but rough for fever / msmarco / hotpotqa.
        python embed_beir_minilm.py --data-dir ~/bier-data \
            --datasets nfcorpus scifact --batch-size 32

  - On a GPU instance (g4dn.xlarge / g5.xlarge / g6.xlarge) it is dramatically
    faster (~3000-10000 docs/s). The default torch wheel from PyPI is CPU-only;
    on a GPU box reinstall torch with the right CUDA wheel BEFORE running:
        pip install --index-url https://download.pytorch.org/whl/cu121 torch
    Then:
        python embed_beir_minilm.py --data-dir ~/bier-data --datasets fever \
            --fp16 --batch-size 512

  - Output goes to ./embeddings_minilm/ by default (separate from the OpenAI
    embeddings in ./embeddings/). Chunk filenames carry a ".minilm-l6." tag so
    accidentally mixing the two corpora is impossible.

  - Resumable: kill it any time (Ctrl-C, instance reboot, OOM); rerunning the
    same command picks up at the last flushed chunk.
==============================================================
EOF

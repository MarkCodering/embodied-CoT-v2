#!/usr/bin/env bash
# One-time (idempotent) setup for the SimplerEnv / ManiSkill2_real2sim closed-loop
# physics simulator used by the "Closed-loop SimplerEnv rollout" section of
# Example.ipynb.
#
# Why this needs its own Python and venv, separate from the repo's main .venv:
#   - The simulator depends on sapien==2.2.2, which only ships wheels for
#     Python <=3.11 (this repo's main venv is 3.12).
#   - sapien 2.2.2's compiled bindings predate NumPy 2.0's C-API/ABI change;
#     with numpy>=2 installed, env.step() segfaults (matches
#     https://github.com/haosulab/SAPIEN/issues/238). So this venv is pinned
#     to numpy<2 (and correspondingly older scipy/opencv-python), which would
#     conflict with the main venv's transformers/opencv pins if shared.
#
# Safe to re-run: every step checks whether its output already exists first.
#
# Requires: an NVIDIA GPU + driver with a Vulkan ICD (checked below), network
# access to github.com/astral-sh (Python build) and github.com/simpler-env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLCHAINS_DIR="$REPO_ROOT/.toolchains"
PY311_DIR="$TOOLCHAINS_DIR/python3.11"
SIM_VENV="$REPO_ROOT/.venv-sim"
SIM_REPO_DIR="$REPO_ROOT/.sim/SimplerEnv"
PY_BUILD_TAG="20260814"
PY_BUILD_VERSION="3.11.16"
PY_BUILD_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_BUILD_TAG}/cpython-${PY_BUILD_VERSION}%2B${PY_BUILD_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"

log() { echo "[setup_sim_env] $*"; }

# --- sanity checks ---------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[setup_sim_env] ERROR: no working NVIDIA GPU/driver found (nvidia-smi failed)." >&2
    echo "SAPIEN requires a GPU to run; this setup cannot continue." >&2
    exit 1
fi

if [ -z "${VK_ICD_FILENAMES:-}" ] && [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ] \
   && [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    echo "[setup_sim_env] WARNING: no nvidia_icd.json found in the usual locations." >&2
    echo "  If sim setup later fails with a Vulkan device error, find it with:" >&2
    echo "    find / -iname nvidia_icd.json 2>/dev/null" >&2
    echo "  and pass it as VK_ICD_FILENAMES=<path> when launching the sim bridge." >&2
fi

# --- 1. standalone Python 3.11 (no root needed) -----------------------------
if [ -x "$PY311_DIR/bin/python3.11" ]; then
    log "Python 3.11 already present at $PY311_DIR"
else
    log "downloading standalone Python $PY_BUILD_VERSION ..."
    mkdir -p "$TOOLCHAINS_DIR"
    tmp_tar="$(mktemp)"
    curl -fL --retry 3 -o "$tmp_tar" "$PY_BUILD_URL"
    tmp_extract="$(mktemp -d)"
    tar xzf "$tmp_tar" -C "$tmp_extract"
    rm -f "$tmp_tar"
    rm -rf "$PY311_DIR"
    mv "$tmp_extract/python" "$PY311_DIR"
    rm -rf "$tmp_extract"
    log "Python 3.11 installed at $PY311_DIR"
fi

# --- 2. sim venv -------------------------------------------------------------
if [ -x "$SIM_VENV/bin/python" ]; then
    log "sim venv already present at $SIM_VENV"
else
    log "creating sim venv at $SIM_VENV ..."
    "$PY311_DIR/bin/python3.11" -m venv "$SIM_VENV"
    "$SIM_VENV/bin/pip" install -q --upgrade pip
fi
SIMPY="$SIM_VENV/bin/python"
SIMPIP="$SIM_VENV/bin/pip"

# --- 3. clone SimplerEnv + ManiSkill2_real2sim submodule --------------------
if [ -d "$SIM_REPO_DIR/.git" ]; then
    log "SimplerEnv repo already present at $SIM_REPO_DIR"
else
    log "cloning SimplerEnv ..."
    mkdir -p "$(dirname "$SIM_REPO_DIR")"
    git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv.git "$SIM_REPO_DIR"
fi

# --- 4. install ManiSkill2_real2sim + simpler_env ---------------------------
if "$SIMPY" -c "import mani_skill2_real2sim, simpler_env" >/dev/null 2>&1; then
    log "mani_skill2_real2sim / simpler_env already importable"
else
    log "installing mani_skill2_real2sim (this pulls sapien==2.2.2) ..."
    "$SIMPIP" install -q -e "$SIM_REPO_DIR/ManiSkill2_real2sim"
    log "installing simpler_env ..."
    "$SIMPIP" install -q -e "$SIM_REPO_DIR"
fi

# --- 5. pin numpy<2 (+ compatible scipy/opencv) — see header comment -------
# ManiSkill2_real2sim's requirements.txt leaves numpy/scipy unpinned, which
# pulls in numpy>=2 and a scipy/opencv-python that require it. Force back
# down every time, cheaply (pip no-ops if already satisfied).
log "pinning numpy<2 + compatible scipy/opencv-python for sapien 2.2.2 ..."
"$SIMPIP" install -q "numpy==1.24.4" "scipy==1.12.0" "opencv-python==4.9.0.80"

if ! "$SIMPIP" check >/dev/null 2>&1; then
    log "WARNING: pip check reports dependency conflicts in .venv-sim:"
    "$SIMPIP" check || true
fi

log "done. Sim Python: $SIMPY"
log "Smoke-test with:"
log "  VK_ICD_FILENAMES=\${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json} \\"
log "  $SIMPY $REPO_ROOT/scripts/sim/sim_bridge_server.py --task widowx_put_eggplant_in_basket --port 0"

#!/usr/bin/env bash
set -eu

source "$HOME/miniconda3/etc/profile.d/conda.sh"

PROJECT="/mnt/c/Users/lewei/Downloads/handoff - Copy - Copy"
cd "$PROJECT"

if conda env list | grep -q '^netket-lr '; then
    echo "netket-lr env already exists"
else
    conda env create -f environment.yml -y
fi

conda activate netket-lr

echo "Installing JAX with CUDA 12..."
pip uninstall -y jax jaxlib || true
pip install "jax[cuda12]==0.7.2"

echo "Verifying GPU..."
python -c "import jax, netket; print('devices:', jax.devices()); print('backend:', jax.default_backend()); print('netket:', netket.__version__)"

echo "Done. Activate with: conda activate netket-lr"

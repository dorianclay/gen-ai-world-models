# gen-ai-world-models

## Clone & Installation

1. **BEFORE CLONING** run `$ git lfs install` somewhere in your system (this only needs to be done once, to set up the hooks in your global git config)
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on your system.
3. Run `$ uv sync`.

## Running an experiment

1. To run a one-off experiment, use `uv run` instead of `python`: e.g. `$ uv run experiments/diffusion/train_antmaze.py`
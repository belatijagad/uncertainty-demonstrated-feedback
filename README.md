# Uncertainty Aware Alignment

---

## Installation

Install `uv` package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment using `uv` and install the project dependencies.

```bash
uv venv -p 3.13
source .venv/bin/activate
uv pip install -e .
```

Create `.env` to store token credentials for `huggingface_hub` and `wandb`

```bash
HUGGINGFACE_API_KEY=
WANDB_API_KEY=
GEMINI_API_KEY=
```

## Experiments

To run experiments, simply create a new configurations in `configs` folder, following the structure of `default_{method}.yaml` file. In order to perform experiments with the custom configuration, run the following command

```bash
uv run -m scripts/{method}.py --config-name configuration_name \
    class.param=value
```

For example,

```bash
uv run -m scripts/dpo.py --config-name pythia160m_dpo \
    trainer.epochs=10 optimizer.lr=5e-6
```

## Running Tests

```bash
uv run pytest -v
```

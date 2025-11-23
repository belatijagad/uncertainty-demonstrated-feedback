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
uv sync
uv pip install -e .
```

Create `.env` to store token credentials for `huggingface_hub` and `wandb`

```bash
HUGGINGFACE_API_KEY=
WANDB_API_KEY=
GEMINI_API_KEY=
```

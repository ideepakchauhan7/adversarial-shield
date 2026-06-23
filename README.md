# 🛡️ AdversarialShield

> Real-time adversarial prompt detector for LLMs.
> Fine-tuned DeBERTa-v3-base · 6 attack classes · OpenAI-compatible middleware · API + Streamlit demo.

## What It Does

AdversarialShield sits between user input and your LLM and classifies prompts into:

- `SAFE` - benign prompts
- `JAILBREAK` - role override and persona manipulation
- `PROMPT_INJECTION` - instruction injection attempts
- `HARMFUL_CONTENT` - dangerous or illegal requests
- `DATA_EXFILTRATION` - attempts to extract system instructions or context
- `SOCIAL_ENGINEERING` - impersonation and urgency manipulation

## Project Layout

- `data/` - dataset loading, cleaning, balancing, and splitting
- `model/` - training, evaluation, and inference
- `api/` - FastAPI app, schema models, and config
- `ui/` - Streamlit demo with API fallback to local checkpoint inference
- `tests/` - unit and integration tests
- `Dockerfile` - API image
- `Dockerfile.streamlit` - Streamlit UI image
- `docker-compose.yml` - local stack with API, UI, and Redis

## Quick Start

```bash
git clone https://github.com/ideepakchauhan7/adversarialshield
cd adversarialshield
cp .env.example .env
docker compose up --build
# Open http://localhost:8501 for the demo
```

The compose stack starts Redis, the FastAPI backend, and the Streamlit UI.
The UI can also classify locally from the saved checkpoint if the API is unavailable.

## Local Development

```bash
python -m model.train
./.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
cd ui && ../.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Run the API first, then the UI. If port `8501` is already in use, stop the old Streamlit process before starting a new one.

If you want autoreload while developing the API, use:

```bash
./.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

- API image: `docker build -t adversarialshield-api .`
- UI image: `docker build -f Dockerfile.streamlit -t adversarialshield-ui .`
- Full stack: `docker compose up --build`

The compose stack mounts `model/checkpoints/` into both containers so the API and the UI fallback can load the trained checkpoint.


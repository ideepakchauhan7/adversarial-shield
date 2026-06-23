import time, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis.asyncio as aioredis
from dotenv import load_dotenv

from model.inference import AdversarialShieldClassifier
from api.schemas import (
    ClassifyRequest, ClassifyResponse,
    BatchClassifyRequest, BatchClassifyResponse, HealthResponse,
)

load_dotenv()

MODEL_PATH  = os.getenv("CHECKPOINT_PATH", "./model/checkpoints/best_model")
THRESHOLD   = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")
RATE_LIMIT  = int(os.getenv("RATE_LIMIT_RPM", "60"))


# ── Global state ───────────────────────────────────────────────
_classifier: AdversarialShieldClassifier | None = None
_redis:       aioredis.Redis | None              = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _redis
    # Startup
    _classifier = AdversarialShieldClassifier(MODEL_PATH, threshold=THRESHOLD)
    _redis      = aioredis.from_url(REDIS_URL, decode_responses=True)
    yield
    # Shutdown
    await _redis.close()


app = FastAPI(
    title="AdversarialShield",
    description="Real-time LLM adversarial prompt detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate limiter dependency ─────────────────────────────────────
async def rate_limit(request: Request):
    if _redis is None:
        return
    client_ip = request.client.host
    key       = f"rl:{client_ip}"
    count     = await _redis.incr(key)
    if count == 1:
        await _redis.expire(key, 60)   # 60-second window
    if count > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 60 req/min.")


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    import torch
    return HealthResponse(
        status="ok",
        model=MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(
    body: ClassifyRequest,
    _: None = Depends(rate_limit),
):
    if _classifier is None:
        raise HTTPException(503, "Classifier not initialized")
    if body.threshold:
        _classifier.threshold = body.threshold
    result = _classifier.classify(body.text)
    return result


@app.post("/v1/batch", response_model=BatchClassifyResponse)
async def batch_classify(
    body: BatchClassifyRequest,
    _: None = Depends(rate_limit),
):
    if _classifier is None:
        raise HTTPException(503, "Classifier not initialized")
    t0      = time.perf_counter()
    results = _classifier.batch_classify(body.texts)
    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return BatchClassifyResponse(
        results=results,
        total=len(results),
        threats_found=sum(1 for r in results if r.is_threat),
        total_ms=total_ms,
    )


# Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
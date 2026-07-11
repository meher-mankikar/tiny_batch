"""
server/app.py

FastAPI HTTP server. Two endpoints:
  POST /generate  — submit a generation request, streams tokens back
  GET  /metrics   — returns live throughput and scheduler stats

Usage:
  uvicorn src.server.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..engine.engine import ModelEngine
from ..scheduler.request import GenerationRequest, Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- config ----------------------------------------------------------------

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# -- global engine ---------------------------------------------------------

engine: Optional[ModelEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    cfg = CONFIG
    engine = ModelEngine(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
        max_batch_size=cfg["scheduler"]["max_batch_size"],
    )
    # start the generation loop as a background task
    loop_task = asyncio.create_task(engine.run_generation_loop())
    yield
    loop_task.cancel()


app = FastAPI(title="tinybatch", lifespan=lifespan)

# -- request/response models -----------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = True


class GenerateResponse(BaseModel):
    request_id: str
    generated_text: str
    num_tokens: int
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]


# -- endpoints -------------------------------------------------------------


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """
    Submit a generation request.
    
    If stream=True (default): returns a text/event-stream of tokens as they're
    generated. Each event is a raw token string followed by newline.
    
    If stream=False: waits for the full response and returns JSON.
    """
    request_id = str(uuid.uuid4())[:8]

    # tokenize the prompt
    prompt_tokens = engine.tokenizer.encode(req.prompt, add_special_tokens=True)

    seq = Sequence(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        arrival_time=time.monotonic(),
    )

    engine.add_request(seq)

    if req.stream:
        return StreamingResponse(
            _stream_tokens(seq, request),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )
    else:
        return await _collect_response(seq)


async def _stream_tokens(seq: Sequence, http_request: Request) -> AsyncGenerator[str, None]:
    """
    Read tokens from the sequence's output queue and yield them as SSE.
    Handles client disconnect cleanly.
    """
    try:
        while True:
            # check if client disconnected
            if await http_request.is_disconnected():
                engine.abort_request(seq.request_id)
                return

            try:
                token_id = await asyncio.wait_for(seq.output_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield "data: [TIMEOUT]\n\n"
                return

            if token_id is None:
                # sentinel: generation finished
                yield "data: [DONE]\n\n"
                return

            token_str = engine.tokenizer.decode([token_id], skip_special_tokens=True)
            yield f"data: {token_str}\n\n"

    except asyncio.CancelledError:
        engine.abort_request(seq.request_id)
        raise


async def _collect_response(seq: Sequence) -> GenerateResponse:
    """Collect all tokens and return as a single response."""
    all_tokens = []
    while True:
        token_id = await asyncio.wait_for(seq.output_queue.get(), timeout=60.0)
        if token_id is None:
            break
        all_tokens.append(token_id)

    generated_text = engine.tokenizer.decode(all_tokens, skip_special_tokens=True)
    return GenerateResponse(
        request_id=seq.request_id,
        generated_text=generated_text,
        num_tokens=len(all_tokens),
        ttft_ms=seq.ttft_ms,
        total_latency_ms=seq.total_latency_ms,
    )


@app.get("/metrics")
async def metrics():
    """Live server metrics — throughput, batch stats, queue depth."""
    return engine.get_metrics()


@app.get("/health")
async def health():
    return {"status": "ok"}

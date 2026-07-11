"""
bench/static_server.py

Naive static batching baseline for comparison with continuous batching.

Strategy:
  - Collect requests for `batch_window_ms`
  - Run them all together as a single padded batch
  - Block until ALL sequences in the batch finish (pad to the longest)
  - Only then admit more requests

This is what most "just use model.generate()" code does, and it's what
continuous batching is designed to improve upon.

Run with:
  uvicorn src.bench.static_server:app --host 0.0.0.0 --port 8001
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

tokenizer = None
model = None
request_queue: asyncio.Queue = None
metrics_store = {
    "tokens_generated": 0,
    "requests_completed": 0,
    "start_time": None,
    "batch_sizes": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, request_queue
    cfg = CONFIG
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=getattr(torch, cfg["model"]["dtype"]),
    ).to(cfg["model"]["device"])
    model.eval()
    request_queue = asyncio.Queue()
    metrics_store["start_time"] = time.monotonic()
    task = asyncio.create_task(_static_batch_loop())
    yield
    task.cancel()


app = FastAPI(title="tinybatch-static", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = 1.0
    top_p: float = 0.9


@app.post("/generate")
async def generate(req: GenerateRequest):
    request_id = str(uuid.uuid4())[:8]
    result_future: asyncio.Future = asyncio.get_event_loop().create_future()
    await request_queue.put((request_id, req, result_future))
    result = await result_future
    return result


@app.get("/metrics")
async def metrics():
    elapsed = time.monotonic() - metrics_store["start_time"]
    avg_batch = (
        sum(metrics_store["batch_sizes"]) / len(metrics_store["batch_sizes"])
        if metrics_store["batch_sizes"] else 0.0
    )
    return {
        "tokens_per_second": metrics_store["tokens_generated"] / max(elapsed, 0.001),
        "requests_completed": metrics_store["requests_completed"],
        "avg_batch_size": round(avg_batch, 2),
    }


async def _static_batch_loop():
    """
    Core static batching loop.
    
    Waits for batch_window_ms, then processes everything that arrived.
    Key inefficiency: pads all sequences to the longest one, and blocks
    until even the shortest sequence has waited for the longest to finish.
    """
    cfg = CONFIG
    batch_window_ms = cfg["scheduler"]["batch_window_ms"]
    max_batch_size = cfg["scheduler"]["max_batch_size"]
    device = cfg["model"]["device"]
    loop = asyncio.get_event_loop()
    executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=1)

    while True:
        # wait for at least one request
        first = await request_queue.get()
        batch = [first]

        # collect more requests up to batch_window_ms
        deadline = time.monotonic() + batch_window_ms / 1000
        while time.monotonic() < deadline and len(batch) < max_batch_size:
            try:
                item = request_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

        metrics_store["batch_sizes"].append(len(batch))
        logger.info(f"[static] processing batch of {len(batch)}")

        # run in thread to not block event loop
        results = await loop.run_in_executor(executor, _run_batch, batch, device)

        for (_, _, future), result in zip(batch, results):
            future.set_result(result)
            metrics_store["requests_completed"] += 1


def _run_batch(batch, device):
    """
    Runs a static batch: pad all to same length, generate until max_new_tokens
    for the LONGEST request in the batch (others idle-compute after finishing).
    """
    request_ids = [b[0] for b in batch]
    reqs = [b[1] for b in batch]

    # tokenize
    prompt_tokens = [
        tokenizer.encode(req.prompt, add_special_tokens=True) for req in reqs
    ]
    max_prompt_len = max(len(t) for t in prompt_tokens)
    max_new = max(req.max_new_tokens for req in reqs)

    # left-pad
    pad_id = tokenizer.pad_token_id
    padded = [[pad_id] * (max_prompt_len - len(t)) + t for t in prompt_tokens]
    masks = [[0] * (max_prompt_len - len(t)) + [1] * len(t) for t in prompt_tokens]

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

    start = time.monotonic()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=reqs[0].temperature,
            top_p=reqs[0].top_p,
            pad_token_id=pad_id,
        )
    elapsed_ms = (time.monotonic() - start) * 1000

    results = []
    for i, (req, req_id) in enumerate(zip(reqs, request_ids)):
        generated = output_ids[i][len(padded[i]):]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        n_tokens = int((generated != pad_id).sum().item())
        metrics_store["tokens_generated"] += n_tokens
        results.append({
            "request_id": req_id,
            "generated_text": text,
            "num_tokens": n_tokens,
            "batch_latency_ms": round(elapsed_ms, 1),
        })

    return results

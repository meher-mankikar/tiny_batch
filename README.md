# tinybatch — a continuous batching inference server

A from-scratch implementation of continuous batching for autoregressive LLMs.

---

## What you're building

A working HTTP inference server that:
1. Accepts streaming generation requests
2. Maintains a **running batch** of active sequences
3. Uses **iteration-level scheduling** — at each forward pass, it can add new
   requests and evict finished ones without waiting for the whole batch to finish
4. Exposes metrics so you can actually see the throughput difference vs. static batching

---

## Background: why does this matter?

### The problem with static batching

In standard batching, you group N requests, run them together, and return when
**all** of them finish. This means a 5-token request sits idle while a 500-token
request finishes. GPU utilization collapses.

```
Static batching:
  Req A (500 tok): ████████████████████████████████████████
  Req B (5 tok):   █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (idle!)
  Req C (arrives while batch is running): WAITING...
```

### Continuous batching (Orca, 2022)

Schedule at the **iteration level**. After every single forward pass (one new
token generated), re-evaluate: can we add new requests? Can we remove finished
ones?

```
Continuous batching:
  Iter 1:  [A, B]      — B finishes after iter 2
  Iter 2:  [A, B]      — B done, C was waiting
  Iter 3:  [A, C]      — C slotted in immediately
  Iter 4:  [A, C, D]   — D also added
```

This is the core idea behind vLLM, TGI, and every modern serving system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      HTTP Server                         │
│              POST /generate  GET /metrics                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Request Queue                          │
│         (priority queue, FCFS by default)                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 Iteration Scheduler                      │
│                                                          │
│  every iteration:                                        │
│    1. evict finished sequences                           │
│    2. admit new sequences up to max_batch_size           │
│    3. build padded batch tensor                          │
│    4. return (input_ids, attention_mask, seq_metadata)   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Engine                           │
│                                                          │
│  - wraps HuggingFace AutoModelForCausalLM                │
│  - manages KV cache across steps (via past_key_values)   │
│  - runs one forward pass per iteration                   │
│  - applies sampling (greedy / top-p / temp)              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 Response Streams                         │
│   each request has an asyncio.Queue it reads tokens from │
└─────────────────────────────────────────────────────────┘
```

---

## File layout

```
tinybatch/
├── src/
│   ├── server/
│   │   └── app.py          # FastAPI app, /generate and /metrics endpoints
│   ├── scheduler/
│   │   ├── request.py      # Request and Sequence dataclasses
│   │   └── scheduler.py    # IterationScheduler
│   ├── engine/
│   │   ├── engine.py       # ModelEngine (the generation loop)
│   │   └── sampling.py     # Sampling strategies
│   └── bench/
│       ├── static_server.py   # Naive static batching baseline
│       └── benchmark.py       # Load generator + metrics comparison
├── tests/
│   ├── test_scheduler.py
│   └── test_engine.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Data model

### Request lifecycle

```
WAITING → RUNNING → FINISHED
           ↓
         PREEMPTED  (stretch goal: if batch is full, preempt lowest-priority)
```

### Sequence metadata (what the scheduler tracks per request)

```python
@dataclass
class Sequence:
    request_id: str
    prompt_tokens: List[int]        # encoded input
    generated_tokens: List[int]     # appended to this as we decode
    max_new_tokens: int
    status: SequenceStatus
    arrival_time: float
    first_token_time: Optional[float]  # for TTFT metric
    output_queue: asyncio.Queue     # tokens streamed back to HTTP handler
```

---

## The scheduler contract

```python
class IterationScheduler:
    def step(self) -> SchedulerOutput:
        """
        Called once per forward pass. Returns the batch to run.
        
        Responsibilities:
          1. Move FINISHED sequences out of running set
          2. Admit sequences from waiting queue into running set
             up to max_batch_size
          3. Construct batch: for each running seq, its current
             input token (just the last generated token, since
             we cache KV) plus its position id
        """
        ...

    def on_token_generated(self, request_id: str, token: int) -> None:
        """
        Called by engine after each forward pass.
        Appends token to sequence, checks stop conditions,
        updates status to FINISHED if needed.
        """
        ...
```

The key insight: once a sequence is in the running set and has gone through at
least one forward pass, its *input* on subsequent steps is just **one token**
(the last generated one) — the rest lives in the KV cache. So the scheduler only
needs to track position, not re-feed the whole prompt.

---

## The engine loop

```python
async def generation_loop(self):
    while True:
        if no running or waiting sequences:
            await asyncio.sleep(0.001)
            continue

        batch = self.scheduler.step()

        # forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch.input_ids,          # [batch, 1] after first step
                attention_mask=batch.attention_mask,
                past_key_values=batch.past_kv,      # per-sequence KV cache
                use_cache=True,
            )

        # sample next token per sequence
        for i, seq_id in enumerate(batch.sequence_ids):
            next_token = self.sampler(outputs.logits[i, -1, :])
            self.scheduler.on_token_generated(seq_id, next_token)
            # put token on the sequence's output queue for HTTP streaming
```

One tricky part: HuggingFace's `past_key_values` is a tuple of tensors for the
**whole batch**. When sequences finish and new ones join, you need to splice/pad
the KV cache. This is where the real implementation complexity lives. Start by
ignoring this (re-run prefill for every sequence every step) and then optimize.

---

## Metrics to implement

| Metric | Description |
|--------|-------------|
| `throughput_tokens_per_sec` | Total tokens generated / wall time |
| `ttft_ms` | Time to first token (per request, track p50/p99) |
| `tpot_ms` | Time per output token (after first token) |
| `batch_size_hist` | Distribution of running batch sizes over time |
| `queue_depth` | Waiting requests at each step |

Expose these at `GET /metrics` as JSON. The benchmark script will hit this.

---

## Benchmark design

The benchmark sends requests with **Poisson-distributed arrivals** and
**exponentially-distributed output lengths** (realistic approximation of real
traffic). You compare:

- **Static server**: collects requests for `batch_window_ms`, then runs them all,
  blocks until all finish, returns
- **Continuous server**: your implementation

```
# terminal 1 — continuous server
source .venv/bin/activate
uvicorn src.server.app:app --port 8000

# terminal 2 — static server
source .venv/bin/activate
uvicorn src.bench.static_server:app --port 8001

# terminal 3 — benchmark
source .venv/bin/activate
python -m src.bench.benchmark --mode both --arrival-rate 1.0 --num-requests 20
```

Expected result: at low arrival rates they're similar. At medium-to-high load,
continuous batching should show meaningfully better throughput and tail latency.
That's the "aha" moment.

---

## Implementation phases

### Phase 1 — correctness (no batching)
Get a single request working end-to-end: HTTP in → tokens generated → streamed
back. Use HuggingFace generate() internally. Don't worry about efficiency.

### Phase 2 — static batching baseline
Implement the naive server: wait `batch_window_ms`, group requests, run together,
return when all done. This is your baseline.

### Phase 3 — continuous batching (core)
Implement IterationScheduler and the generation loop. Start by re-running prefill
every step (wasteful but correct). Verify outputs match the static baseline.

### Phase 4 — KV cache reuse
Only feed one token per step for sequences past their first iteration. This
requires managing `past_key_values` per sequence and splicing on batch changes.
This is the hardest part.

### Phase 5 — benchmark + metrics
Wire up the benchmark script. Run experiments. Plot throughput vs. arrival rate
for static vs. continuous. See it work.

---

## Suggested model

**GPT-2** (124M) is ideal for this project:
- Runs fine on MPS / CPU
- Fast enough that you can iterate quickly
- HuggingFace integration is well-documented
- Small enough that KV cache splicing is manageable

For later phases, GPT-2 Medium (355M) or DistilGPT-2 works too.

---

## Things that will bite you

1. **Padding and attention masks**: when sequences in a batch are different
   lengths, you need left-padding and careful attention masking or logits will be
   wrong.

2. **KV cache indexing**: HuggingFace returns `past_key_values` as
   `(num_layers, 2, batch, heads, seq_len, head_dim)`. When you drop sequence i
   from the batch, you need to remove index i from every layer's KV tensors.

3. **Asyncio + PyTorch**: the generation loop is CPU-bound (even on MPS). Run it
   in a thread pool executor or it will block the event loop.

4. **Stop conditions**: EOS token, max_new_tokens, and client disconnect all need
   to cleanly remove a sequence from the running set.

---

## Resources

- [Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu) — the
  original continuous batching paper, very readable
- [vLLM paper](https://arxiv.org/abs/2309.06180) — PagedAttention; skip the KV
  memory parts for now, focus on the scheduling description
- [HuggingFace generate() source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
  — read this to understand what you're reimplementing
- [Dissecting batching effects in GPT inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  — good intuition-building post


## Progress
7/11 - Phase 4 forward with kv implementation done. Starting testing. 
- First testing single request. 
```
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_new_tokens": 30, "stream": false}'
```

- SEnd a few concurrent requests
```
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 20, "stream": false}' \
  -H "Content-Type: application/json" &

curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "A recipe for pasta:", "max_new_tokens": 20, "stream": false}' \
  -H "Content-Type: application/json" &

wait
```

results look correct:
```
{"request_id":"1992e613","generated_text":" in the long-term interest of its territorial integrity.\n\n\n\"France seems more hopeful about","num_tokens":19,"ttft_ms":408.4959589818027,"total_latency_ms":634.5784169971012}{"request_id":"7b8b457e","generated_text":" whole wheat spaghetti (if I'm honest, it's a good recipe), 1 Tbsp,","num_tokens":19,"ttft_ms":408.3047919848468,"total_latency_ms":634.3877500039525}[2]  + done       curl -X POST http://localhost:8000/generate -d  -H 
``` 

- run the benchmark
```
python -m src.bench.benchmark --mode both --arrival-rate 2.0 --num-requests 30
```

Results: KV cache improves throughput and latency and TTFT. 
```
==================================================
  Continuous Batching (port 8000)
==================================================
  Requests:          30
  Total time:        13.6s
  Throughput:        2.20 req/s
  Latency p50:       867ms
  Latency p95:       3316ms
  Latency p99:       4730ms
  TTFT p50:          38ms
  TTFT p99:          451ms

Running static batching test...
  sent 10/30 requests...
  sent 20/30 requests...
  sent 30/30 requests...
  waiting for 30 requests to complete...

==================================================
  Static Batching (port 8001)
==================================================
  Requests:          30
  Total time:        19.2s
  Throughput:        1.56 req/s
  Latency p50:       8215ms
  Latency p95:       10492ms
  Latency p99:       10536ms
```
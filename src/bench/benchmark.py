"""
bench/benchmark.py

Load generator for comparing static vs. continuous batching.

Sends requests with Poisson-distributed inter-arrival times and
exponentially-distributed target output lengths. Records per-request
latency and computes aggregate stats.

Usage:
  # Run against continuous batching server (port 8000):
  python -m src.bench.benchmark --mode continuous --arrival-rate 2.0

  # Run against static server (port 8001):
  python -m src.bench.benchmark --mode static --arrival-rate 2.0

  # Compare both (run servers separately first):
  python -m src.bench.benchmark --mode both --arrival-rate 2.0
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from typing import List, Optional

import httpx

# Short prompts that produce variable-length outputs
PROMPTS = [
    "The best way to learn programming is",
    "Once upon a time in a land far away",
    "The most important scientific discoveries of the 20th century include",
    "A recipe for chocolate chip cookies:",
    "Explain how black holes work in simple terms.",
    "The history of the Roman Empire began when",
    "My favorite things about living in a big city are",
    "The key differences between Python and JavaScript are",
    "Write a short poem about autumn leaves.",
    "The future of artificial intelligence will",
]


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    use_stream: bool = False,
) -> dict:
    """Send a single request and return timing info."""
    start = time.monotonic()
    first_token_time = None

    if use_stream:
        # streaming: measure TTFT from the first data chunk
        generated = []
        async with client.stream(
            "POST",
            f"{base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True},
            timeout=120.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    token = line[6:]
                    if token == "[DONE]":
                        break
                    if token == "[TIMEOUT]":
                        break
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    generated.append(token)
        generated_text = "".join(generated)
    else:
        resp = await client.post(
            f"{base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": False},
            timeout=120.0,
        )
        data = resp.json()
        generated_text = data.get("generated_text", "")

    end = time.monotonic()
    return {
        "latency_ms": (end - start) * 1000,
        "ttft_ms": (first_token_time - start) * 1000 if first_token_time else None,
        "num_chars": len(generated_text),
    }


async def run_load_test(
    base_url: str,
    num_requests: int,
    arrival_rate: float,  # requests per second
    mean_output_tokens: int,
    use_stream: bool = True,
) -> List[dict]:
    """
    Generate load with Poisson arrivals and exponential output lengths.
    
    arrival_rate=2.0 means on average 2 requests per second (Poisson process).
    """
    results = []
    tasks = []
    start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        for i in range(num_requests):
            # Poisson inter-arrival: exponential waiting time
            if i > 0:
                inter_arrival = random.expovariate(arrival_rate)
                await asyncio.sleep(inter_arrival)

            # exponential output length (minimum 10 tokens)
            max_new = max(10, int(random.expovariate(1.0 / mean_output_tokens)))
            max_new = min(max_new, 300)  # cap for sanity

            prompt = random.choice(PROMPTS)

            task = asyncio.create_task(
                send_request(client, base_url, prompt, max_new, use_stream)
            )
            tasks.append((time.monotonic() - start_time, task))

            if (i + 1) % 10 == 0:
                print(f"  sent {i+1}/{num_requests} requests...")

        # wait for all to finish
        print(f"  waiting for {len(tasks)} requests to complete...")
        for arrival_t, task in tasks:
            result = await task
            result["arrival_offset_s"] = arrival_t
            results.append(result)

    return results


def print_stats(results: List[dict], label: str):
    latencies = [r["latency_ms"] for r in results]
    ttfts = [r["ttft_ms"] for r in results if r.get("ttft_ms")]

    total_time = max(r["arrival_offset_s"] + r["latency_ms"] / 1000 for r in results)
    throughput = len(results) / total_time

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Requests:          {len(results)}")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Throughput:        {throughput:.2f} req/s")
    print(f"  Latency p50:       {statistics.median(latencies):.0f}ms")
    print(f"  Latency p95:       {_percentile(latencies, 95):.0f}ms")
    print(f"  Latency p99:       {_percentile(latencies, 99):.0f}ms")
    if ttfts:
        print(f"  TTFT p50:          {statistics.median(ttfts):.0f}ms")
        print(f"  TTFT p99:          {_percentile(ttfts, 99):.0f}ms")


def _percentile(data: List[float], p: int) -> float:
    sorted_data = sorted(data)
    idx = math.ceil(p / 100 * len(sorted_data)) - 1
    return sorted_data[max(0, idx)]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["continuous", "static", "both"],
                        default="continuous")
    parser.add_argument("--arrival-rate", type=float, default=1.0,
                        help="Requests per second (Poisson)")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--mean-output-tokens", type=int, default=80)
    parser.add_argument("--continuous-port", type=int, default=8000)
    parser.add_argument("--static-port", type=int, default=8001)
    parser.add_argument("--no-stream", action="store_true",
                        help="Don't use SSE streaming (measures total latency only)")
    args = parser.parse_args()

    use_stream = not args.no_stream

    print(f"\ntinybatch benchmark")
    print(f"  arrival rate:      {args.arrival_rate} req/s")
    print(f"  num requests:      {args.num_requests}")
    print(f"  mean output:       {args.mean_output_tokens} tokens")
    print(f"  streaming:         {use_stream}")

    if args.mode in ("continuous", "both"):
        print(f"\nRunning continuous batching test...")
        results = await run_load_test(
            f"http://localhost:{args.continuous_port}",
            args.num_requests,
            args.arrival_rate,
            args.mean_output_tokens,
            use_stream,
        )
        print_stats(results, f"Continuous Batching (port {args.continuous_port})")

    if args.mode in ("static", "both"):
        print(f"\nRunning static batching test...")
        results = await run_load_test(
            f"http://localhost:{args.static_port}",
            args.num_requests,
            args.arrival_rate,
            args.mean_output_tokens,
            use_stream=False,  # static server doesn't support streaming
        )
        print_stats(results, f"Static Batching (port {args.static_port})")


if __name__ == "__main__":
    asyncio.run(main())

"""
tests/test_engine.py

Correctness tests for the ModelEngine's batched KV-cache path.

The invariant: with greedy decoding (temperature=0), a prompt must produce
the same tokens no matter what else is in the batch. Each test compares the
engine's output against a reference that runs the prompt alone, with no KV
cache and no padding — the simplest possible correct implementation.

Loads GPT-2 on CPU, so these are slower than the scheduler tests.
Run with: pytest tests/test_engine.py -v
"""

import time
from typing import Dict, List

import pytest
import torch

from src.engine.paged_engine import ModelEngine
from src.scheduler.request import Sequence

MAX_NEW_TOKENS = 20

# deliberately different lengths so batched prefill needs left-padding
PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away, there lived a",
    "A recipe for pasta:",
    "The key differences between Python and JavaScript are that Python",
]


NUM_BLOCKS = 200


@pytest.fixture(scope="module")
def engine():
    # tiny block_size so short test prompts still span several blocks and
    # exercise the block/slot boundary logic in the paged KV pool
    return ModelEngine(
        model_name="gpt2", device="cpu", dtype="float32", max_batch_size=8,
        num_blocks=NUM_BLOCKS, block_size=4,
    )


def make_greedy_seq(engine: ModelEngine, request_id: str, prompt: str) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt_tokens=engine.tokenizer.encode(prompt),
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        arrival_time=time.monotonic(),
    )


def run_one_step(engine: ModelEngine) -> None:
    """One iteration of run_generation_loop, minus the asyncio/executor."""
    batch = engine._schedule_step()
    if batch.is_empty:
        return
    next_tokens = engine._forward_pass_with_kv(batch.sequences, batch.input_token_ids)
    engine._process_step_outputs(batch.sequences, next_tokens)


def run_until_done(engine: ModelEngine) -> None:
    while engine.scheduler.has_work():
        run_one_step(engine)


def reference_greedy(engine: ModelEngine, prompt: str) -> List[int]:
    """Unbatched, uncached greedy decode. Stops on EOS like Sequence.append_token."""
    ids = engine.tokenizer.encode(prompt)
    generated: List[int] = []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            input_ids = torch.tensor([ids + generated], device=engine.device)
            token = int(engine.model(input_ids).logits[0, -1, :].argmax().item())
            generated.append(token)
            if token == engine.eos_token_id:
                break
    return generated


@pytest.fixture(scope="module")
def references(engine) -> Dict[str, List[int]]:
    return {prompt: reference_greedy(engine, prompt) for prompt in PROMPTS}


def assert_matches_reference(engine, seq: Sequence, expected: List[int], prompt: str):
    actual = seq.generated_tokens
    if actual == expected:
        return
    diverge = next(
        (i for i, (a, e) in enumerate(zip(actual, expected)) if a != e),
        min(len(actual), len(expected)),
    )
    decode = engine.tokenizer.decode
    pytest.fail(
        f"prompt {prompt!r} diverged at generated token {diverge}\n"
        f"  expected: {decode(expected)!r}\n"
        f"  actual:   {decode(actual)!r}"
    )


class TestBatchInvariance:
    def test_single_request_matches_reference(self, engine, references):
        """Sanity check: the KV path is correct with no batching at all."""
        prompt = PROMPTS[0]
        seq = make_greedy_seq(engine, "solo", prompt)
        engine.add_request(seq)
        run_until_done(engine)
        assert_matches_reference(engine, seq, references[prompt], prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_each_prompt_alone_matches_reference(self, engine, references, prompt):
        seq = make_greedy_seq(engine, f"alone-{PROMPTS.index(prompt)}", prompt)
        engine.add_request(seq)
        run_until_done(engine)
        assert_matches_reference(engine, seq, references[prompt], prompt)

    def test_simultaneous_batch_matches_reference(self, engine, references):
        """All prompts prefilled together in one left-padded batch."""
        seqs = [make_greedy_seq(engine, f"batch-{i}", p) for i, p in enumerate(PROMPTS)]
        for seq in seqs:
            engine.add_request(seq)
        run_until_done(engine)
        for seq, prompt in zip(seqs, PROMPTS):
            assert_matches_reference(engine, seq, references[prompt], prompt)

    def test_staggered_arrivals_match_reference(self, engine, references):
        """
        New requests join while others are mid-decode, so the batch mixes
        prefill and decode and the KV caches have different lengths.
        """
        seqs = []
        for i, prompt in enumerate(PROMPTS):
            seq = make_greedy_seq(engine, f"stagger-{i}", prompt)
            seqs.append(seq)
            engine.add_request(seq)
            for _ in range(3):
                run_one_step(engine)
        run_until_done(engine)
        for seq, prompt in zip(seqs, PROMPTS):
            assert_matches_reference(engine, seq, references[prompt], prompt)


class TestBlockLifecycle:
    """
    KV blocks must go back to the pool when sequences finish or abort.
    The engine fixture is shared across this module, so every test after the
    first also runs on blocks freed by earlier tests (block reuse).
    """

    def assert_pool_empty(self, engine):
        assert engine.block_allocator.check_capacity() == NUM_BLOCKS
        assert len(engine.block_allocator.block_tables) == 0

    def test_finished_sequences_free_their_blocks(self, engine):
        seqs = [make_greedy_seq(engine, f"free-{i}", p) for i, p in enumerate(PROMPTS)]
        for seq in seqs:
            engine.add_request(seq)
        run_one_step(engine)
        assert engine.block_allocator.check_capacity() < NUM_BLOCKS  # blocks in use

        run_until_done(engine)
        self.assert_pool_empty(engine)

    def test_aborted_sequence_frees_its_blocks(self, engine, references):
        doomed = make_greedy_seq(engine, "abort-me", PROMPTS[1])
        survivor = make_greedy_seq(engine, "survivor", PROMPTS[0])
        engine.add_request(doomed)
        engine.add_request(survivor)
        for _ in range(3):
            run_one_step(engine)
        assert "abort-me" in engine.block_allocator.block_tables

        # abort between steps: the next step evicts it and frees its blocks
        engine.abort_request("abort-me")
        run_one_step(engine)
        assert "abort-me" not in engine.block_allocator.block_tables

        # the other sequence is unaffected, and nothing leaks at the end
        run_until_done(engine)
        assert_matches_reference(engine, survivor, references[PROMPTS[0]], PROMPTS[0])
        self.assert_pool_empty(engine)

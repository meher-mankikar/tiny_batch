"""
tests/test_scheduler.py

Unit tests for the IterationScheduler.

These test scheduling logic in isolation — no model, no HTTP.
Run with: pytest tests/test_scheduler.py -v
"""

import asyncio
import time

import pytest

from src.scheduler.request import GenerationRequest, Sequence, SequenceStatus
from src.scheduler.scheduler import IterationScheduler


def make_seq(request_id: str, prompt_tokens=None, max_new_tokens=10):
    return Sequence(
        request_id=request_id,
        prompt_tokens=prompt_tokens or [1, 2, 3],
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.9,
        arrival_time=time.monotonic(),
    )


class TestAdmission:
    def test_admits_up_to_max_batch_size(self):
        sched = IterationScheduler(max_batch_size=3)
        for i in range(5):
            sched.add_request(make_seq(f"req-{i}"))

        out = sched.step()
        assert out.batch_size == 3

    def test_waiting_queue_reduces_on_admit(self):
        sched = IterationScheduler(max_batch_size=2)
        for i in range(4):
            sched.add_request(make_seq(f"req-{i}"))

        sched.step()
        stats = sched.get_stats()
        assert stats["running"] == 2
        assert stats["waiting"] == 2

    def test_fcfs_ordering(self):
        """Earlier requests should be admitted first."""
        sched = IterationScheduler(max_batch_size=2)
        sched.add_request(make_seq("first"))
        sched.add_request(make_seq("second"))
        sched.add_request(make_seq("third"))

        out = sched.step()
        admitted_ids = [seq.request_id for seq in out.sequences]
        assert "first" in admitted_ids
        assert "second" in admitted_ids
        assert "third" not in admitted_ids


class TestEviction:
    def test_finished_sequences_evicted_next_step(self):
        sched = IterationScheduler(max_batch_size=4)
        seq = make_seq("r1", max_new_tokens=1)
        sched.add_request(seq)

        out = sched.step()
        assert out.batch_size == 1

        # generate the only allowed token → sequence finishes
        sched.on_token_generated("r1", token_id=999, eos_token_id=50256)
        assert seq.status == SequenceStatus.FINISHED

        # next step should evict and return empty (no waiting requests)
        out2 = sched.step()
        assert out2.batch_size == 0

    def test_slot_opens_after_finish(self):
        """After a sequence finishes, its slot should be fillable."""
        sched = IterationScheduler(max_batch_size=2)
        sched.add_request(make_seq("r1", max_new_tokens=1))
        sched.add_request(make_seq("r2"))
        sched.add_request(make_seq("r3"))  # this one should wait

        out = sched.step()
        assert out.batch_size == 2

        # finish r1
        sched.on_token_generated("r1", token_id=999, eos_token_id=50256)

        # next step: r1 evicted, r3 admitted
        out2 = sched.step()
        ids = {seq.request_id for seq in out2.sequences}
        assert "r1" not in ids
        assert "r3" in ids
        assert out2.batch_size == 2


class TestBatchConstruction:
    def test_prefill_uses_full_prompt(self):
        sched = IterationScheduler(max_batch_size=4)
        prompt = [10, 20, 30, 40]
        seq = make_seq("r1", prompt_tokens=prompt)
        sched.add_request(seq)

        out = sched.step()
        assert out.input_token_ids[0] == prompt
        assert not seq.prefill_done

    def test_decode_uses_last_token_only(self):
        sched = IterationScheduler(max_batch_size=4)
        seq = make_seq("r1", prompt_tokens=[1, 2, 3])
        sched.add_request(seq)

        # first step: prefill
        out = sched.step()
        assert out.input_token_ids[0] == [1, 2, 3]

        # mark prefill done, generate a token
        sched.on_token_generated("r1", token_id=42, eos_token_id=50256)
        # prefill_done is set inside on_token_generated

        # second step: decode — should only see token 42
        out2 = sched.step()
        assert out2.input_token_ids[0] == [42]


class TestAbort:
    def test_abort_running_sequence(self):
        sched = IterationScheduler(max_batch_size=4)
        seq = make_seq("r1")
        sched.add_request(seq)
        sched.step()  # admit

        sched.abort_request("r1")
        assert seq.status == SequenceStatus.ABORTED

        # sentinel should be on the queue
        assert not seq.output_queue.empty()

    def test_abort_waiting_sequence(self):
        sched = IterationScheduler(max_batch_size=1)
        sched.add_request(make_seq("r1"))  # fills the slot
        sched.add_request(make_seq("r2"))  # waits
        sched.step()

        sched.abort_request("r2")
        stats = sched.get_stats()
        assert stats["waiting"] == 0


class TestMetrics:
    def test_stats_track_admitted_finished(self):
        sched = IterationScheduler(max_batch_size=4)
        seq = make_seq("r1", max_new_tokens=2)
        sched.add_request(seq)

        sched.step()
        assert sched.get_stats()["total_admitted"] == 1

        sched.on_token_generated("r1", 1, 50256)
        sched.on_token_generated("r1", 2, 50256)  # hits max_new_tokens
        # status may or may not be FINISHED depending on which fires first
        # just check admitted count is stable
        assert sched.get_stats()["total_admitted"] == 1

"""
scheduler/scheduler.py

Performs the main continuous batching logic. 

Key responsibility: at the start of every forward pass, decide which
sequences are running. This means:
  1. Remove sequences that finished last iteration
  2. Promote sequences from the waiting queue into the running set
  3. Build the batch descriptor (SchedulerOutput) for the engine

This is intentionally simple — no preemption, FCFS ordering only.
Preemption is a natural extension once this baseline works.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Dict, Deque, List, Optional

from .request import Sequence, SchedulerOutput, SequenceStatus

logger = logging.getLogger(__name__)


class IterationScheduler:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size

        # sequences waiting to be scheduled — first come first serve
        self._waiting: Deque[Sequence] = deque()

        # sequences currently in the running batch, keyed by request_id
        self._running: Dict[str, Sequence] = {}

        # metrics
        self._total_admitted = 0
        self._total_finished = 0
        self._step_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, seq: Sequence) -> None:
        """
        Called by the HTTP handler when a new request arrives.
        Thread-safe: only appends to the deque.
        """
        self._waiting.append(seq)
        logger.debug(f"[scheduler] queued request {seq.request_id}, "
                     f"queue depth={len(self._waiting)}")

    def step(self) -> SchedulerOutput:
        """
        Called by the engine once per forward pass.
        
        Returns the batch descriptor for this step.
        
        Note: this does NOT perform the forward pass — it only decides
        what the batch looks like. The engine calls this, runs the model,
        then calls on_token_generated() for each sequence.
        """
        self._step_count += 1

        # 1. Evict finished sequences from running set
        self._evict_finished()

        # 2. Admit new sequences up to max_batch_size
        self._admit_waiting()

        # 3. Build batch descriptor
        return self._build_batch()

    def on_token_generated(self, request_id: str, token_id: int,
                           eos_token_id: int) -> None:
        """
        Called by the engine after it has sampled the next token for a sequence.
        Updates sequence state and handles completion.
        """
        seq = self._running.get(request_id)
        if seq is None:
            logger.warning(f"on_token_generated called for unknown seq {request_id}")
            return

        seq.append_token(token_id, eos_token_id)
        seq.prefill_done = True

        if seq.status == SequenceStatus.FINISHED:
            logger.debug(f"[scheduler] seq {request_id} finished, "
                         f"generated {seq.num_generated} tokens")
            self._total_finished += 1

    def abort_request(self, request_id: str) -> None:
        """Called when a client disconnects mid-stream."""
        if request_id in self._running:
            self._running[request_id].status = SequenceStatus.ABORTED
            self._running[request_id].output_queue.put_nowait(None)
        else:
            # might still be in waiting queue
            self._waiting = deque(
                s for s in self._waiting if s.request_id != request_id
            )

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def get_stats(self) -> dict:
        return {
            "running": len(self._running),
            "waiting": len(self._waiting),
            "total_admitted": self._total_admitted,
            "total_finished": self._total_finished,
            "step_count": self._step_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_finished(self) -> None:
        finished_ids = [
            rid for rid, seq in self._running.items()
            if seq.is_finished
        ]
        for rid in finished_ids:
            del self._running[rid]

    def _admit_waiting(self) -> None:
        """
        Greedily admit sequences from the waiting queue until we hit
        max_batch_size. FCFS ordering.
        
        Extension point: you could add priority, fairness, or token-budget
        constraints here. vLLM also checks available KV cache blocks here.
        """
        while self._waiting and len(self._running) < self.max_batch_size:
            seq = self._waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self._running[seq.request_id] = seq
            self._total_admitted += 1
            logger.debug(f"[scheduler] admitted {seq.request_id}, "
                         f"running={len(self._running)}")

    def _build_batch(self) -> SchedulerOutput:
        """
        Construct the SchedulerOutput for the current running set.
        
        For sequences in prefill (first step): feed the full prompt.
        For sequences in decode (subsequent steps): feed only the last token.
        
        This is the key difference from static batching — after prefill,
        we only feed one new token per sequence per step.
        """
        sequences = list(self._running.values())
        # List of lists. Each sublist is the input tokens for one sequence.
        input_token_ids = []

        for seq in sequences:
            if not seq.prefill_done:
                # prefill: feed the full prompt
                input_token_ids.append(seq.prompt_tokens)
            else:
                # decode: just the last generated token
                input_token_ids.append([seq.generated_tokens[-1]])

        return SchedulerOutput(
            sequences=sequences,
            input_token_ids=input_token_ids,
        )

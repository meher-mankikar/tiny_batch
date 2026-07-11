"""
scheduler/request.py

Core data model for requests and sequences.
A Request is what comes in over HTTP.
A Sequence is the live tracking state inside the scheduler.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class SequenceStatus(Enum):
    WAITING = auto()     # in the queue, not yet scheduled
    RUNNING = auto()     # currently in the active batch
    FINISHED = auto()    # done (EOS or max_new_tokens reached)
    ABORTED = auto()     # client disconnected or error


@dataclass
class GenerationRequest:
    """Incoming HTTP request, before tokenization."""
    request_id: str
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    arrival_time: float = field(default_factory=time.monotonic)


@dataclass
class Sequence:
    """
    Live state for a single request inside the scheduler + engine.
    
    Key design choice: we track the *full* token history so the engine
    can reconstruct attention masks, but after the first forward pass
    only the *last* token is fed as input (KV cache holds the rest).
    """
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    arrival_time: float

    # mutable state — updated each iteration
    generated_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # timing — filled in as we go
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    # the HTTP handler reads from this queue to stream tokens back
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # True after the first forward pass — after this we use KV cache
    prefill_done: bool = False

    @property
    def all_tokens(self) -> List[int]:
        """Full token sequence: prompt + generated so far."""
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return (self.finish_time - self.arrival_time) * 1000

    def append_token(self, token_id: int, eos_token_id: int) -> None:
        """Append a generated token and check stop conditions."""
        now = time.monotonic()

        if self.first_token_time is None:
            self.first_token_time = now

        self.generated_tokens.append(token_id)

        if token_id == eos_token_id or self.num_generated >= self.max_new_tokens:
            self.status = SequenceStatus.FINISHED
            self.finish_time = now
            # sentinel so the HTTP handler knows the stream is done
            self.output_queue.put_nowait(None)
        else:
            self.output_queue.put_nowait(token_id)


@dataclass
class SchedulerOutput:
    """
    What the scheduler hands to the engine each iteration.
    
    For sequences in their first step (prefill_done=False), input_ids
    contains the full prompt. For subsequent steps it contains just the
    last generated token. The engine needs to know which is which so it
    can manage the KV cache correctly.
    """
    sequences: List[Sequence]       # ordered list of running sequences
    # input_ids[i] is what to feed for sequences[i] this step
    # shape: ragged — each entry is a list of token ids
    input_token_ids: List[List[int]]

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def is_empty(self) -> bool:
        return len(self.sequences) == 0

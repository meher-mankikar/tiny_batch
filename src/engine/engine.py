"""
engine/engine.py

The ModelEngine owns the model, the tokenizer, and the generation loop.
It runs as a background asyncio task that continuously:
  1. Asks the scheduler what to run
  2. Builds the padded batch tensor
  3. Runs one forward pass
  4. Dispatches generated tokens back to sequences

--- KV cache strategy (Phase 3 / 4 progression) ---

Phase 3 (start here): re-run prefill every step for every sequence.
Wasteful but correct and simple. No KV cache management needed.

Phase 4 (optimize): maintain per-sequence past_key_values and only
feed one token per step for decode. Requires splicing KV tensors when
the batch changes. See _splice_past_key_values() stub.

The engine is written for Phase 3 by default. Phase 4 additions are
marked with # [PHASE 4] comments.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ..scheduler.request import Sequence, SequenceStatus
from ..scheduler.scheduler import IterationScheduler
from .sampling import sample_token

logger = logging.getLogger(__name__)


class ModelEngine:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "float32",
        max_batch_size: int = 8,
    ):
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.max_batch_size = max_batch_size

        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.eos_token_id = self.tokenizer.eos_token_id

        self.scheduler = IterationScheduler(max_batch_size=max_batch_size)

        # metrics
        self._tokens_generated = 0
        self._start_time = time.monotonic()
        self._batch_sizes: List[int] = []

        # [PHASE 4] per-sequence KV cache storage
        self._kv_cache: Dict[str, tuple] = {}

        # thread pool for running torch in a thread so it doesn't block
        # the asyncio event loop
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, seq: Sequence) -> None:
        self.scheduler.add_request(seq)

    def abort_request(self, request_id: str) -> None:
        self.scheduler.abort_request(request_id)
        self._kv_cache.pop(request_id, None)

    async def run_generation_loop(self) -> None:
        """
        Main generation loop. Run this as a background asyncio task.
        
        We offload the actual torch work to a thread executor to avoid
        blocking the event loop (which would stall HTTP responses).
        """
        logger.info("Generation loop started.")
        loop = asyncio.get_event_loop()

        while True:
            try:
                if not self.scheduler.has_work():
                    await asyncio.sleep(0.001)
                    continue

                batch = self.scheduler.step()

                if batch.is_empty:
                    await asyncio.sleep(0.001)
                    continue

                self._batch_sizes.append(batch.batch_size)

                next_tokens = await loop.run_in_executor(
                    self._executor,
                    self._forward_pass_with_kv,
                    batch.sequences,
                    batch.input_token_ids,
                )

                for seq, token_id in zip(batch.sequences, next_tokens):
                    self.scheduler.on_token_generated(
                        seq.request_id, token_id, self.eos_token_id
                    )
                    self._tokens_generated += 1
            except Exception as e:
                logger.exception(f"Generation loop error: {e}")
                await asyncio.sleep(0.1)

    def get_metrics(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        avg_batch = (
            sum(self._batch_sizes) / len(self._batch_sizes)
            if self._batch_sizes else 0.0
        )
        return {
            "tokens_per_second": self._tokens_generated / max(elapsed, 0.001),
            "total_tokens_generated": self._tokens_generated,
            "avg_batch_size": round(avg_batch, 2),
            "uptime_seconds": round(elapsed, 1),
            **self.scheduler.get_stats(),
        }

    # ------------------------------------------------------------------
    # Forward pass (runs in thread executor)
    # ------------------------------------------------------------------

    def _forward_pass(
        self,
        sequences: List[Sequence],
        input_token_ids: List[List[int]],
    ) -> List[int]:
        """
        Run one forward pass for the current batch.
        Returns a list of next token ids, one per sequence.
        
        Phase 3 implementation: re-runs full context every step.
        This is O(n^2) in sequence length but correct and simple.
        """
        # build padded input tensor
        # left-pad so the rightmost token is the one we're predicting from
        input_ids, attention_mask = self._pad_sequences(input_token_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,  # [PHASE 4]: set True and manage past_key_values
            )

        # outputs.logits: [batch, seq_len, vocab_size]
        # we want the logits at the LAST non-padded position for each sequence
        next_tokens = []
        for i, seq in enumerate(sequences):
            # last position logits
            last_logit = outputs.logits[i, -1, :]
            token = sample_token(last_logit, seq.temperature, seq.top_p)
            next_tokens.append(token)

        return next_tokens

    def _pad_sequences(
        self,
        token_id_lists: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Left-pad a ragged batch of token sequences.
        
        Returns:
          input_ids:      [batch, max_len]
          attention_mask: [batch, max_len]  (1 = real token, 0 = pad)
        """
        max_len = max(len(ids) for ids in token_id_lists)
        pad_id = self.tokenizer.pad_token_id

        padded = []
        masks = []
        for ids in token_id_lists:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # [PHASE 4] KV cache management stubs
    # This implementation will use a kv cache and only pass in the previous token to the forward pass.
    # ------------------------------------------------------------------
    def _extract_sequence_kv(self, batched_past_kv: tuple, batch_idx: int) -> tuple:
        """Pull one sequence's KV out of a batched past_key_values tuple."""
        # iterate over the layers. for each layer the key, vlaue are shape 
        # (batch, heads, seq_len, head_dim)
        # slicing [batch_idx : batch_idx + 1] pulls out one sequ's kv state without 
        # squashing the batch dimension (size 1). 
        # print(f"layer 0 len: {len(batched_past_kv[0])}") # len 2
        # print(f"layer 0 shapes: {[x.shape for x in batched_past_kv[0]]}") # torch.Size([1, 12, seq_len at thbat time, 64])]
        return tuple(
            (key[batch_idx : batch_idx + 1], value[batch_idx : batch_idx + 1])
            for key, value in batched_past_kv
        )

    def _merge_sequence_kv(self, kv_list: List[tuple]) -> tuple:
        """Stack per-sequence KV tuples into one batched past_key_values tuple."""
        num_layers = len(kv_list[0])

        # find the max seq_len across all sequences for this merge
        max_seq_len = max(kv[0][0].shape[-2] for kv in kv_list)

        def pad_to(tensor, target_len):
            # tensor shape: [1, heads, seq_len, head_dim]
            seq_len = tensor.shape[-2]
            if seq_len == target_len:
                return tensor
            pad_len = target_len - seq_len
            # left-pad the seq_len dimension with zeros
            padding = torch.zeros(
                *tensor.shape[:-2], pad_len, tensor.shape[-1],
                dtype=tensor.dtype, device=tensor.device
            )
            return torch.cat([padding, tensor], dim=-2)

        # for each layer, concatenate the individual key tensors (shape [1, heads, seq_len, head_dim])
        # along dim=0 to get [batch, heads, seq_len, head_dim]. same for values. 
        # this is inverse of _extract_sequence_kv. 
        return tuple(
            (
                torch.cat([pad_to(kv[layer_idx][0], max_seq_len) for kv in kv_list], dim=0), # keys
                torch.cat([pad_to(kv[layer_idx][1], max_seq_len) for kv in kv_list], dim=0), # values
            )
            for layer_idx in range(num_layers)
        )

    # def _build_decode_attention_mask(
    #     self,
    #     seq_ids: List[int],
    #     sequences: List[Sequence],
    # ) -> torch.Tensor:
    #     """Full-length attention mask (cached tokens + new token) for decode."""
    #     masks = []
    #     for sid in seq_ids:
    #         request_id = sequences[sid].request_id
    #         # first [0] is layer 0, second [0] is they key tensor, shape[-2] is the sequence length dim,. 
    #         # this tells how many tokens are already in the cache for this sequence. 
    #         # mask needs to cover all those positions plus the one new token we're about to feed, hence cahced_len + 1. 
    #         # different from prefill mask whcih only covers the prompt tokens. 
    #         cached_len = self._kv_cache[request_id][0][0].shape[-2]
    #         masks.append([1] * (cached_len + 1))
    #     return torch.tensor(masks, dtype=torch.long, device=self.device)

    # def _extract_sequence_kv(self, batched_cache: DynamicCache, batch_idx: int) -> DynamicCache:
    #     # DynamicCache stores keys and values as lists indexed by layer. 
    #     # cache.key_cache[layer_idx]    # shape: [batch, heads, seq_len, head_dim]
    #     # cache.value_cache[layer_idx]  # shape: [batch, heads, seq_len, head_dim]
    #     single = DynamicCache()
    #     for layer_idx in range(len(batched_cache.key_cache)):
    #         single.update(
    #             batched_cache.key_cache[layer_idx][batch_idx:batch_idx+1],
    #             batched_cache.value_cache[layer_idx][batch_idx:batch_idx+1],
    #             layer_idx,
    #         )
    #     return single

    # def _merge_sequence_kv(self, kv_list: List[DynamicCache]) -> DynamicCache:
    #     merged = DynamicCache()
    #     for layer_idx in range(len(kv_list[0].key_cache)):
    #         merged.update(
    #             torch.cat([kv.key_cache[layer_idx] for kv in kv_list], dim=0),
    #             torch.cat([kv.value_cache[layer_idx] for kv in kv_list], dim=0),
    #             layer_idx,
    #         )
    #     return merged

    def _build_decode_attention_mask(
        self,
        seq_ids: List[int],
        sequences: List[Sequence],
    ) -> torch.Tensor:
        """Full-length attention mask (cached tokens + new token) for decode."""
        masks = []
        for sid in seq_ids:
            request_id = sequences[sid].request_id
            # first [0] is layer 0, second [0] is they key tensor, shape[-2] is the sequence length dim,. 
            # this tells how many tokens are already in the cache for this sequence. 
            # mask needs to cover all those positions plus the one new token we're about to feed, hence cahced_len + 1. 
            # different from prefill mask whcih only covers the prompt tokens. 
            # cached_len = self._kv_cache[request_id].get_seq_length()
            cached_len = self._kv_cache[request_id][0][0].shape[-2]
            masks.append([1] * (cached_len + 1))

        # pad all masks to the same length (longest sequence)
        max_len = max(len(m) for m in masks)
        padded = [[0] * (max_len - len(m)) + m for m in masks]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def _forward_pass_with_kv(self, sequences, input_token_ids) -> List[int]:
        """
        Phase 4: use cached KV states for sequences past prefill.
        
        For each sequence:
          - if prefill_done=False: run full prompt, store past_key_values
          - if prefill_done=True: run single token with stored KV
        
        Challenge: HF past_key_values is batched. When the batch
        composition changes (sequences finish, new ones join), you
        must splice the KV tensors to match the new batch ordering.
        See _splice_past_key_values() below.
        """
        input_ids, attention_mask = self._pad_sequences(input_token_ids)
        

        # Run prefill for the new sequences first
        new_seq_ids = [i for i, seq in enumerate(sequences) if not seq.prefill_done]
        remaining_seq_ids = [i for i, seq in enumerate(sequences) if seq.prefill_done]

        # print(f"batch: {[(seq.request_id, seq.prompt_tokens[:5]) for seq in sequences]}")
        # print(f"new_seq_ids: {new_seq_ids}")
        # print(f"remaining_seq_ids: {remaining_seq_ids}")

        if new_seq_ids:
            input_ids_new = input_ids[new_seq_ids, :]
            attention_mask_new = attention_mask[new_seq_ids, :]
            with torch.no_grad():
                outputs_new = self.model(
                    input_ids=input_ids_new,
                    attention_mask=attention_mask_new,
                    use_cache=True,
                )
                # batched kv state for all the new sequences. 
                # (num_layers, 2, len(new_seq_ids), num_heads, seq_len, head_dim)
                past_key_values_new = outputs_new.past_key_values

            # split the batched kv state into per-sequence kv states. 
            for micro_idx, seq_id in enumerate(new_seq_ids):
                request_id = sequences[seq_id].request_id
                # print("HERE")
                # print(type(past_key_values_new))
                # print(dir(past_key_values_new))
                self._kv_cache[request_id] = self._extract_sequence_kv(
                    past_key_values_new, micro_idx
                )

        # now run decode pass the remaining sequences
        if remaining_seq_ids:
            input_ids_remaining = input_ids[remaining_seq_ids, :]
            attention_mask_decode = self._build_decode_attention_mask(
                remaining_seq_ids, sequences
            )
            with torch.no_grad():
                # input_ids_remaining[:, -1:] gets only last token
                # attention_mask_decode, full mask. even though input is 1 token, 
                # model needs to know whole mask so attention works. 

                outputs_remaining = self.model(
                    input_ids=input_ids_remaining[:, -1:],
                    attention_mask=attention_mask_decode,
                    use_cache=True,
                    # turn individual kv states into a batched kv state in order of remaining seq ids. 
                    past_key_values=self._splice_past_key_values(
                        remaining_seq_ids, sequences, self._kv_cache
                    ),
                )
                past_key_values_remaining = outputs_remaining.past_key_values

            for micro_idx, seq_id in enumerate(remaining_seq_ids):
                request_id = sequences[seq_id].request_id
                self._kv_cache[request_id] = self._extract_sequence_kv(
                    past_key_values_remaining, micro_idx 
                )

        # return the next tokens for all sequences
        next_tokens = []
        for i, seq in enumerate(sequences):

            # [:, -1, :] takes the last token position's logit. 
            # for prefill, thats the last prompt token, for decode, 
            # its the last single input token. either way, its the position
            # that the model is predicting from. 
            if i in new_seq_ids:
                # last position logits
                last_logit = outputs_new.logits[new_seq_ids.index(i), -1, :]
                token = sample_token(last_logit, seq.temperature, seq.top_p)
                # print(f"seq {seq.request_id} -> token {token} ({self.tokenizer.decode([token])})")

                next_tokens.append(token)
            elif i in remaining_seq_ids:
                last_logit = outputs_remaining.logits[remaining_seq_ids.index(i), -1, :]
                token = sample_token(last_logit, seq.temperature, seq.top_p)
                # print(f"seq {seq.request_id} -> token {token} ({self.tokenizer.decode([token])})")

                next_tokens.append(token)
        return next_tokens

    def _splice_past_key_values(
        self,
        seq_ids: List[int],
        sequences: List[Sequence],
        kv_cache: Dict[str, tuple],
    ) -> tuple:
        """
        Gather per-sequence KV caches and merge into one batched past_key_values
        tuple, ordered to match seq_ids (current scheduler batch order).
        """
        kv_list = [kv_cache[sequences[sid].request_id] for sid in seq_ids]
        return self._merge_sequence_kv(kv_list)


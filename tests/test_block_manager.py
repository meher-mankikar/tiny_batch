"""
tests/test_block_manager.py

Unit tests for the BlockAllocator.

These test block bookkeeping in isolation — no model, no tensors.
A tiny block_size (4) keeps the "block exactly full" cases easy to write.
Run with: pytest tests/test_block_manager.py -v
"""

import pytest

from src.engine.block_allocator import BlockAllocator

BLOCK_SIZE = 4


def make_allocator(num_blocks: int = 8) -> BlockAllocator:
    return BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)


class TestPrefillAllocation:
    @pytest.mark.parametrize(
        "prompt_length, expected_blocks",
        [(1, 1), (3, 1), (4, 1), (5, 2), (8, 2), (9, 3)],
    )
    def test_blocks_needed_rounds_up(self, prompt_length, expected_blocks):
        alloc = make_allocator()
        alloc.allocate_prompt_for_prefill("r1", prompt_length)
        assert len(alloc.get_block_table("r1")) == expected_blocks

    def test_free_count_drops_by_blocks_used(self):
        alloc = make_allocator(num_blocks=8)
        alloc.allocate_prompt_for_prefill("r1", 9)  # 3 blocks
        assert alloc.check_capacity() == 5

    def test_seq_length_matches_prompt(self):
        alloc = make_allocator()
        alloc.allocate_prompt_for_prefill("r1", 6)
        assert alloc.seq_lengths["r1"] == 6

    def test_not_enough_blocks_raises(self):
        alloc = make_allocator(num_blocks=2)
        with pytest.raises(RuntimeError):
            alloc.allocate_prompt_for_prefill("r1", 9)  # needs 3

    def test_failed_prefill_allocates_nothing(self):
        """No partial allocation: a failed prefill leaves the pool untouched."""
        alloc = make_allocator(num_blocks=3)
        alloc.allocate_prompt_for_prefill("r1", 5)  # 2 blocks, 1 left
        with pytest.raises(RuntimeError):
            alloc.allocate_prompt_for_prefill("r2", 9)  # needs 3

        assert alloc.check_capacity() == 1
        assert "r2" not in alloc.block_tables
        assert alloc.get_block_table("r1") == [0, 1]


class TestDecodeAllocation:
    def test_no_new_block_while_last_block_has_room(self):
        alloc = make_allocator()
        alloc.allocate_prompt_for_prefill("r1", 5)  # block 1 holds 1 of 4
        for _ in range(3):  # fills block 1 to 4 of 4
            alloc.get_next_block("r1")
        assert len(alloc.get_block_table("r1")) == 2
        assert alloc.seq_lengths["r1"] == 8

    def test_new_block_exactly_when_last_block_full(self):
        alloc = make_allocator()
        alloc.allocate_prompt_for_prefill("r1", 4)  # exactly fills block 0
        new_block = alloc.get_next_block("r1")
        table = alloc.get_block_table("r1")
        assert len(table) == 2
        assert new_block == table[-1]

    def test_returns_current_block_when_not_full(self):
        alloc = make_allocator()
        alloc.allocate_prompt_for_prefill("r1", 2)
        assert alloc.get_next_block("r1") == alloc.get_block_table("r1")[-1]

    def test_decode_out_of_blocks_raises(self):
        alloc = make_allocator(num_blocks=1)
        alloc.allocate_prompt_for_prefill("r1", 4)  # pool now empty, block full
        with pytest.raises(RuntimeError):
            alloc.get_next_block("r1")
        # failed decode must not change state
        assert alloc.seq_lengths["r1"] == 4
        assert alloc.get_block_table("r1") == [0]


class TestFree:
    def test_free_returns_all_blocks(self):
        alloc = make_allocator(num_blocks=8)
        alloc.allocate_prompt_for_prefill("r1", 9)
        alloc.free_request("r1")
        assert alloc.check_capacity() == 8
        assert "r1" not in alloc.block_tables

    def test_double_free_is_safe(self):
        alloc = make_allocator(num_blocks=8)
        alloc.allocate_prompt_for_prefill("r1", 5)
        alloc.free_request("r1")
        alloc.free_request("r1")
        assert alloc.check_capacity() == 8

    def test_free_unknown_request_is_safe(self):
        alloc = make_allocator()
        alloc.free_request("never-allocated")
        assert alloc.check_capacity() == 8

    def test_reused_request_id_starts_fresh(self):
        alloc = make_allocator()
        alloc.allocate_prompt_for_prefill("r1", 9)
        alloc.free_request("r1")
        alloc.allocate_prompt_for_prefill("r1", 1)
        assert alloc.seq_lengths["r1"] == 1
        assert len(alloc.get_block_table("r1")) == 1

    def test_freed_blocks_are_reused(self):
        alloc = make_allocator(num_blocks=2)
        alloc.allocate_prompt_for_prefill("r1", 8)  # takes the whole pool
        alloc.free_request("r1")
        alloc.allocate_prompt_for_prefill("r2", 8)  # would fail if not freed
        assert sorted(alloc.get_block_table("r2")) == [0, 1]


class TestIsolation:
    def test_sequences_never_share_blocks(self):
        alloc = make_allocator(num_blocks=8)
        alloc.allocate_prompt_for_prefill("r1", 5)
        alloc.allocate_prompt_for_prefill("r2", 5)
        for _ in range(4):  # interleave decode so both grow into new blocks
            alloc.get_next_block("r1")
            alloc.get_next_block("r2")

        r1_blocks = set(alloc.get_block_table("r1"))
        r2_blocks = set(alloc.get_block_table("r2"))
        assert r1_blocks.isdisjoint(r2_blocks)
        assert alloc.check_capacity() == 8 - len(r1_blocks) - len(r2_blocks)

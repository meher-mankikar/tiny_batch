from collections import defaultdict
from typing import List


class BlockAllocator:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # 1 if the block is free and 0 if it is not.
        self.free_blocks : List[int] = [1 for _ in range(num_blocks)]

        # Maps req_id -> List of block ids. 
        # Each block holds K and V for B tokens (B = block size)
        self.block_tables : dict[str, List[int]] = defaultdict(list)

        # keep track of how long each sequence is so far. 
        # req_id -> length of sequence. tells us when
        # we need to allocate a new block for a sequence.
        self.seq_lengths : dict[str, int] = defaultdict(int)


    """
    How many blocks does a new prompt need? 
    e.g. block_size=16. prompt size=5 => 1 block
    block_size=16. prompt size=16 => 1 block
    block_size=16. prompt size=17 => 2 blocks?

    When does decoding a sequence need a new block?
    When the last block is full. need to know how many
    tokens have used the last block basically. 

    What happens when there aren't enough free blocks? 
    For now, we can raise an error or return "can't allocate"

    Freeing:
    When a sequence finishes or is aborted, all its blocks are freed. 
    Need to make sure can't double free. 

    Plan for later: prefix caching. 
    Same blocks will be shared by several sequences. 
    Then, freeing can't be deleting the blocks because the other seq may
    still need them. 
    """
    def get_new_block(self):
        """
        Return a new block from the free table or -1 if one doesn't exist
        """
        for idx in range(len(self.free_blocks)):
            if self.free_blocks[idx] == 1:
                # it's free, return it
                return idx
        return -1

    def get_next_block(self, req_id):
        """
        Make room for one more token. If the last block is full, 
        get a new one. otherwise, do nothing. return the new block
        (or the old one).
        Used by: engine on decode. 
        """
        # If last block is full
        if self.seq_lengths[req_id] % self.block_size == 0:
            # get a new block from the free table. 
            new_block_idx = self.get_new_block()
            if new_block_idx == -1: 
                raise RuntimeError("Cannot allocate new block")
            # remove from free
            self.free_blocks[new_block_idx] = 0
            # add to block table
            self.block_tables[req_id].append(new_block_idx)
            self.seq_lengths[req_id] += 1
            return new_block_idx

        else:
            # return the block we have been using
            self.seq_lengths[req_id] += 1
            return self.block_tables[req_id][-1]

    def allocate_prompt_for_prefill(self, req_id, prompt_length):
        """
        Given a request and the prompt length, take free blocks from the
        list and record them as for this req in the block table.
        Used by: engine at prefill.
        """
        # first, check if we have enough blocks free for the whole prompt length
        # equivalent to ceil(prompt_length / block_size.)
        num_blocks_needed = (prompt_length + self.block_size - 1) // self.block_size
        if self.check_capacity() < num_blocks_needed:
            raise RuntimeError("Cannot allocate enough blocks for this prompt")
        # Otherwise, allocate the blocks one by one. 
        for tok in range(prompt_length):
            self.get_next_block(req_id)

    def free_request(self, req_id):
        """
        Return all the blocks to the pool for a req_id
        and remove its block table. Make sure its safe
        if it gets called twice.
        """
        if req_id in self.block_tables:
            req_blocks = self.block_tables[req_id]
            for block in req_blocks:
                self.free_blocks[block] = 1
            del self.block_tables[req_id]
            del self.seq_lengths[req_id]

    def check_capacity(self):
        """
        Can I allocate N more blocks right now? 
        Or just checks how many are frree. 
        Needed by scheduler in _admit_waiting
        """
        return sum(self.free_blocks)

    def get_block_table(self, req_id):
        """
        Needed by engine in milestone 2
        """
        return self.block_tables[req_id]

    
"""
engine/sampling.py

Token sampling strategies. Keeping this separate makes it easy to
extend without touching the engine.
"""

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,   # shape: [vocab_size]
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> int:
    """
    Sample a single next token from logits.
    High temp (>1) flattens the distribution (more random), low temp (<1) concentrates it (more deterministic).
    Top p optionally masks out low probability tokens before sampling. 
    Softmax converts to prob distribution
    
    Supports:
      - Greedy (temperature=0)
      - Temperature scaling
      - Nucleus (top-p) sampling
    """
    if temperature == 0.0:
        return int(logits.argmax().item())

    # temperature scaling
    logits = logits / temperature

    # top-p (nucleus) sampling
    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Zero out logits outside the top-p nucleus.
    Returns modified logits (everything outside nucleus set to -inf).

    Main idea: sort by probability. walk down the list accumulating probability mass until 
    you cover top_p of the mass. 
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # cumulative sum of the probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # remove tokens with cumulative prob above the threshold
    # shift right by 1 so we include the token that crosses the threshold
    # This checks if top_p was hit before the current indices mass is added. 
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # scatter back to original indexing
    logits = torch.full_like(logits, float('-inf'))
    logits.scatter_(0, sorted_indices, sorted_logits)
    return logits

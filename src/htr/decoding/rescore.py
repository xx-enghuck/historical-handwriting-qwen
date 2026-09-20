"""Differentiable teacher-forced sequence scoring, independent of generate() scores."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from htr.models.qwen import QwenEncoder, to_device


def sequence_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, length_normalization: float = 0.0
) -> torch.Tensor:
    """Sum next-token log P over assistant positions, including a present EOS.

    The causal shift is essential: logits at t-1 predict labels at t. Prompt and
    padding labels are -100. Only active response logits are promoted to fp32.
    """
    if length_normalization < 0:
        raise ValueError("length_normalization must be nonnegative")
    shifted_labels = labels[:, 1:]
    active = shifted_labels != -100
    counts = active.sum(-1)
    if (counts == 0).any():
        raise ValueError("Cannot score a sequence with zero supervised tokens")
    selected_logits = logits[:, :-1][active].float()
    token_logprobs = -F.cross_entropy(selected_logits, shifted_labels[active], reduction="none")
    batch_indices = (
        torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(active)[active]
    )
    sums = torch.zeros(labels.shape[0], dtype=token_logprobs.dtype, device=logits.device)
    sums = sums.scatter_add(0, batch_indices, token_logprobs)
    return sums / counts.float().pow(length_normalization)


def rescore_candidates(
    model,
    encoder: QwenEncoder,
    image,
    candidates: list[dict],
    length_normalization: float = 0.0,
    checkpoint_forward: bool = False,
) -> torch.Tensor:
    """Re-forward each exact generated token sequence; never use detached beam scores.

    An outer non-reentrant checkpoint optionally recomputes each candidate on
    backward, including when deterministic scoring puts the model in eval mode.
    """
    scores = []
    for candidate in candidates:
        batch = to_device(encoder.batch([image], candidate_ids=[candidate["token_ids"]]), model)

        def score(batch=batch):
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            logits = model(**inputs, use_cache=False).logits
            return sequence_logprobs(logits, batch["labels"], length_normalization)[0]

        scores.append(checkpoint(score, use_reentrant=False) if checkpoint_forward else score())
    return torch.stack(scores)

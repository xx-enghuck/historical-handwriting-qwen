"""N-best minimum expected WER with detached risks and mean-risk centering."""

import torch


def nbest_probabilities(sequence_scores: torch.Tensor) -> torch.Tensor:
    if sequence_scores.ndim != 1 or sequence_scores.numel() < 2:
        raise ValueError("Expected at least two scores in one N-best list")
    if not torch.isfinite(sequence_scores).all():
        raise ValueError("Nonfinite sequence scores")
    return torch.softmax(sequence_scores.float(), dim=0)


def mwer_loss(
    sequence_scores: torch.Tensor, risks: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """Risks are constants; gradients flow exclusively through normalized sequence scores."""
    if sequence_scores.shape != risks.shape or not torch.isfinite(risks).all() or (risks < 0).any():
        raise ValueError("Expected finite nonnegative risks with the same shape as scores")
    risks = risks.detach().to(sequence_scores.device, dtype=torch.float32)
    centered = risks - risks.mean() if center else risks
    return (nbest_probabilities(sequence_scores) * centered).sum()

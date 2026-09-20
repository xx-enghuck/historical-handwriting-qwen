"""Alignment protocol and transcript-constrained CTC Viterbi alignment."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from htr.evaluation.metrics import sample_metrics


@dataclass(frozen=True)
class Segment:
    text: str
    left: int
    right: int
    quality: float


class AlignmentError(ValueError):
    """A line cannot produce a valid, sufficiently accurate alignment."""


class Aligner(Protocol):
    def align(self, image: Image.Image, transcription: str) -> list[Segment]: ...


def forced_ctc_path(log_probs: torch.Tensor, targets: list[int], blank: int = 0) -> list[int]:
    """Return the optimal extended-target state at each frame, including CTC blanks.

    Repeated characters require an intervening blank. Invalid alignments fail;
    no equal-width fallback is used.
    """
    scores = log_probs.detach().float().cpu().numpy()
    if scores.ndim != 2 or not targets or blank in targets:
        raise AlignmentError("Expected T x V log probabilities and nonblank target IDs")
    frames, vocab = scores.shape
    if max(targets) >= vocab or min(targets) < 0:
        raise AlignmentError("Target outside CTC vocabulary")
    minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
    if frames < minimum:
        raise AlignmentError(f"CTC has {frames} frames but needs at least {minimum}")
    extended = np.asarray([blank] + [token for t in targets for token in (t, blank)])
    states = len(extended)
    previous = np.full(states, -np.inf)
    previous[:2] = scores[0, extended[:2]]
    back = np.zeros((frames, states), dtype=np.int64)
    can_skip = np.zeros(states, dtype=bool)
    can_skip[2:] = (extended[2:] != blank) & (extended[2:] != extended[:-2])
    indices = np.arange(states)
    for frame in range(1, frames):
        stay = previous
        step = np.r_[-np.inf, previous[:-1]]
        skip = np.r_[-np.inf, -np.inf, previous[:-2]]
        skip[~can_skip] = -np.inf
        choices = np.stack((stay, step, skip))
        transition = choices.argmax(axis=0)
        back[frame] = indices - transition
        previous = choices[transition, indices] + scores[frame, extended]
    state = states - 1 if previous[-1] >= previous[-2] else states - 2
    if not np.isfinite(previous[state]):
        raise AlignmentError("No finite CTC path")
    path = [state]
    for frame in range(frames - 1, 0, -1):
        state = int(back[frame, state])
        path.append(state)
    return list(reversed(path))


def segments_from_ctc(
    log_probs: torch.Tensor, text: str, ids: list[int], width: int
) -> list[Segment]:
    path = forced_ctc_path(log_probs, ids)
    spans = [[i for i, state in enumerate(path) if state == 2 * k + 1] for k in range(len(ids))]
    if any(not span for span in spans):
        raise AlignmentError("A character has no aligned frames")
    # Boundaries use actual CTC state transitions, distributing inter-character blanks.
    boundaries = (
        [0]
        + [round(((a[-1] + b[0] + 1) / 2) * width / len(path)) for a, b in zip(spans, spans[1:])]
        + [width]
    )
    result = []
    for i, (char, token, span) in enumerate(zip(text, ids, spans, strict=True)):
        quality = float(log_probs[span, token].exp().mean())
        if boundaries[i + 1] <= boundaries[i]:
            raise AlignmentError("CTC segment has zero pixel width")
        result.append(Segment(char, boundaries[i], boundaries[i + 1], quality))
    return result


class CTCAligner:
    def __init__(self, model, alphabet: list[str], height: int, max_cer: float):
        self.model, self.alphabet, self.height, self.max_cer = model, alphabet, height, max_cer
        self.ids = {char: i + 1 for i, char in enumerate(alphabet)}

    @torch.no_grad()
    def align(self, image: Image.Image, transcription: str) -> list[Segment]:
        from htr.data.stackmix.ctc import line_tensor

        if not transcription or any(c not in self.ids for c in transcription):
            raise AlignmentError("Empty or out-of-vocabulary transcription")
        self.model.eval()
        tensor = (
            line_tensor(image, self.height).unsqueeze(0).to(next(self.model.parameters()).device)
        )
        log_probs = self.model(tensor)[0].log_softmax(-1).cpu()
        greedy, previous = [], -1
        for token in log_probs.argmax(-1).tolist():
            if token and token != previous:
                greedy.append(self.alphabet[token - 1])
            previous = token
        if sample_metrics(transcription, "".join(greedy))["cer"] > self.max_cer:
            raise AlignmentError("CTC greedy CER exceeds configured alignment quality threshold")
        return segments_from_ctc(
            log_probs, transcription, [self.ids[c] for c in transcription], image.width
        )

"""Character and word error rates for validation, CTC alignment and MWER."""

import unicodedata
from collections.abc import Sequence

from htr.config import NormalizeConfig


def normalize(text: str, cfg: NormalizeConfig | None = None) -> str:
    cfg = cfg or NormalizeConfig()
    if cfg.unicode_form:
        text = unicodedata.normalize(cfg.unicode_form, text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.collapse_whitespace:
        text = " ".join(text.split())
    if cfg.strip:
        text = text.strip()
    return text


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance in O(nm) time and O(m) memory."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def sample_metrics(reference: str, hypothesis: str, cfg: NormalizeConfig | None = None) -> dict:
    ref, hyp = normalize(reference, cfg), normalize(hypothesis, cfg)
    chars, words = edit_distance(ref, hyp), edit_distance(ref.split(), hyp.split())
    cer = chars / max(1, len(ref))
    wer = words / max(1, len(ref.split()))
    return {
        "cer": cer,
        "wer": wer,
        "score": 0.5 * (cer + wer),
        "char_edits": chars,
        "word_edits": words,
        "reference_chars": len(ref),
        "reference_words": len(ref.split()),
    }


def corpus_metrics(
    references: list[str], hypotheses: list[str], cfg: NormalizeConfig | None = None
) -> dict:
    """Micro-average edit counts; an empty denominator is clamped to one."""
    if not references or len(references) != len(hypotheses):
        raise ValueError("Expected equally sized, nonempty reference/hypothesis lists")
    per_sample = [sample_metrics(r, h, cfg) for r, h in zip(references, hypotheses, strict=True)]
    chars = sum(s["char_edits"] for s in per_sample)
    words = sum(s["word_edits"] for s in per_sample)
    cer = chars / max(1, sum(s["reference_chars"] for s in per_sample))
    wer = words / max(1, sum(s["reference_words"] for s in per_sample))
    return {"cer": cer, "wer": wer, "score": 0.5 * (cer + wer), "per_sample": per_sample}

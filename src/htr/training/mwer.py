"""Sequence-level minimum-risk fine-tuning with optional reference CE."""

from dataclasses import asdict

import torch

from htr.config import Config
from htr.data.dataset import samples_hash
from htr.decoding.rescore import rescore_candidates
from htr.evaluation.metrics import sample_metrics
from htr.training.checkpoint import bundle_hash
from htr.training.losses import mwer_loss
from htr.training.sft import sft_loss
from htr.utils.io import file_hash, read_json, read_jsonl


def load_nbest(cfg: Config, train_samples: list) -> dict:
    path = cfg.path(cfg.mwer.nbest_path)
    meta = read_json(path.with_suffix(".metadata.json"))
    if (
        meta["train_split_hash"] != samples_hash(train_samples)
        or meta["role"] != "original_train_only"
    ):
        raise ValueError("N-best cache must use exactly the original training split")
    if not cfg.train.checkpoint or meta["checkpoint_hash"] != bundle_hash(
        cfg.path(cfg.train.checkpoint)
    ):
        raise ValueError("N-best source SFT checkpoint differs from train.checkpoint")
    if meta["nbest_hash"] != file_hash(path):
        raise ValueError("N-best cache contents changed")
    for key in ("num_beams", "num_return_sequences"):
        if meta["decode"][key] != getattr(cfg.mwer, key):
            raise ValueError(f"N-best {key} differs from MWER config")
    for key in ("prompt", "min_pixels", "max_pixels"):
        if meta["model"][key] != getattr(cfg.model, key):
            raise ValueError("N-best prompt or image processor settings differ")
    if {k: v for k, v in meta["crop"].items() if k != "debug_dir"} != {
        k: v for k, v in asdict(cfg.crop).items() if k != "debug_dir"
    }:
        raise ValueError("N-best crop settings differ")
    rows = read_jsonl(path)
    result = {r["image_id"]: r for r in rows}
    if len(result) != len(rows) or set(result) != {s.image_id for s in train_samples}:
        raise ValueError("N-best IDs do not match training samples exactly once")
    for sample in train_samples:
        row = result[sample.image_id]
        if row["ground_truth"] != sample.transcription:
            raise ValueError("N-best reference differs from train manifest")
        candidates = row["candidates"]
        keys = {tuple(c["token_ids"]) for c in candidates}
        if (
            len(keys) != len(candidates)
            or len(keys) < 2
            or any(not c["token_ids"] for c in candidates)
        ):
            raise ValueError("N-best must contain at least two unique, nonempty token sequences")
    return result


def verify_candidate_tokens(nbest: dict, tokenizer) -> None:
    """Ensure WER text and teacher-forced token sequence describe the same candidate."""
    for row in nbest.values():
        for candidate in row["candidates"]:
            ids = candidate["token_ids"]
            if any(not isinstance(token, int) or not 0 <= token < len(tokenizer) for token in ids):
                raise ValueError("Candidate token ID outside saved processor vocabulary")
            decoded = tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if decoded != candidate["text"]:
                raise ValueError("Candidate text does not match saved generated token IDs")


def mwer_sample_loss(
    model, encoder, image, sample, row: dict, cfg: Config
) -> tuple[torch.Tensor, dict]:
    scores = rescore_candidates(
        model,
        encoder,
        image,
        row["candidates"],
        cfg.mwer.length_normalization,
        cfg.train.gradient_checkpointing,
    )
    # Python edit distance creates no autograd graph. Explicit detach is defensive.
    risks = torch.tensor(
        [
            sample_metrics(sample.transcription, candidate["text"], cfg.normalization)["wer"]
            for candidate in row["candidates"]
        ],
        device=scores.device,
    ).detach()
    minimum_risk = mwer_loss(scores, risks, cfg.mwer.center_risks)
    ce = (
        sft_loss(model, encoder, [image], [sample.transcription])
        if cfg.mwer.lambda_ce
        else scores.new_zeros(())
    )
    return minimum_risk + cfg.mwer.lambda_ce * ce, {
        "mwer_loss": float(minimum_risk.detach()),
        "ce_loss": float(ce.detach()),
        "risk_mean": float(risks.mean()),
        "risk_spread": float(risks.max() - risks.min()),
    }

"""Single-device research loop with validation-only model selection and exact epoch resume."""

import math
import random

import torch

from htr.config import Config
from htr.data.crop import crop_line
from htr.data.dataset import load_image, samples_hash
from htr.data.split import load_splits
from htr.evaluation.metrics import corpus_metrics
from htr.models.lora import add_lora, parameter_report, set_trainable, train_mode
from htr.models.qwen import QwenEncoder, load_qwen, to_device
from htr.training.checkpoint import (
    load_bundle,
    restore_training_state,
    save_bundle,
    save_training_state,
)
from htr.training.sft import sft_loss
from htr.utils.io import read_json
from htr.utils.logging import RunLogger
from htr.utils.seed import seed_everything


def prepare_image(sample, cfg: Config):
    debug = cfg.path(cfg.crop.debug_dir) / f"{sample.image_id}.png" if cfg.crop.debug_dir else None
    return crop_line(load_image(sample, cfg.path(cfg.data.image_root)), cfg.crop, debug)


@torch.no_grad()
def validate(model, encoder: QwenEncoder, samples: list, cfg: Config) -> dict:
    model.eval()
    predictions, references = [], []
    nll, tokens = 0.0, 0
    for sample in samples:
        image = prepare_image(sample, cfg)
        batch = to_device(encoder.batch([image], texts=[sample.transcription]), model)
        count = (batch["labels"][:, 1:] != -100).sum().item()
        loss = model(**batch, use_cache=False).loss
        nll += float(loss) * count
        tokens += count
        from htr.decoding.greedy import greedy_decode

        predictions.append(
            greedy_decode(model, encoder, image, cfg.decode)["candidates"][0]["text"]
        )
        references.append(sample.transcription)
    metrics = corpus_metrics(references, predictions, cfg.normalization)
    return {k: v for k, v in metrics.items() if k != "per_sample"} | {
        "validation_loss": nll / tokens
    }


def _resume_contract(cfg: Config) -> dict:
    raw = cfg.to_dict()
    # Epoch count may be extended; all scientific settings must stay fixed.
    for key in ("epochs", "resume"):
        raw["train"].pop(key)
    return raw


def train(cfg: Config, stage: str = "sft", components: tuple | None = None) -> dict:
    """Train SFT or MWER. components is an explicit offline integration-test hook."""
    seed_everything(cfg.seed, cfg.deterministic)
    splits = load_splits(cfg)
    train_samples, validation = splits["train"], splits["validation"]
    split_hash = samples_hash(train_samples)
    directory = cfg.path(cfg.train.output_dir)
    if (directory / "metadata.json").exists() and not cfg.train.resume:
        raise ValueError("Run directory already exists; choose a new output_dir or resume")
    if cfg.train.resume:
        source = cfg.path(cfg.train.resume)
        model, processor, bundle = load_bundle(cfg, source, stage)
        old = bundle["training_config"]
        for key in ("epochs", "resume"):
            old["train"].pop(key)
        if old != _resume_contract(cfg) or bundle["stage"] != stage:
            raise ValueError("Resume settings differ; only epochs and resume may change")
    elif components:
        model, processor = components
        model = add_lora(model, cfg.lora)
        set_trainable(model, stage)
        bundle = None
    elif stage == "mwer":
        if not cfg.train.checkpoint:
            raise ValueError("MWER requires train.checkpoint pointing to an SFT bundle")
        model, processor, bundle = load_bundle(cfg, cfg.path(cfg.train.checkpoint), stage)
    else:
        model, processor = load_qwen(cfg.model)
        model = add_lora(model, cfg.lora)
        bundle = None
    if bundle and bundle["train_split_hash"] != split_hash:
        raise ValueError("Checkpoint train split hash differs")
    if stage == "mwer":
        from htr.training.mwer import load_nbest, mwer_sample_loss

        nbest = load_nbest(cfg, train_samples)
    elif stage != "sft":
        raise ValueError(f"Unknown stage: {stage}")
    if cfg.stackmix.enabled and stage == "sft":
        from htr.data.stackmix.augment import load_synthetic

        train_samples = train_samples + load_synthetic(cfg, train_samples)
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    encoder = QwenEncoder(processor, cfg.model)
    if stage == "mwer":
        from htr.training.mwer import verify_candidate_tokens

        verify_candidate_tokens(nbest, processor.tokenizer)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    epoch_start, best, stale = 0, math.inf, 0
    if cfg.train.resume:
        state = restore_training_state(cfg.path(cfg.train.resume) / "training_state.pt", optimizer)
        epoch_start, best, stale = state["epoch"], state["best"], state["stale"]
        if epoch_start >= cfg.train.epochs:
            raise ValueError("Resume epoch is already >= train.epochs")
    logger = RunLogger(
        cfg, stage, {k: samples_hash(v) for k, v in splits.items()}, parameter_report(model)
    )
    for epoch in range(epoch_start, cfg.train.epochs):
        train_mode(model, stage)
        order = list(range(len(train_samples)))
        random.Random(cfg.seed + epoch).shuffle(order)
        batches = [
            order[i : i + cfg.train.batch_size] for i in range(0, len(order), cfg.train.batch_size)
        ]
        total, count = 0.0, 0
        diagnostics: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(batches), cfg.train.gradient_accumulation):
            window = batches[start : start + cfg.train.gradient_accumulation]
            # Exact token-average CE across accumulation windows; MWER averages lists.
            window_weight = (
                sum(
                    len(encoder.response_ids(train_samples[i].transcription))
                    for indices in window
                    for i in indices
                )
                if stage == "sft"
                else sum(len(indices) for indices in window)
            )
            for indices in window:
                rows = [train_samples[i] for i in indices]
                images = [prepare_image(sample, cfg) for sample in rows]
                if stage == "sft":
                    loss = sft_loss(model, encoder, images, [r.transcription for r in rows])
                    weight = sum(len(encoder.response_ids(r.transcription)) for r in rows)
                else:
                    scored = [
                        mwer_sample_loss(model, encoder, image, row, nbest[row.image_id], cfg)
                        for image, row in zip(images, rows, strict=True)
                    ]
                    loss = torch.stack([item[0] for item in scored]).mean()
                    for _, values in scored:
                        for key, value in values.items():
                            diagnostics[key] = diagnostics.get(key, 0.0) + value
                    weight = len(rows)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; checkpoint not updated")
                (loss * (weight / window_weight)).backward()
                total += float(loss.detach()) * weight
                count += weight
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.train.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = validate(model, encoder, validation, cfg)
        metrics.update(epoch=epoch + 1, training_loss=total / count)
        metrics.update({key: value / len(train_samples) for key, value in diagnostics.items()})
        if metrics["score"] < best:
            best, stale = metrics["score"], 0
            save_bundle(model, processor, cfg, directory / "best", stage, split_hash)
            save_training_state(
                directory / "best/training_state.pt", optimizer, epoch + 1, best, stale
            )
            from htr.utils.io import write_json

            write_json(directory / "best/validation.json", metrics)
        else:
            stale += 1
        save_bundle(model, processor, cfg, directory / "last", stage, split_hash)
        from htr.utils.io import write_json

        write_json(directory / "last/validation.json", metrics)
        save_training_state(directory / "last/training_state.pt", optimizer, epoch + 1, best, stale)
        logger.event(metrics)
        if (
            cfg.train.early_stopping_patience is not None
            and stale >= cfg.train.early_stopping_patience
            and stale > 0
        ):
            break
    best_metrics = read_json(directory / "best/validation.json")
    logger.finish(
        {
            **best_metrics,
            "model_checkpoint": str(directory / "best"),
            "selection": "validation greedy combined score",
            "last_epoch": epoch + 1,
        }
    )
    return best_metrics

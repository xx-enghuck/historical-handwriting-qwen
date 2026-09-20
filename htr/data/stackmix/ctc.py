"""Small train-only auxiliary Conv-BiLSTM recognizer for weak CTC boundaries."""

import logging
import random
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image
from torch import nn

from htr.config import Config
from htr.data.dataset import samples_hash
from htr.data.split import load_splits
from htr.models.qwen import resolve_device
from htr.utils.io import write_json
from htr.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def line_tensor(image: Image.Image, height: int) -> torch.Tensor:
    width = max(4, round(image.width * height / image.height))
    resized = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(1.0 - np.asarray(resized, dtype=np.float32) / 255.0).unsqueeze(0)


class LineCTC(nn.Module):
    """Horizontal stride four; grayscale height pooling then bidirectional LSTM."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv(images).mean(2).transpose(1, 2)
        return self.head(self.rnn(features)[0])


def crop_signature(cfg: Config) -> dict:
    return {k: v for k, v in asdict(cfg.crop).items() if k != "debug_dir"}


def train_ctc(cfg: Config) -> dict:
    """Use only the frozen train manifest for vocabulary, images, labels and optimization."""
    from htr.training.engine import prepare_image

    seed_everything(cfg.seed, cfg.deterministic)
    samples = load_splits(cfg)["train"]
    alphabet = sorted(set("".join(s.transcription for s in samples)))
    ids = {c: i + 1 for i, c in enumerate(alphabet)}
    model = LineCTC(len(alphabet) + 1).to(resolve_device(cfg.model.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.stackmix.ctc_learning_rate)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    prepared = [
        (
            line_tensor(prepare_image(row, cfg), cfg.stackmix.height),
            torch.tensor([ids[c] for c in row.transcription], dtype=torch.long),
        )
        for row in samples
    ]
    history = []
    for epoch in range(cfg.stackmix.ctc_epochs):
        order = list(range(len(samples)))
        random.Random(cfg.seed + epoch).shuffle(order)
        total = 0.0
        for index in order:
            image, target = prepared[index]
            if not target.numel():
                raise ValueError("CTC alignment training requires nonempty transcriptions")
            logits = model(image.unsqueeze(0).to(next(model.parameters()).device))
            frames = logits.shape[1]
            minimum = len(target) + int((target[1:] == target[:-1]).sum())
            if frames < minimum:
                raise ValueError(
                    "Line too narrow for CTC; increase stackmix.height or inspect the label"
                )
            # CPU lengths and int64 targets select the generic CTC implementation.
            # Avoid nondeterministic cuDNN CTC when reproducibility is requested.
            with torch.backends.cudnn.flags(enabled=False):
                loss = criterion(
                    logits.log_softmax(-1).transpose(0, 1),
                    target.to(logits.device),
                    torch.tensor([frames]),
                    torch.tensor([len(target)]),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CTC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "training_loss": total / len(samples)})
        logger.info("CTC %s", history[-1])
    path = cfg.path(cfg.stackmix.ctc_checkpoint)
    if path.exists():
        raise ValueError(
            "CTC checkpoint exists; choose a new path rather than overwrite provenance"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "alphabet": alphabet,
        "train_split_hash": samples_hash(samples),
        "train_ids": [s.image_id for s in samples],
        "crop": crop_signature(cfg),
        "height": cfg.stackmix.height,
        "seed": cfg.seed,
        "history": history,
        "config": cfg.to_dict(),
    }
    torch.save({"state_dict": model.cpu().state_dict(), **meta}, path)
    write_json(path.with_suffix(".json"), meta)
    return meta


def load_aligner(cfg: Config, train_samples: list):
    from htr.data.stackmix.aligner import CTCAligner

    state = torch.load(cfg.path(cfg.stackmix.ctc_checkpoint), map_location="cpu", weights_only=True)
    if state["train_split_hash"] != samples_hash(train_samples) or set(state["train_ids"]) != {
        s.image_id for s in train_samples
    }:
        raise ValueError("CTC checkpoint was not trained on this training split")
    if state["crop"] != crop_signature(cfg) or state["height"] != cfg.stackmix.height:
        raise ValueError("CTC preprocessing differs from current config")
    model = LineCTC(len(state["alphabet"]) + 1).to(resolve_device(cfg.model.device))
    model.load_state_dict(state["state_dict"])
    return CTCAligner(model, state["alphabet"], state["height"], cfg.stackmix.max_alignment_cer)

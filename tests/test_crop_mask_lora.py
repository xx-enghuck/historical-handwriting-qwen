import pytest
import torch
from PIL import Image, ImageDraw

from htr.config import CropConfig, LoraConfig
from htr.data.crop import crop_line
from htr.models.lora import add_lora, inspect_layout, set_trainable
from htr.models.qwen import response_labels


def test_crop(tmp_path):
    image = Image.new("RGB", (100, 40), "white")
    ImageDraw.Draw(image).rectangle((20, 10, 79, 29), fill="black")
    text = "Old spelling!"
    sample = {"image": image, "transcription": text}
    sample["image"] = crop_line(image, CropConfig(padding=3), tmp_path / "debug.png")
    assert sample["transcription"] == text
    assert sample["image"].size == (66, 26)
    assert crop_line(image, CropConfig(enabled=False)).size == image.size
    blank = Image.new("RGB", (12, 9), "white")
    assert crop_line(blank, CropConfig()).size == (12, 9)


def test_response_mask():
    # pad == EOS: the final active EOS must still be supervised.
    ids = torch.tensor([[0, 8, 3, 9, 11, 0], [8, 3, 9, 12, 13, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    labels = response_labels(ids, mask, [3, 3])
    assert labels.tolist() == [[-100, -100, -100, -100, 11, 0], [-100, -100, -100, 12, 13, 0]]


def test_lora_freeze_policy(tiny_components):
    model, _ = tiny_components
    layout = inspect_layout(model, LoraConfig())
    assert len(layout.targets) == 7
    with pytest.raises(ValueError, match="Missing"):
        inspect_layout(model, LoraConfig(targets=["nonexistent"]))
    model = add_lora(model, LoraConfig(r=2, alpha=4))
    for stage in ("sft", "mwer"):
        set_trainable(model, stage)
        for name, parameter in model.named_parameters():
            expected = ".lora_" in name or (stage == "sft" and ".visual.merger." in name)
            assert parameter.requires_grad == expected, name
    batch = {
        "input_ids": torch.tensor([[8, 9, 10, 11]]),
        "labels": torch.tensor([[-100, -100, 10, 11]]),
    }
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_B" in n
    )

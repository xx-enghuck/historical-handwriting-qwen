import torch
from PIL import Image

from htr.config import Config, LoraConfig, ModelConfig
from htr.data.dummy import generate_dummy
from htr.data.split import prepare_splits
from htr.models.lora import add_lora, merger_module
from htr.models.qwen import QwenEncoder
from htr.testing import tiny_components
from htr.training.engine import train


def test_image_sft_checkpoint_resume(tmp_path):
    torch.set_num_threads(1)
    model, processor = tiny_components()
    base = tmp_path / "tiny_base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    cfg = Config(project_root=str(tmp_path))
    cfg.model = ModelConfig(
        name=str(base),
        device="cpu",
        dtype="float32",
        min_pixels=64,
        max_pixels=256,
        prompt="Transcribe.",
    )
    cfg.lora = LoraConfig(r=2, alpha=4, dropout=0)
    cfg.data.image_root = "dummy"
    cfg.data.source_csv = "dummy/metadata.csv"
    cfg.train.epochs = 1
    cfg.train.gradient_accumulation = 2
    cfg.train.batch_size = 2
    cfg.decode.max_new_tokens = 2
    generate_dummy(tmp_path / "dummy", count=8)
    prepare_splits(cfg)
    result = train(cfg, components=(model, processor))
    assert result["validation_loss"] > 0
    assert (tmp_path / "runs/sft/best/merger.safetensors").is_file()
    assert not (tmp_path / "runs/sft/best/model.safetensors").exists()
    cfg.train.resume = "runs/sft/last"
    cfg.train.epochs = 2
    train(cfg)
    assert len((tmp_path / "runs/sft/history.jsonl").read_text().splitlines()) == 2


def test_multimodal_mask_and_merger_gradient():
    model, processor = tiny_components()
    model = add_lora(model, LoraConfig(r=2))
    encoder = QwenEncoder(processor, ModelConfig(prompt="Transcribe."))
    images = [Image.new("RGB", (32, 16), "white"), Image.new("RGB", (24, 16), "gray")]
    batch = encoder.batch(images, texts=["abc", "a b"])
    expected = [encoder.response_ids("abc"), encoder.response_ids("a b")]
    for labels, ids in zip(batch["labels"], expected, strict=True):
        assert labels[labels != -100].tolist() == ids
    model(**batch, use_cache=False).loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in merger_module(model).parameters()
    )

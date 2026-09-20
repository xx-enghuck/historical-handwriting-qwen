"""Exercise the full training and evaluation pipeline on generated data."""

import torch

from htr.config import Config, LoraConfig, ModelConfig
from htr.data.dummy import generate_dummy
from htr.data.split import prepare_splits
from htr.data.stackmix.augment import synthesize
from htr.data.stackmix.ctc import train_ctc
from htr.data.stackmix.segment_bank import build_bank
from htr.decoding.infer import infer
from htr.decoding.nbest import generate_nbest
from htr.evaluation.evaluate import evaluate
from htr.experiments.ablation import run_ablations
from htr.training.engine import train
from htr.utils.io import file_hash


def test_training_pipeline(tmp_path, tiny_components):
    root = tmp_path
    torch.set_num_threads(1)
    cfg = Config(project_root=str(root))
    cfg.data.image_root = "dummy"
    cfg.data.source_csv = "dummy/metadata.csv"
    generate_dummy(root / "dummy", count=8)
    prepare_splits(cfg)
    model, processor = tiny_components
    base = root / "tiny_base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    cfg.model = ModelConfig(
        name=str(base),
        device="cpu",
        dtype="float32",
        prompt="Transcribe.",
        min_pixels=64,
        max_pixels=256,
    )
    cfg.lora = LoraConfig(r=2, alpha=4, dropout=0)
    cfg.train.epochs = 1
    cfg.train.gradient_accumulation = 2
    cfg.decode.max_new_tokens = 3
    train(cfg, components=(model, processor))
    cfg.train.checkpoint = "runs/sft/best"
    for beams in (1, 2, 4):
        cfg.decode.num_beams = beams
        cfg.decode.output = f"outputs/beam{beams}.jsonl"
        infer(cfg)
        evaluate(cfg)
    cfg.stackmix.enabled = True
    cfg.stackmix.synthetic_dir = "dummy/synthetic"
    cfg.stackmix.ctc_epochs = 1
    cfg.stackmix.min_quality = 0
    cfg.stackmix.max_alignment_cer = 10
    cfg.stackmix.min_words = cfg.stackmix.max_words = 2
    train_ctc(cfg)
    build_bank(cfg)
    synthesize(cfg)
    cfg.train.output_dir = "runs/stackmix_sft"
    cfg.train.checkpoint = None
    train(cfg)
    cfg.train.checkpoint = "runs/stackmix_sft/best"
    generate_nbest(cfg)
    cfg.train.output_dir = "runs/mwer"
    cfg.train.learning_rate = 1e-05
    train(cfg, stage="mwer")
    assert file_hash(root / "runs/stackmix_sft/best/merger.safetensors") == file_hash(
        root / "runs/mwer/best/merger.safetensors"
    )
    cfg.train.checkpoint = "runs/mwer/best"
    cfg.decode.output = "outputs/mwer.jsonl"
    infer(cfg)
    evaluate(cfg)
    cfg.train.checkpoint = None
    cfg.train.learning_rate = 0.0002
    cfg.mwer.ablation_epochs = 1
    cfg.mwer.nbest_path = "artifacts/nbest/ablation.jsonl"
    run_ablations(cfg, [f"E{i}" for i in range(6)], "runs/ablations")

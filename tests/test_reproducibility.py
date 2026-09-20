from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file

from htr.config import Config, LoraConfig, ModelConfig, load_config
from htr.data.dummy import generate_dummy
from htr.data.split import prepare_splits
from htr.experiments.ablation import run_ablations
from htr.experiments.aggregate import aggregate
from htr.testing import tiny_components
from htr.training.engine import train
from htr.utils.io import write_json


def test_epoch_resume_matches_uninterrupted_with_dropout(tmp_path):
    torch.set_num_threads(1)
    model, processor = tiny_components()
    base = tmp_path / "base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    cfg = Config(project_root=str(tmp_path))
    cfg.model = ModelConfig(
        name=str(base), device="cpu", dtype="float32", prompt="Read.", min_pixels=64, max_pixels=256
    )
    cfg.lora = LoraConfig(r=2, alpha=4, dropout=0.1)
    cfg.data.image_root = "dummy"
    cfg.data.source_csv = "dummy/metadata.csv"
    cfg.train.batch_size = 2
    cfg.train.gradient_accumulation = 2
    cfg.decode.max_new_tokens = 2
    generate_dummy(tmp_path / "dummy", count=8)
    prepare_splits(cfg)
    full = deepcopy(cfg)
    full.train.output_dir, full.train.epochs = "runs/full", 2
    train(full)
    cfg.train.output_dir, cfg.train.epochs = "runs/resumed", 1
    train(cfg)
    cfg.train.resume, cfg.train.epochs = "runs/resumed/last", 2
    train(cfg)
    for filename in ("adapter_model.safetensors", "merger.safetensors"):
        expected = load_file(str(tmp_path / "runs/full/last" / filename))
        actual = load_file(str(tmp_path / "runs/resumed/last" / filename))
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_config_inheritance_root_is_relative_to_declaration(tmp_path):
    (tmp_path / "configs/nested").mkdir(parents=True)
    (tmp_path / "configs/base.yaml").write_text("project_root: ..\nseed: 123\n")
    (tmp_path / "configs/nested/child.yaml").write_text("extends: ../base.yaml\n")
    cfg = load_config(tmp_path / "configs/nested/child.yaml", ["crop.enabled=false"])
    assert cfg.project_root == str(tmp_path)
    assert cfg.seed == 123 and cfg.crop.enabled is False
    (tmp_path / "cycle.yaml").write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "cycle.yaml")


def test_ablation_plan_and_aggregation_guard(tmp_path):
    cfg = Config(project_root=str(tmp_path))
    plan = run_ablations(cfg, ["E0", "E3", "E5"], "plan", dry_run=True)
    assert plan["training_stages"] == ["stackmix_sft", "mwer"]
    assert plan["experiments"]["E3"]["checkpoint_stage"] == "stackmix_sft"
    paths = []
    # Hand-specified unit-test data; not exported as reported research results.
    for i in range(2):
        path = tmp_path / f"test{i}.json"
        write_json(
            path,
            {
                "config": cfg.to_dict(),
                "experiment": f"test{i}",
                "seed": 42,
                "evaluation_split_hash": str(i),
                "cer": 0.1,
                "wer": 0.2,
                "score": 0.15,
                "average_generation_seconds": 1.0,
                "average_output_tokens": 10,
                "average_output_chars": 15,
                "model_checkpoint": "unit-test",
            },
        )
        paths.append(path)
    with pytest.raises(ValueError, match="Different evaluation"):
        aggregate(paths, tmp_path / "table")
    assert len(aggregate(paths, tmp_path / "table", allow_mixed=True)) == 2


@pytest.mark.parametrize(
    "setting",
    [
        "lora.bias=all",
        "stackmix.word_spacing=0",
        "decode.num_beams=0",
        "train.early_stopping_patience=0",
    ],
)
def test_invalid_config_fails_early(tmp_path, setting):
    path = tmp_path / "config.yaml"
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_config(path, [setting])

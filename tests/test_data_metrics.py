from dataclasses import replace

import pytest

from htr.config import Config, load_config
from htr.data.dataset import Sample, read_samples
from htr.data.dummy import generate_dummy
from htr.data.split import assert_disjoint, load_splits, prepare_splits, split_samples
from htr.evaluation.metrics import corpus_metrics, edit_distance, sample_metrics


def test_metrics():
    assert edit_distance("kitten", "sitting") == 3
    result = sample_metrics("a b", "a c")
    assert result["cer"] == pytest.approx(1 / 3)
    assert result["wer"] == 0.5
    assert result["score"] == pytest.approx(5 / 12)
    assert sample_metrics("Old!", "old")["cer"] == 0.5
    assert sample_metrics("", "two words")["wer"] == 2
    result = corpus_metrics(["a", "a b c"], ["x", "a b c"])
    assert result["wer"] == 0.25
    assert result["cer"] == pytest.approx(1 / 6)


def test_split_and_leakage(tmp_path):
    csv = generate_dummy(tmp_path)
    samples = read_samples(csv, tmp_path, "writer_id")
    train, val = split_samples(samples, 0.2, 42)
    assert (train, val) == split_samples(samples, 0.2, 42)
    assert_disjoint(train, val)
    for field in ("image_path", "image_id", "image_hash", "group"):
        leaked = replace(val[0], **{field: getattr(train[0], field)})
        with pytest.raises(ValueError, match="leakage"):
            assert_disjoint(train, [leaked])


def test_config_and_manifest(tmp_path):
    generate_dummy(tmp_path / "dummy")
    cfg = Config(project_root=str(tmp_path))
    cfg.data.image_root = "dummy"
    cfg.data.source_csv = "dummy/metadata.csv"
    prepare_splits(cfg)
    assert len(load_splits(cfg)["train"]) == 16
    (tmp_path / "bad.yaml").write_text("train:\n  learnng_rate: 1")
    with pytest.raises(TypeError):
        load_config(tmp_path / "bad.yaml")


def test_path_escape(tmp_path):
    (tmp_path / "bad.csv").write_text("image_path,transcription\n../outside.png,text\n")
    with pytest.raises(ValueError, match="escapes"):
        read_samples(tmp_path / "bad.csv", tmp_path)


def test_group_required():
    with pytest.raises(ValueError, match="two groups"):
        split_samples([Sample("a", "a.png", "a", "writer")], 0.2, 42)

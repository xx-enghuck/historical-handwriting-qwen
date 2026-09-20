from copy import deepcopy

import pytest
import torch

from htr.config import Config
from htr.data.dataset import samples_hash
from htr.data.dummy import generate_dummy
from htr.data.split import load_splits, prepare_splits
from htr.data.stackmix.aligner import AlignmentError, forced_ctc_path, segments_from_ctc
from htr.data.stackmix.augment import load_synthetic, synthesize
from htr.data.stackmix.ctc import train_ctc
from htr.data.stackmix.segment_bank import build_bank, verify_bank


def test_forced_alignment_repeated_characters_and_nonuniform_boundaries():
    logits = torch.full((9, 3), -10.0)
    tokens = [0, 1, 1, 0, 1, 0, 2, 2, 0]
    logits[range(9), tokens] = 10
    probs = logits.log_softmax(-1)
    path = forced_ctc_path(probs, [1, 1, 2])
    assert [path[i] for i in (1, 4, 6)] == [1, 3, 5]
    segments = segments_from_ctc(probs, "aab", [1, 1, 2], 90)
    assert [s.right - s.left for s in segments] == [35, 20, 35]
    assert all(s.quality > 0.99 for s in segments)
    with pytest.raises(AlignmentError):
        forced_ctc_path(probs[:2], [1, 1])


def test_train_only_bank_synthesis(tmp_path):
    torch.set_num_threads(1)
    cfg = Config(project_root=str(tmp_path))
    cfg.model.device = "cpu"
    cfg.data.image_root = "dummy"
    cfg.data.source_csv = "dummy/metadata.csv"
    cfg.stackmix.enabled = True
    cfg.stackmix.synthetic_dir = "dummy/synthetic"
    cfg.stackmix.ctc_epochs = 1
    # Explicitly weak settings test software flow, never research-quality boundaries.
    cfg.stackmix.min_quality = 0
    cfg.stackmix.max_alignment_cer = 10
    cfg.stackmix.min_words = cfg.stackmix.max_words = 2
    generate_dummy(tmp_path / "dummy", count=8)
    prepare_splits(cfg)
    train = load_splits(cfg)["train"]
    train_ctc(cfg)
    bank = build_bank(cfg)
    assert bank["signature"]["train_split_hash"] == samples_hash(train)
    assert {s["source_id"] for s in bank["segments"]} <= {s.image_id for s in train}
    assert build_bank(cfg) == bank
    bad = deepcopy(bank)
    bad["segments"][0]["source_id"] = "validation-id"
    with pytest.raises(ValueError, match="non-training"):
        verify_bank(bad, train)
    synthesize(cfg)
    synthetic = load_synthetic(cfg, train)
    assert len(synthetic) == len(train)
    assert all(s.transcription for s in synthetic)

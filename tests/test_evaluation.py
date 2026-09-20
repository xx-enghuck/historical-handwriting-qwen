import csv

import pytest
import torch

from htr.config import Config, ModelConfig
from htr.data.dummy import generate_dummy
from htr.decoding.infer import infer
from htr.evaluation.evaluate import evaluate
from htr.testing import tiny_components


def test_image_only_inference_and_evaluation_provenance(tmp_path):
    torch.set_num_threads(1)
    cfg = Config(project_root=str(tmp_path))
    cfg.data.image_root = "dummy"
    cfg.model = ModelConfig(device="cpu", prompt="Read.")
    cfg.decode.input_csv = "dummy/metadata.csv"
    cfg.decode.max_new_tokens = 2
    csv_path = generate_dummy(tmp_path / "dummy", count=2)
    components = tiny_components()
    first = infer(cfg, components=components)
    metrics = evaluate(cfg, reference_csv="dummy/metadata.csv")
    assert metrics["samples"] == 2
    assert metrics["training_loss"] is None
    # Even if labels are present, changing them cannot affect generated IDs/text.
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["transcription"] = "UNUSED test ground truth"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    second = infer(cfg, components=components)
    assert [row["prediction"] for row in first] == [row["prediction"] for row in second]
    path = cfg.path(cfg.decode.output)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        evaluate(cfg, reference_csv="dummy/metadata.csv")

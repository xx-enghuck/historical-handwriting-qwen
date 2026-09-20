"""Join saved predictions to references only after decoding has finished."""

from dataclasses import asdict
from pathlib import Path

from htr.config import Config
from htr.data.dataset import image_set_hash, read_samples, samples_hash
from htr.data.split import load_splits
from htr.evaluation.metrics import corpus_metrics
from htr.utils.io import file_hash, read_json, read_jsonl, write_json, write_jsonl


def evaluate(cfg: Config, split: str = "validation", reference_csv: str | None = None) -> dict:
    samples = (
        read_samples(cfg.path(reference_csv), cfg.path(cfg.data.image_root))
        if reference_csv
        else load_splits(cfg)[split]
    )
    if split == "test" and not reference_csv:
        raise ValueError(
            "Test manifests contain no labels; supply a held-out reference CSV after inference"
        )
    path = cfg.path(cfg.decode.output)
    metadata = read_json(path.with_suffix(".metadata.json"))
    if metadata["predictions_hash"] != file_hash(path) or metadata[
        "image_set_hash"
    ] != image_set_hash(samples):
        raise ValueError("Prediction file or reference image set differs from inference provenance")
    predictions = read_jsonl(path)
    by_id = {r["image_id"]: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {s.image_id for s in samples}:
        raise ValueError(
            "Predictions must cover reference IDs exactly once; no missing/extra samples"
        )
    ordered = [by_id[s.image_id] for s in samples]
    metrics = corpus_metrics(
        [s.transcription for s in samples], [p["prediction"] for p in ordered], cfg.normalization
    )
    per_sample = metrics.pop("per_sample")
    write_jsonl(
        path.with_suffix(".per_sample.jsonl"),
        [
            {
                "image_id": sample.image_id,
                **row,
                "reference": sample.transcription,
                "prediction": pred["prediction"],
            }
            for sample, row, pred in zip(samples, per_sample, ordered, strict=True)
        ],
    )
    result = {
        **metadata,
        **metrics,
        "evaluation_split": split,
        "evaluation_split_hash": samples_hash(samples),
        "metric_normalization": asdict(cfg.normalization),
        "samples": len(samples),
        "average_generation_seconds": sum(p["generation_seconds"] for p in ordered) / len(samples),
        "average_output_tokens": sum(p["output_tokens"] for p in ordered) / len(samples),
        "average_output_chars": sum(p["output_chars"] for p in ordered) / len(samples),
        "training_loss": None,
        "validation_loss": None,
    }
    if metadata.get("checkpoint_hash"):
        train_result = Path(metadata["model_checkpoint"]) / "validation.json"
        if train_result.exists():
            train_metrics = read_json(train_result)
            result.update({key: train_metrics[key] for key in ("training_loss", "validation_loss")})
    write_json(path.with_suffix(".metrics.json"), result)
    return result

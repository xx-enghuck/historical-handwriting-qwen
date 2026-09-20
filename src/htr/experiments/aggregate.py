"""Aggregate measured runs only; reject incomparable evaluation protocols."""

import csv
from pathlib import Path

from htr.utils.io import digest, read_json


def aggregate(paths: list[Path], output: Path, allow_mixed: bool = False) -> list[dict]:
    if not paths:
        raise ValueError("No result files found")
    rows, protocols = [], set()
    for path in sorted(paths):
        result = read_json(path)
        config = result["config"]
        protocols.add(
            digest(
                {
                    "split": result["evaluation_split_hash"],
                    "normalization": result.get("metric_normalization", config["normalization"]),
                }
            )
        )
        rows.append(
            {
                "experiment": result.get("experiment", path.stem),
                "seed": result["seed"],
                "beams": config["decode"]["num_beams"],
                **{
                    key: result[key]
                    for key in (
                        "cer",
                        "wer",
                        "score",
                        "average_generation_seconds",
                        "average_output_tokens",
                        "average_output_chars",
                    )
                },
                "split_hash": result["evaluation_split_hash"],
                "checkpoint": result["model_checkpoint"],
                "source": str(path),
            }
        )
    if len(protocols) > 1 and not allow_mixed:
        raise ValueError(
            "Different evaluation splits/normalization cannot share an ablation table; use --allow-mixed to label exploratory results"
        )
    keys = list(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    display = [
        "experiment",
        "seed",
        "beams",
        "cer",
        "wer",
        "score",
        "average_generation_seconds",
        "average_output_tokens",
    ]
    lines = ["| " + " | ".join(display) + " |", "| " + " | ".join(["---"] * len(display)) + " |"]
    for row in rows:
        values = [f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in display]
        lines.append("| " + " | ".join(values) + " |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return rows

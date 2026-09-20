"""CSV ingestion that preserves transcriptions verbatim."""

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from htr.utils.io import digest, file_hash


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: str
    transcription: str
    group: str | None = None
    image_hash: str = ""


def image_path(root: Path, relative: str) -> Path:
    """Resolve image paths inside an explicit image root, including symlink checks."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes image_root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_samples(
    csv_path: Path, root: Path, group_column: str | None = None, require_labels: bool = True
) -> list[Sample]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path"} | ({"transcription"} if require_labels else set())
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{csv_path} requires columns: {sorted(required)}")
        if group_column and group_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Missing group column {group_column!r}; configure null for line splitting"
            )
        result = []
        for row in reader:
            path = image_path(root, row["image_path"])
            relative = path.relative_to(root.resolve()).as_posix()
            group = row[group_column] if group_column else None
            if group_column and not group:
                raise ValueError(f"Empty group for {relative}")
            text = row["transcription"] if require_labels else ""
            if text is None:
                raise ValueError(f"Missing transcription for {relative}")
            supplied_id = row.get("image_id")
            if supplied_id and not re.fullmatch(r"[A-Za-z0-9_-]+", supplied_id):
                raise ValueError(
                    "image_id must contain only ASCII letters, digits, underscore or hyphen"
                )
            result.append(
                Sample(
                    row.get("image_id") or digest(relative)[:20],
                    relative,
                    text,
                    group,
                    file_hash(path),
                )
            )
    if not result:
        raise ValueError(f"Empty dataset: {csv_path}")
    for field in ("image_id", "image_path"):
        if len({getattr(row, field) for row in result}) != len(result):
            raise ValueError(f"Duplicate {field} in {csv_path}")
    return result


def load_image(sample: Sample, root: Path) -> Image.Image:
    with Image.open(image_path(root, sample.image_path)) as image:
        return image.convert("RGB")


def samples_hash(samples: list[Sample]) -> str:
    return digest([asdict(row) for row in sorted(samples, key=lambda x: x.image_id)])


def image_set_hash(samples: list[Sample]) -> str:
    """Fingerprint prediction inputs independently of labels and grouping metadata."""
    return digest(
        [
            {key: getattr(s, key) for key in ("image_id", "image_path", "image_hash")}
            for s in sorted(samples, key=lambda row: row.image_id)
        ]
    )

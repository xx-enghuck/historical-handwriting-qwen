"""Generate original printed dummy lines; these are not a handwriting benchmark."""

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_dummy(output: Path, count: int = 20, seed: int = 42) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    font = ImageFont.load_default(size=24)
    for index in range(count):
        text = f"Ledger {index}: {rng.choice(['paid', 'owed', 'received'])} {index + 3} shillings."
        image = Image.new("RGB", (540, 80), "white")
        ImageDraw.Draw(image).text((20, 24), text, fill=(25, 25, 25), font=font)
        name = f"line_{index:04d}.png"
        image.save(output / name)
        rows.append(
            {"image_path": name, "transcription": text, "writer_id": f"writer_{index // 2}"}
        )
    path = output / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "transcription", "writer_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path

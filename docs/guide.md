# Training guide

Run commands from the repository root after installation. `htr` and
`python -m htr.cli` accept the same arguments.

## Data

Edit `configs/data.yaml` for your dataset:

```yaml
project_root: ..
data:
  image_root: data/private
  source_csv: data/private/train.csv
  validation_csv: null
  test_csv: null
  group_column: writer_id
  validation_fraction: 0.2
  split_dir: artifacts/splits
```

The CSV needs `image_path` and `transcription`. Keep spelling, punctuation and
whitespace as they appear in the source. Paths are relative to `data.image_root`.

With `validation_csv: null`, the split holds out entire writer groups. Set
`group_column` to a document column instead, or explicitly to `null` for a
line-level split. An optional test CSV supplies images without loading labels.
Overlapping IDs, paths, image hashes and groups are rejected. Use a new
`split_dir` when changing data. Near-duplicates still need manual review.

```bash
htr prepare-data --config configs/data.yaml
```

For a local fixture with the default config, run
`htr dummy --output data/dummy --count 20 --seed 42` first.

## Supervised fine-tuning

SFT trains the language model's LoRA adapters and visual merger. The vision
encoder and base language weights stay frozen. Loss covers transcription tokens
and EOS; image, prompt and padding tokens are masked.

```bash
htr train-sft --config configs/sft.yaml

# Continue from the last completed epoch in the same run directory
htr train-sft --config configs/sft.yaml \
  --set train.resume=runs/sft/last --set train.epochs=6
```

YAML configs inherit through `extends`; `--set` overrides a dotted key.
`project_root` is relative to the config that declares it. Pin `model.revision`
to a Hub commit for a reproducible run. For exact resume, only `train.epochs`
and `train.resume` may change.

`best/` is selected by validation `0.5 * CER + 0.5 * WER`; `last/` stores the most
recent epoch. Bundles include LoRA, merger weights, processor and base-model
configuration. Run metadata records the environment, seed, split hashes and settings.

## StackMix-style augmentation

An auxiliary CTC recognizer aligns characters in training lines. Accepted
segments form a bank; synthetic lines use words from training transcriptions.
This is a character-segment variant of StackMix. Inspect alignments before using
it on cursive text; the small CTC model needs a reasonable fit to produce a bank.

```bash
htr train-ctc --config configs/stackmix.yaml
htr build-stackmix-bank --config configs/stackmix.yaml
htr synthesize --config configs/stackmix.yaml
htr train-sft --config configs/stackmix.yaml
```

Set `stackmix.synthetic_dir` under your `data.image_root`. The default ratio of
1.0 adds one synthetic line per training line. Validation and test data are excluded.

## MWER

Generate a fixed Beam-4 N-best cache from the augmented SFT checkpoint, then train
LoRA with expected WER and optional reference cross-entropy (`lambda_ce=0.1`).
The merger stays frozen at its SFT weights. Candidate scores are recomputed with
teacher forcing; WER risks are detached. Candidates are not refreshed during training.

```bash
htr generate-nbest --config configs/mwer.yaml
htr train-mwer --config configs/mwer.yaml
```

`mwer.length_normalization` controls sequence-score length normalization;
`mwer.lambda_ce=0` selects pure MWER. Change artifact paths when changing the
checkpoint, data or decoding settings, since cached candidates bind to those inputs.

## Inference and evaluation

```bash
# SFT checkpoint, validation split
htr infer --config configs/inference.yaml --set decode.num_beams=4
htr evaluate --config configs/inference.yaml

# MWER checkpoint, external image CSV
htr infer --config configs/inference.yaml \
  --set train.checkpoint=runs/mwer/best \
  --set decode.input_csv=data/private/test_images.csv \
  --set decode.num_beams=4 --set decode.output=outputs/test.jsonl
```

External inference CSVs need `image_path` and optional `image_id`. A transcription
column is ignored. For zero-shot inference, set `train.checkpoint=null`.
Use the crop, prompt and pixel settings from training.

Evaluate held-out labels only after saving predictions:

```bash
htr evaluate --config configs/inference.yaml --split test \
  --references-csv data/private/test_references.csv \
  --set train.checkpoint=runs/mwer/best --set decode.output=outputs/test.jsonl
```

CER uses Unicode code points; WER uses whitespace-separated words. Both are corpus
edit rates, reported as fractions. Normalization is off by default. Prediction IDs
must match references exactly. Evaluation writes metrics and per-sample JSONL files.

## Ablations

| Run | Training | Decoding |
| --- | --- | --- |
| E0 | Pretrained model | Greedy |
| E1 | SFT on original lines | Greedy |
| E2 | SFT with StackMix | Greedy |
| E3 | E2 checkpoint | Beam-4 |
| E4 | E2 + MWER | Greedy |
| E5 | E4 checkpoint | Beam-4 |

```bash
htr ablations --config configs/stackmix.yaml --dry-run --output runs/ablations
htr ablations --config configs/stackmix.yaml --output runs/ablations
htr aggregate --results runs/ablations/E*.metrics.json --output outputs/ablation
```

The runner shares checkpoints across decoding comparisons and evaluates validation
only. SFT uses `train.epochs` and `train.learning_rate`; MWER uses
`mwer.ablation_epochs` and `mwer.ablation_learning_rate`. Fresh experiments need
fresh output paths. StackMix increases training updates when epochs are held fixed.

## Scope

The implementation uses a single device. Distributed training, quantization,
full-page layout analysis and online N-best refresh are not implemented.

## References

- [Qwen2.5-VL model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [StackMix](https://arxiv.org/abs/2108.11667)
- [Minimum Word Error Rate Training](https://arxiv.org/abs/1712.01818)

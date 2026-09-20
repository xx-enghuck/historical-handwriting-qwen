# Verification record

Local software verification used Python 3.13, Torch 2.14.0, Transformers 5.17.0
and PEFT 0.21.0 on macOS ARM64 without CUDA.

| Phase | Software checks | Offline smoke scope |
| --- | --- | --- |
| 1 | Metrics, split/group leakage, config | Original dummy images and train/validation manifests |
| 2 | Assistant masking, Qwen LoRA layout, merger gradients | Image SFT, adapter/merger save, reload and resume |
| 3 | Greedy/Beam-2/Beam-4 determinism; no generation gradients | Decode and evaluate saved predictions |
| 4 | CTC blank/repeat transitions, nonuniform boundaries, train-only sources | CTC training, bank, synthetic images, augmented SFT |
| 5 | Exact toy loss/gradient, detached risks, real LoRA gradients | N-best, MWER, frozen-merger equality, final inference |
| 6 | Inheritance, resume equality, aggregation compatibility | E0–E5 runner and CSV/Markdown output |

`scripts/smoke.py --phase N` supports phases 1–6. It generates fixtures inside a
temporary directory, removes them on exit, and downloads no pretrained weights.
It constructs the real Qwen model/processor with tiny random weights and a local
tokenizer. CTC quality thresholds are relaxed explicitly to exercise the pipeline.
Random-model metrics are not research results.

The resume test compares adapter and merger tensors bit for bit after two
uninterrupted epochs versus one epoch plus resume, with LoRA dropout enabled.
MWER tests cover ordinary/checkpointed scoring, causal shift, label masks, length
normalization, probability sums, known gradients, and frozen parameters/risks.

Not verified here: pretrained 3B forward/training, CUDA/bf16 kernels, GPU capacity,
historical handwriting alignment quality, convergence, accuracy improvements,
or execution of the hosted GitHub Actions workflow.

Leakage checks cover canonical paths, identical image bytes, IDs, and one grouping
relation. They cannot establish near-duplicate semantics or unknown pretraining
provenance. Augmentation uses only training records. Test manifests omit labels.
Decoding accepts only image, prompt and generation settings; reference evaluation
joins saved predictions afterward.

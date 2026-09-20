"""Ablations reuse exact SFT/MWER checkpoints across decoding comparisons."""

from copy import deepcopy
from dataclasses import replace

from htr.config import Config
from htr.data.split import prepare_splits
from htr.decoding.infer import infer
from htr.evaluation.evaluate import evaluate
from htr.experiments.aggregate import aggregate
from htr.training.engine import train
from htr.utils.io import read_json, write_json

EXPERIMENTS = {
    "E0": ("zero_shot", 1),
    "E1": ("sft", 1),
    "E2": ("stackmix_sft", 1),
    "E3": ("stackmix_sft", 4),
    "E4": ("mwer", 1),
    "E5": ("mwer", 4),
}


def run_ablations(cfg: Config, experiments: list[str], output: str, dry_run: bool = False) -> dict:
    """Build prerequisite checkpoints once. Existing training runs must match their configs."""
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    if cfg.train.resume:
        raise ValueError(
            "Resume individual stages first; ablation runner requires train.resume=null"
        )
    base = deepcopy(cfg)
    stages = {EXPERIMENTS[e][0] for e in experiments}
    required = [
        s
        for s in ("sft", "stackmix_sft", "mwer")
        if s in stages or (s == "stackmix_sft" and "mwer" in stages)
    ]
    plan = {
        "experiments": {
            e: {"checkpoint_stage": EXPERIMENTS[e][0], "beams": EXPERIMENTS[e][1]}
            for e in experiments
        },
        "training_stages": required,
        "output": output,
        "config": cfg.to_dict(),
    }
    write_json(cfg.path(output) / "plan.json", plan)
    if dry_run:
        return plan
    prepare_splits(cfg)
    stage_configs = {}
    for stage in required:
        current = deepcopy(base)
        current.train.output_dir = f"{output}/{stage}"
        current.stackmix.enabled = stage != "sft"
        if stage == "mwer":
            current.train.learning_rate = cfg.mwer.ablation_learning_rate
            current.train.epochs = cfg.mwer.ablation_epochs
            current.train.checkpoint = f"{output}/stackmix_sft/best"
        else:
            current.train.checkpoint = None
        stage_configs[stage] = current
        result_path = current.path(current.train.output_dir) / "result.json"
        if result_path.exists():
            if read_json(result_path)["config"] != current.to_dict():
                raise ValueError(f"Existing {stage} settings differ; choose a new ablation output")
            continue
        if stage == "stackmix_sft":
            from htr.data.stackmix.augment import synthesize
            from htr.data.stackmix.ctc import train_ctc
            from htr.data.stackmix.segment_bank import build_bank

            if not current.path(current.stackmix.ctc_checkpoint).exists():
                train_ctc(current)
            build_bank(current)
            synthesize(current)
        if stage == "mwer":
            from htr.data.split import load_splits
            from htr.decoding.nbest import generate_nbest
            from htr.training.mwer import load_nbest

            if not current.path(current.mwer.nbest_path).exists():
                generate_nbest(current)
            else:
                load_nbest(current, load_splits(current)["train"])
        train(current, stage="mwer" if stage == "mwer" else "sft")
    paths = []
    for experiment in experiments:
        stage, beams = EXPERIMENTS[experiment]
        current = deepcopy(stage_configs.get(stage, base))
        current.train.checkpoint = None if stage == "zero_shot" else f"{output}/{stage}/best"
        current.decode = replace(
            current.decode,
            num_beams=beams,
            num_return_sequences=1,
            input_csv=None,
            output=f"{output}/{experiment}.jsonl",
        )
        infer(current, "validation")
        result = evaluate(current, "validation")
        result["experiment"] = experiment
        path = current.path(current.decode.output).with_suffix(".metrics.json")
        write_json(path, result)
        paths.append(path)
    aggregate(paths, cfg.path(output) / "ablation")
    return plan

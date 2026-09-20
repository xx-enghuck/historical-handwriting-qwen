"""Commands for data preparation, training and evaluation."""

import argparse
import logging
from pathlib import Path

from htr.config import load_config
from htr.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Handwriting recognition with Qwen2.5-VL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare-data",
        "train-sft",
        "train-ctc",
        "build-stackmix-bank",
        "synthesize",
        "generate-nbest",
        "train-mwer",
        "infer",
        "evaluate",
        "ablations",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if command in {"infer", "evaluate"}:
            child.add_argument(
                "--split", choices=["validation", "test", "train"], default="validation"
            )
        if command == "evaluate":
            child.add_argument(
                "--references-csv", help="Held-out labels read only after predictions exist"
            )
        if command == "ablations":
            child.add_argument(
                "--experiments", nargs="+", default=list("E" + str(i) for i in range(6))
            )
            child.add_argument("--output", default="runs/ablations")
            child.add_argument("--dry-run", action="store_true")
    agg = subparsers.add_parser("aggregate")
    agg.add_argument("--results", nargs="+", type=Path, required=True)
    agg.add_argument("--output", type=Path, required=True)
    agg.add_argument("--allow-mixed", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "aggregate":
        from htr.experiments.aggregate import aggregate

        aggregate(args.results, args.output, args.allow_mixed)
        return
    cfg = load_config(args.config, args.set)
    if args.command == "prepare-data":
        from htr.data.split import prepare_splits

        result = prepare_splits(cfg)
        logging.info("Split counts: %s", {k: len(v) for k, v in result["splits"].items()})
    elif args.command in {"train-sft", "train-mwer"}:
        from htr.training.engine import train

        train(cfg, "mwer" if args.command == "train-mwer" else "sft")
    elif args.command == "train-ctc":
        from htr.data.stackmix.ctc import train_ctc

        train_ctc(cfg)
    elif args.command == "build-stackmix-bank":
        from htr.data.stackmix.segment_bank import build_bank

        build_bank(cfg)
    elif args.command == "synthesize":
        from htr.data.stackmix.augment import synthesize

        synthesize(cfg)
    elif args.command == "generate-nbest":
        from htr.decoding.nbest import generate_nbest

        generate_nbest(cfg)
    elif args.command == "infer":
        from htr.decoding.infer import infer

        infer(cfg, args.split)
    elif args.command == "evaluate":
        from htr.evaluation.evaluate import evaluate

        evaluate(cfg, args.split, args.references_csv)
    elif args.command == "ablations":
        from htr.experiments.ablation import run_ablations

        run_ablations(cfg, args.experiments, args.output, args.dry_run)


if __name__ == "__main__":
    main()

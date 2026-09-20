"""CLI wrapper for htr train-sft."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["train-sft", *sys.argv[1:]])

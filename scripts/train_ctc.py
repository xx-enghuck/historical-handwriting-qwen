"""CLI wrapper for htr train-ctc."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["train-ctc", *sys.argv[1:]])

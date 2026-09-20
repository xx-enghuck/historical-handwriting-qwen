"""CLI wrapper for htr train-mwer."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["train-mwer", *sys.argv[1:]])

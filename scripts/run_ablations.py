"""CLI wrapper for htr ablations."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["ablations", *sys.argv[1:]])

"""CLI wrapper for htr build-stackmix-bank."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["build-stackmix-bank", *sys.argv[1:]])

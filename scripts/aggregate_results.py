"""CLI wrapper for htr aggregate."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["aggregate", *sys.argv[1:]])

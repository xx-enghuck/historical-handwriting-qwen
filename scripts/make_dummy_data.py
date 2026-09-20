"""CLI wrapper for htr dummy."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["dummy", *sys.argv[1:]])

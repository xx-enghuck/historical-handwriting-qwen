"""CLI wrapper for htr prepare-data."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["prepare-data", *sys.argv[1:]])

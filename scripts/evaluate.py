"""CLI wrapper for htr evaluate."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])

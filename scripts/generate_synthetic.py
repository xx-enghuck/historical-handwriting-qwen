"""CLI wrapper for htr synthesize."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["synthesize", *sys.argv[1:]])

"""CLI wrapper for htr infer."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["infer", *sys.argv[1:]])

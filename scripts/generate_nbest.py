"""CLI wrapper for htr generate-nbest."""

import sys

from htr.cli import main

if __name__ == "__main__":
    main(["generate-nbest", *sys.argv[1:]])

#!/usr/bin/env python3

import sys

from chandra_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["match", *sys.argv[1:]]))

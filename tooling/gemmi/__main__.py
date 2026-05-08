"""CLI: python -m tooling.gemmi <oracle-dir> <function-qname> [--model M] [--verbose]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import generate_gemmi, DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Port an MMDB function + its test to gemmi in one agent session"
    )
    parser.add_argument("oracle_dir", type=Path, help="Path to oracle-data/<name>/")
    parser.add_argument("function_qname",
                        help="Qualified name of the function to port")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--commit",  action="store_true",
                        help="Commit the port into the coot source tree on success")
    args = parser.parse_args()

    try:
        test_cc = generate_gemmi(args.oracle_dir, args.function_qname,
                                 model=args.model, verbose=args.verbose,
                                 commit=args.commit)
        print(f"Generated: {test_cc}")
        print(f"Compile:   sh {test_cc.parent / 'compile_gemmi.sh'}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


main()

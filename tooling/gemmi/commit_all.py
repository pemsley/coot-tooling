"""Scan generated-tests/ and commit every gemmi port that passed.

Pass criteria (per `generated-tests/<sanitized>/gemmi/`):
  - function.hh exists
  - test.cc exists
  - run.log exists and contains "[  PASSED  ]"

Usage:
    python -m tooling.gemmi.commit_all              # dry-run, lists status
    python -m tooling.gemmi.commit_all --commit     # actually commit each pass
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from ..db import connect
from .commit import commit_gemmi_port

GENERATED_ROOT = Path("generated-tests")
PASS_MARKER = "[  PASSED  ]"


@dataclass
class PortStatus:
    sanitized: str
    qname: str
    gemmi_dir: Path
    has_function_hh: bool
    has_test_cc: bool
    has_run_log: bool
    passed: bool

    @property
    def ok(self) -> bool:
        return self.has_function_hh and self.has_test_cc and self.passed

    def reason(self) -> str:
        if not self.has_function_hh:
            return "missing function.hh"
        if not self.has_test_cc:
            return "missing test.cc"
        if not self.has_run_log:
            return "missing run.log"
        if not self.passed:
            return "test did not PASS"
        return "ok"


def _scan(root: Path) -> list[PortStatus]:
    results: list[PortStatus] = []
    if not root.exists():
        return results
    for entry in sorted(root.iterdir()):
        gemmi_dir = entry / "gemmi"
        if not gemmi_dir.is_dir():
            continue
        sanitized = entry.name
        qname = sanitized.replace("__", "::")
        run_log = gemmi_dir / "run.log"
        passed = False
        if run_log.exists():
            passed = PASS_MARKER in run_log.read_text(errors="replace")
        results.append(PortStatus(
            sanitized=sanitized,
            qname=qname,
            gemmi_dir=gemmi_dir,
            has_function_hh=(gemmi_dir / "function.hh").exists(),
            has_test_cc=(gemmi_dir / "test.cc").exists(),
            has_run_log=run_log.exists(),
            passed=passed,
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true",
                        help="Commit every passing port (otherwise dry-run)")
    parser.add_argument("--root", type=Path, default=GENERATED_ROOT,
                        help=f"Generated-tests root (default: {GENERATED_ROOT})")
    args = parser.parse_args()

    statuses = _scan(args.root)
    if not statuses:
        print(f"No gemmi outputs under {args.root}", file=sys.stderr)
        sys.exit(1)

    passing = [s for s in statuses if s.ok]
    failing = [s for s in statuses if not s.ok]

    print(f"Scanned {len(statuses)} port(s) under {args.root}")
    print(f"  passing: {len(passing)}")
    print(f"  failing: {len(failing)}")
    print()

    for s in statuses:
        mark = "PASS" if s.ok else "SKIP"
        print(f"  [{mark}] {s.qname}  ({s.reason()})")

    if not args.commit:
        print("\nDry run. Re-run with --commit to commit each passing port.")
        return

    if not passing:
        print("\nNothing to commit.")
        return

    print(f"\nCommitting {len(passing)} port(s)...")
    conn = connect()
    committed = 0
    failed: list[tuple[str, str]] = []
    try:
        for s in passing:
            print(f"  -> {s.qname}")
            try:
                commit_gemmi_port(conn, s.qname, s.gemmi_dir)
                committed += 1
            except Exception as exc:
                failed.append((s.qname, str(exc)))
                print(f"     FAILED: {exc}", file=sys.stderr)
    finally:
        conn.close()

    print(f"\nDone. Committed {committed}/{len(passing)}.")
    if failed:
        print("Failures:")
        for qname, err in failed:
            print(f"  {qname}: {err}")
        sys.exit(1)


main()

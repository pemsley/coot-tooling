#!/usr/bin/env python3
"""
Run oracle generation for all (or filtered) methods in a class, with optional
compile and test hooks executed after each oracle is generated.

Usage:
  # Generate oracles for every method in coot::molecule_t
  python -m tooling.batch "coot::molecule_t"

  # Only methods whose name contains a substring
  python -m tooling.batch "coot::molecule_t" --filter cid

  # With a compile hook — {dir} is substituted with the oracle output directory
  python -m tooling.batch "coot::molecule_t" \\
      --compile "g++ -std=c++17 -o {dir}/oracle {dir}/oracle.cc -lmmdb2"

  # With compile + test (test receives the compiled binary path via {dir})
  python -m tooling.batch "coot::molecule_t" \\
      --compile "g++ -std=c++17 -o {dir}/oracle {dir}/oracle.cc -lmmdb2" \\
      --test    "{dir}/oracle my_structure.pdb"

  # Second-pass critique, parallel workers, skip already-generated
  python -m tooling.batch "coot::molecule_t" --second-pass --workers 4 --skip-existing

Hook placeholders:
  {dir}      absolute path to the oracle output directory  (e.g. oracle-data/foo)
  {oracle}   absolute path to oracle/oracle.cc
  {second}   absolute path to oracle/oracle_second_pass.cc (may not exist)
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..db import connect, get_class_functions
from .generate import DEFAULT_MODEL, OUT_ROOT, generate_one, sanitize_name


# ── result tracking ──────────────────────────────────────────────────────────

class Result:
    def __init__(self, qname: str):
        self.qname       = qname
        self.skipped     = False
        self.generate_ok = False
        self.compile_ok: bool | None = None
        self.test_ok:    bool | None = None
        self.error:      str  | None = None

    @property
    def short(self) -> str:
        return self.qname.rsplit("::", 1)[-1]


# ── hook execution ────────────────────────────────────────────────────────────

def _run_hook(cmd_template: str, out_dir: Path) -> tuple[bool, str]:
    """Expand placeholders and run a shell hook. Returns (success, output)."""
    cmd = cmd_template.format(
        dir    = out_dir,
        oracle = out_dir / "oracle" / "oracle.cc",
        second = out_dir / "oracle" / "oracle_second_pass.cc",
    )
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, f"[_run_hook] timed out after 300s: {cmd}"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


# ── per-function worker ───────────────────────────────────────────────────────

def _process(
    qname: str,
    model: str,
    second_pass: bool,
    agent: bool,
    verbose: bool,
    compile_cmd: str | None,
    test_cmd:    str | None,
    skip_existing: bool,
) -> Result:
    r = Result(qname)
    out_dir = OUT_ROOT / sanitize_name(qname)

    if skip_existing and (out_dir / "oracle" / "oracle.cc").exists():
        r.skipped = True
        return r

    conn = connect()
    try:
        result_dir = generate_one(conn, qname, model=model, second_pass=second_pass, agent=agent, verbose=verbose)
    except urllib.error.URLError as e:
        r.error = f"Ollama unreachable: {e}"
        return r
    finally:
        conn.close()

    if result_dir is None:
        r.error = "not found in DB"
        return r

    r.generate_ok = True

    if compile_cmd:
        ok, out = _run_hook(compile_cmd, result_dir)
        r.compile_ok = ok
        if not ok:
            r.error = f"compile failed:\n{out}"
            return r

    if test_cmd:
        ok, out = _run_hook(test_cmd, result_dir)
        r.test_ok = ok
        if not ok:
            r.error = f"test failed:\n{out}"

    return r


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(results: list[Result], compile_cmd: str | None, test_cmd: str | None) -> None:
    col = {"ok": "✓", "fail": "✗", "skip": "–", "na": " "}

    def sym(val: bool | None) -> str:
        if val is None: return col["na"]
        return col["ok"] if val else col["fail"]

    header = f"{'method':<50}  gen  "
    if compile_cmd: header += "cmp  "
    if test_cmd:    header += "tst  "
    print("\n" + header)
    print("-" * len(header))

    ok = fail = skip = 0
    for r in results:
        if r.skipped:
            skip += 1
            row = f"{r.short:<50}  {col['skip']}"
        else:
            g = col["ok"] if r.generate_ok else col["fail"]
            row = f"{r.short:<50}  {g}    "
            if compile_cmd: row += f"{sym(r.compile_ok)}    "
            if test_cmd:    row += f"{sym(r.test_ok)}    "
            if r.error:
                row += f"  ← {r.error.splitlines()[0]}"
            if r.generate_ok and (r.compile_ok is not False) and (r.test_ok is not False):
                ok += 1
            else:
                fail += 1
        print(row)

    print(f"\n{ok} ok  {fail} failed  {skip} skipped  ({len(results)} total)")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate oracles for all methods in a class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("class_name", help="Fully-qualified class name, e.g. coot::molecule_t")
    parser.add_argument("--filter",       metavar="STR",  help="Only process methods containing STR")
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--second-pass",  action="store_true", help="Run critique pass on each oracle")
    parser.add_argument("--agent",        action="store_true", help="Use agentic mode with tool calls")
    parser.add_argument("--verbose",      action="store_true", help="Print thinking and tool calls to the console")
    parser.add_argument("--compile",      metavar="CMD",  help="Shell command to compile; {dir} is substituted")
    parser.add_argument("--test",         metavar="CMD",  help="Shell command to test;    {dir} is substituted")
    parser.add_argument("--workers",      type=int, default=1, metavar="N",
                        help="Parallel workers (default 1; >1 requires Ollama to handle concurrency)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip functions that already have an oracle.cc")
    parser.add_argument("--list",         action="store_true",
                        help="List matching methods and exit without generating")
    args = parser.parse_args()

    conn = connect()
    qnames = get_class_functions(conn, args.class_name)
    conn.close()

    if not qnames:
        print(f"No methods found for class: {args.class_name}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        qnames = [q for q in qnames if args.filter in q]
        if not qnames:
            print(f"No methods match filter '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    if args.list:
        for q in qnames:
            print(q)
        print(f"\n{len(qnames)} methods")
        return

    print(f"Processing {len(qnames)} methods from {args.class_name} "
          f"(model={args.model}, workers={args.workers})")

    results: list[Result] = []

    if args.workers == 1:
        for i, qname in enumerate(qnames, 1):
            print(f"[{i}/{len(qnames)}] {qname.rsplit('::', 1)[-1]} ...", end=" ", flush=True)
            r = _process(qname, args.model, args.second_pass, args.agent, args.verbose,
                         args.compile, args.test, args.skip_existing)
            results.append(r)
            if r.skipped:       print("skipped")
            elif r.error:       print(f"FAILED — {r.error.splitlines()[0]}")
            else:               print("ok")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for qname in qnames:
                f = pool.submit(_process, qname, args.model, args.second_pass, args.agent, args.verbose,
                                args.compile, args.test, args.skip_existing)
                futures[f] = qname
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                status = "skipped" if r.skipped else ("ok" if not r.error else "FAILED")
                print(f"  {r.short}: {status}")

    results.sort(key=lambda r: r.qname)
    _print_summary(results, args.compile, args.test)


if __name__ == "__main__":
    main()

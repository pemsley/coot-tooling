"""CLI for tooling.evaluate.

Examples:

  # Evaluate a single function dir (auto-detects first failing stage):
  python -m tooling.evaluate generated-tests/coot__util__number_of_residues_in_molecule

  # Force a specific stage:
  python -m tooling.evaluate generated-tests/coot__... --stage gemmi

  # Evaluate every function dir under generated-tests/:
  python -m tooling.evaluate generated-tests/ --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detect import first_failing_stage, stage_statuses, STAGES
from .evaluator import evaluate_trace, EvaluationResult, DEFAULT_MODEL


def _qname_from_dir(d: Path) -> str:
    return d.name.replace("__", "::")


def _is_function_dir(d: Path) -> bool:
    return d.is_dir() and any((d / s / "agent_trace.txt").exists() for s in STAGES)


def _evaluate_one(func_dir: Path, *, stage: str | None,
                  model: str, write: bool) -> EvaluationResult | None:
    qname = _qname_from_dir(func_dir)
    if stage is None:
        s = first_failing_stage(func_dir)
        if s is None:
            print(f"[skip] {func_dir.name}: no failing stage detected", flush=True)
            return None
        stage_name, reason = s.name, s.reason
    else:
        all_statuses = {x.name: x for x in stage_statuses(func_dir)}
        if stage not in all_statuses or not all_statuses[stage].present:
            print(f"[skip] {func_dir.name}: stage {stage!r} not attempted", flush=True)
            return None
        stage_name = stage
        reason = all_statuses[stage].reason

    trace_path = func_dir / stage_name / "agent_trace.txt"
    if not trace_path.exists():
        print(f"[skip] {func_dir.name}: no {stage_name}/agent_trace.txt", flush=True)
        return None

    print(f"[eval] {func_dir.name} :: {stage_name} ({reason})", flush=True)
    result = evaluate_trace(
        qname=qname,
        stage=stage_name,
        detection_reason=reason,
        trace_path=trace_path,
        model=model,
    )

    if write:
        out_dir = func_dir / "evaluate"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{stage_name}.json"
        out_file.write_text(json.dumps(result.to_dict(), indent=2))

    print(
        f"   → {result.failure_mode} (confidence={result.confidence})\n"
        f"     {result.note}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tooling.evaluate")
    ap.add_argument("path", type=Path,
                    help="A generated-tests/<func>/ dir, or generated-tests/ with --all.")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="Force evaluation of this stage instead of the first failing one.")
    ap.add_argument("--all", action="store_true",
                    help="Treat `path` as a parent dir and evaluate every function subdir.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"LLM model name (default: {DEFAULT_MODEL}).")
    ap.add_argument("--no-write", action="store_true",
                    help="Do not write evaluate/<stage>.json next to the trace.")
    ap.add_argument("--summary", type=Path, default=None,
                    help="With --all: write an aggregate JSON to this path.")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2

    if args.all:
        dirs = sorted(d for d in args.path.iterdir() if _is_function_dir(d))
        if not dirs:
            print(f"error: no function dirs found under {args.path}", file=sys.stderr)
            return 2
        results: list[EvaluationResult] = []
        for d in dirs:
            r = _evaluate_one(d, stage=args.stage, model=args.model,
                              write=not args.no_write)
            if r is not None:
                results.append(r)
        # Print a final tally.
        from collections import Counter
        tally = Counter(r.failure_mode for r in results)
        print("\n=== Failure-mode tally ===")
        for mode, n in tally.most_common():
            print(f"  {n:4d}  {mode}")
        if args.summary:
            args.summary.write_text(json.dumps(
                {"results": [r.to_dict() for r in results],
                 "tally": dict(tally)},
                indent=2,
            ))
            print(f"\nwrote {args.summary}")
        return 0

    if not _is_function_dir(args.path):
        print(f"error: {args.path} does not look like a function dir "
              "(no */agent_trace.txt). Did you mean --all?", file=sys.stderr)
        return 2
    r = _evaluate_one(args.path, stage=args.stage, model=args.model,
                      write=not args.no_write)
    return 0 if r is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

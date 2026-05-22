"""Detect which pipeline stage failed for a generated-tests/<func>/ directory.

Mirrors the success criteria used by tooling/batch.py so the evaluator
classifies a function the same way the batch runner does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGES = ("oracle", "test", "gemmi")


@dataclass
class StageStatus:
    name: str          # "oracle" | "test" | "gemmi"
    present: bool      # the stage was attempted (dir + agent_trace.txt exist)
    passed: bool       # the stage's success artefacts are in place
    reason: str        # short why-it-failed string


def _read_exit(p: Path) -> str | None:
    if not p.exists():
        return None
    return p.read_text().strip()


def _oracle_status(d: Path) -> StageStatus:
    stage = d / "oracle"
    present = (stage / "agent_trace.txt").exists()
    result_json = stage / "result.json"
    binary = stage / "oracle"
    run_exit = _read_exit(stage / "run.exit")
    if result_json.exists():
        return StageStatus("oracle", present, True, "result.json present")
    if not binary.exists():
        return StageStatus("oracle", present, False, "no oracle binary (compile failed)")
    if run_exit is not None and run_exit != "0":
        return StageStatus("oracle", present, False, f"oracle ran but exited {run_exit}")
    return StageStatus("oracle", present, False, "oracle binary present but no result.json")


def _test_status(d: Path) -> StageStatus:
    stage = d / "test"
    present = (stage / "agent_trace.txt").exists()
    binary = stage / "test"
    run_exit = _read_exit(stage / "run.exit")
    run_log = stage / "run.log"
    if run_exit == "0":
        return StageStatus("test", present, True, "run.exit == 0")
    if not binary.exists():
        return StageStatus("test", present, False, "no test binary (compile failed)")
    if run_exit is not None:
        return StageStatus("test", present, False, f"test binary exited {run_exit}")
    log_text = run_log.read_text() if run_log.exists() else ""
    if "[  PASSED  ]" in log_text:
        return StageStatus("test", present, True, "run.log shows PASSED (legacy)")
    return StageStatus("test", present, False, "test binary present, no PASS evidence")


def _gemmi_status(d: Path) -> StageStatus:
    stage = d / "gemmi"
    present = (stage / "agent_trace.txt").exists()
    has_files = (stage / "function.hh").exists() and (stage / "test.cc").exists()
    binary = stage / "test_check"
    run_exit = _read_exit(stage / "run.exit")
    run_log = stage / "run.log"
    if not has_files:
        return StageStatus("gemmi", present, False, "missing function.hh or test.cc")
    if run_exit == "0":
        return StageStatus("gemmi", present, True, "run.exit == 0")
    if not binary.exists():
        return StageStatus("gemmi", present, False, "no test_check binary (compile failed)")
    if run_exit is not None:
        return StageStatus("gemmi", present, False, f"test_check exited {run_exit}")
    log_text = run_log.read_text() if run_log.exists() else ""
    if "[  PASSED  ]" in log_text:
        return StageStatus("gemmi", present, True, "run.log shows PASSED (legacy)")
    return StageStatus("gemmi", present, False, "test_check present, no PASS evidence")


def stage_statuses(func_dir: Path) -> list[StageStatus]:
    return [_oracle_status(func_dir), _test_status(func_dir), _gemmi_status(func_dir)]


def first_failing_stage(func_dir: Path) -> StageStatus | None:
    """Return the earliest stage that was attempted but did not pass.

    Returns None if every attempted stage passed (or none were attempted).
    """
    for s in stage_statuses(func_dir):
        if s.present and not s.passed:
            return s
    return None

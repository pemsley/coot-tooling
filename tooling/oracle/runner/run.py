"""Run compiled oracle binaries and return structured results."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from .results import OracleResult, parse_output, save_result


RUN_TIMEOUT_SECONDS = 20


def run_binary(binary: Path, args: list[str] | None = None, cwd: Path | None = None,
               attempts: int = 2) -> tuple[int, str, str]:
    """Run a binary, return (returncode, stdout, stderr).

    Retries once on timeout — the hang appears non-deterministic and a fresh
    invocation typically completes in milliseconds.
    """
    cmd = [str(binary.absolute())] + (args or [])
    cwd_str = str(cwd) if cwd else str(binary.parent)
    last_err = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=cwd_str, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT_SECONDS)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            last_err = f"[run_binary] timed out after {RUN_TIMEOUT_SECONDS}s (attempt {attempt}/{attempts})"
    return -1, "", last_err


def run_oracle(oracle_dir: Path) -> OracleResult:
    """Run the compiled oracle in oracle_dir, parse its output, and save result.json."""
    binary = oracle_dir / "oracle"
    if not binary.exists():
        return OracleResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=f"Binary not found: {binary}",
            inputs={},
            outputs={},
        )

    returncode, stdout, stderr = run_binary(binary, cwd=oracle_dir)
    result = parse_output(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )

    # Save stdout/stderr logs alongside the binary.
    (oracle_dir / "run.log").write_text(stdout + stderr)
    save_result(oracle_dir / "result.json", result)

    return result

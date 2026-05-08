"""Run compiled oracle binaries and return structured results."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .results import OracleResult, parse_output, save_result


RUN_TIMEOUT_SECONDS = 300


def run_binary(binary: Path, args: list[str] | None = None, cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a binary, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            [str(binary.absolute())] + (args or []),
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else str(binary.parent),
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return -1, stdout, stderr + f"\n[run_binary] timed out after {RUN_TIMEOUT_SECONDS}s"
    return result.returncode, result.stdout, result.stderr


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

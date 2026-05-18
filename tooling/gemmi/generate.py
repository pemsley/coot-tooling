"""Drive the combined gemmi port + test generation."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from ..db import connect, get_function
from .compile import compile_gemmi, run_gemmi_test_binary, write_compile_script
from .agent import _dep_extra_includes, _dep_extra_sources
from .commit import commit_gemmi_port

DEFAULT_MODEL = "qwen3.6"


def _write_files(oracle_dir: Path, blocks: dict[str, str]) -> Path:
    gemmi_subdir = oracle_dir / "gemmi"
    gemmi_subdir.mkdir(exist_ok=True)
    (gemmi_subdir / "function.hh").write_text(blocks["function.hh"])
    (gemmi_subdir / "test.cc").write_text(blocks["test.cc"])
    has_cc = "function.cc" in blocks and blocks["function.cc"].strip()
    if has_cc:
        (gemmi_subdir / "function.cc").write_text(blocks["function.cc"])
    return gemmi_subdir / "test.cc"


def generate_gemmi(
    oracle_dir: Path,
    function_qname: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    conn: sqlite3.Connection | None = None,
    commit: bool = False,
    sig_hash: str | None = None,
) -> Path:
    """Emit oracle_dir/gemmi/{function.hh, [function.cc], test.cc}.

    Requires oracle_dir/test/test.cc to exist (the MMDB test whose
    assertions are carried over unchanged). `sig_hash` pins which overload
    of `function_qname` is being ported.
    """
    from .agent import generate_gemmi_port_with_agent

    oracle_dir = Path(oracle_dir)
    original_test = oracle_dir / "test" / "test.cc"
    if not original_test.exists():
        raise FileNotFoundError(
            f"MMDB test not found at {original_test} — run tooling.test first"
        )

    gemmi_subdir = oracle_dir / "gemmi"
    _conn = conn or connect()
    try:
        row = get_function(_conn, function_qname, sig_hash)
        if row is None or not row["source_code"]:
            raise RuntimeError(
                f"No source found in code_graph.db for {function_qname}"
            )
        # Compute dep build info once — reused by agent compile tool + verify.
        dep_includes = _dep_extra_includes(_conn, function_qname)
        dep_sources  = _dep_extra_sources(_conn, function_qname)
        gemmi_subdir.mkdir(parents=True, exist_ok=True)
        (gemmi_subdir / "original.cc").write_text(row["source_code"])
        blocks, trace = generate_gemmi_port_with_agent(
            _conn,
            original_function_src=row["source_code"],
            function_qname=function_qname,
            original_test_cc=original_test.read_text(),
            gemmi_subdir=gemmi_subdir,
            model=model,
            verbose=verbose,
        )
    finally:
        if conn is None:
            _conn.close()

    # Always persist the trace — it's the primary debugging artifact, and
    # without it a failed run leaves nothing behind but prompt.txt.
    gemmi_subdir.mkdir(parents=True, exist_ok=True)
    (gemmi_subdir / "agent_trace.txt").write_text(trace)

    if blocks is None:
        raise RuntimeError("Agent produced no usable port.")

    # Verify before committing files to disk — if compile or run fails the
    # gemmi/ dir won't contain function.hh + test.cc, so _is_complete stays
    # False and the next batch run will retry rather than skip.
    gemmi_subdir = oracle_dir / "gemmi"
    gemmi_subdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_hh   = tmp_path / "function.hh"
        tmp_test = tmp_path / "test.cc"
        tmp_cc   = tmp_path / "function.cc"
        tmp_bin  = tmp_path / "test"

        tmp_hh.write_text(blocks["function.hh"])
        tmp_test.write_text(blocks["test.cc"])
        has_cc = "function.cc" in blocks and blocks["function.cc"].strip()
        if has_cc:
            tmp_cc.write_text(blocks["function.cc"])

        ok, output = compile_gemmi(
            tmp_test, tmp_bin, tmp_cc if has_cc else None,
            dep_includes, dep_sources,
        )
        (gemmi_subdir / "compile.log").write_text(output)
        if not ok:
            raise RuntimeError(f"gemmi test compile failed:\n{output[:500]}")

        ok, output = run_gemmi_test_binary(tmp_bin)
        (gemmi_subdir / "run.log").write_text(output)
        if not ok:
            raise RuntimeError(f"gemmi test failed:\n{output[:500]}")

    # Tests passed — write final files and compile script (with dep flags).
    test_cc = _write_files(oracle_dir, blocks)
    write_compile_script(
        oracle_dir / "gemmi",
        has_function_cc=has_cc,
        extra_includes=dep_includes,
        extra_sources=dep_sources,
    )

    if commit:
        print(f"[gemmi] Committing port for {function_qname}...")
        _commit_conn = conn or connect()
        try:
            commit_gemmi_port(
                _commit_conn, function_qname, oracle_dir / "gemmi",
                sig_hash=sig_hash,
            )
        except Exception as e:
            print(f"[gemmi] Warning: git commit failed — {e}")
        finally:
            if conn is None:
                _commit_conn.close()

    return test_cc

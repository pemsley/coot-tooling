"""Compilation helpers for the combined gemmi port + test.

gemmi 0.7.x ships both a header tree and a shared library (libgemmi_cpp), so
the ported function can split across function.hh / function.cc if needed, and
we link against -lgemmi_cpp for symbols that live in the library.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from ..oracle.compile import (
    CXX, GEMMI_INCLUDE, BOOST_INCLUDE, GSL_INCLUDE, GLM_INCLUDE,
    CLIPPER_INCLUDE, AUTOBUILD_LIB, RDKIT_INCLUDE,
)

CLIPPER_LIBS = [
    "clipper-core", "clipper-ccp4", "clipper-cif", "clipper-cns",
    "clipper-contrib", "clipper-minimol", "clipper-mmdb", "clipper-phs",
]
from ..db import PROJECT_ROOT
from ..test.compile import GTEST_INCLUDE, GTEST_LIB_DIR

GEMMI_LIB_DIR  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/lib64"
GEMMI_LIB_NAME = "gemmi_cpp"

MAX_COMPILE_ATTEMPTS = 20


def make_gemmi_compile_cmd(
    test_cc: Path,
    output_bin: Path,
    function_cc: Path | None = None,
    extra_includes: list[Path] | None = None,
    extra_sources: list[Path] | None = None,
) -> str:
    """Compile test.cc (and optionally function.cc) linking against gemmi + gtest.

    extra_includes: additional -I dirs (used for ported coot:: dep headers).
    extra_sources:  additional .cc files to compile (ported coot:: dep function.cc).
    """
    include_dirs = [str(test_cc.parent), PROJECT_ROOT, GEMMI_INCLUDE,
                    BOOST_INCLUDE, GSL_INCLUDE, GLM_INCLUDE, GTEST_INCLUDE,
                    CLIPPER_INCLUDE, RDKIT_INCLUDE]
    for d in (extra_includes or []):
        include_dirs.append(str(d))
    includes = " ".join(f'-I"{i}"' for i in include_dirs)
    sources = f'"{test_cc.absolute()}"'
    if function_cc is not None and function_cc.exists():
        sources += f' "{function_cc.absolute()}"'
    for s in (extra_sources or []):
        if s.exists():
            sources += f' "{s.absolute()}"'
    clipper_libs = " ".join(f'-l{l}' for l in CLIPPER_LIBS)
    rdkit_libs = " ".join(f'-lRDKit{l}' for l in [
        "GraphMol", "SmilesParse", "FileParsers", "RDGeneral",
        "RDStreams", "RDGeometryLib", "SubstructMatch", "Depictor",
        "MolTransforms", "RDInchiLib",
    ])
    return (
        f'{CXX} -std=c++20 {sources} -o "{output_bin.absolute()}" '
        f'{includes} -pthread '
        f'-Wl,-rpath,{GEMMI_LIB_DIR} '
        f'-Wl,-rpath,{AUTOBUILD_LIB} '
        f'-L "{GEMMI_LIB_DIR}" -l{GEMMI_LIB_NAME} '
        f'-L "{AUTOBUILD_LIB}" {clipper_libs} {rdkit_libs} '
        f'-L "{GTEST_LIB_DIR}" -lgtest -lgtest_main -lm -no-pie'
    )


def compile_gemmi(
    test_cc: Path,
    output_bin: Path,
    function_cc: Path | None = None,
    extra_includes: list[Path] | None = None,
    extra_sources: list[Path] | None = None,
) -> tuple[bool, str]:
    cmd = make_gemmi_compile_cmd(
        test_cc, output_bin, function_cc, extra_includes, extra_sources,
    )
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=str(test_cc.parent), timeout=180)
    except subprocess.TimeoutExpired:
        return False, "[compile_gemmi] timed out after 180s"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def run_gemmi_test_binary(test_bin: Path, attempts: int = 2) -> tuple[bool, str]:
    """Retries once on timeout — the hang is non-deterministic."""
    cmd = ["stdbuf", "-oL", "-eL", str(test_bin.absolute())]
    cwd = str(test_bin.parent)
    last_out = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=cwd, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            last_out = f"[run_gemmi_test_binary] timed out after 20s (attempt {attempt}/{attempts})"
            continue
        try:
            (test_bin.parent / "run.exit").write_text(str(proc.returncode))
        except OSError:
            pass
        return proc.returncode == 0, (stdout + stderr).strip()
    return False, last_out


def write_compile_script(
    gemmi_subdir: Path,
    has_function_cc: bool,
    extra_includes: list[Path] | None = None,
    extra_sources: list[Path] | None = None,
) -> Path:
    test_cc     = gemmi_subdir / "test.cc"
    function_cc = gemmi_subdir / "function.cc" if has_function_cc else None
    test_bin    = gemmi_subdir / "test"
    cmd = make_gemmi_compile_cmd(
        test_cc, test_bin, function_cc, extra_includes, extra_sources,
    )
    script = gemmi_subdir / "compile.sh"
    script.write_text(f"#!/bin/sh\nset -ex\n{cmd}\n")
    script.chmod(0o755)
    return script

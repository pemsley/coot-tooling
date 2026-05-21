"""Compilation helpers for test.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..oracle.compile import (
    CXX, COOT_API_DIR, MMDB_API_NAME,
    AUTOBUILD_LIB, COOT_BUILD_DIR,
    GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE,
    MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE,
    RDKIT_INCLUDE,
)
from ..db import PROJECT_ROOT

AUTOBUILD = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal"
GTEST_INCLUDE = f"/lmb/home/jdialpuri/Development/coot-tooling/third-party/google-test/include"
GTEST_LIB_DIR = f"/lmb/home/jdialpuri/Development/coot-tooling/third-party/google-test/lib"

MAX_COMPILE_ATTEMPTS = 20


def make_test_compile_cmd(test_cc: Path, output_bin: Path) -> str:
    includes = " ".join(f'-I"{i}"' for i in [
        PROJECT_ROOT, GEMMI_INCLUDE, CLIPPER_INCLUDE, BOOST_INCLUDE,
        MMDB_INCLUDE, GSL_INCLUDE, PNG_INCLUDE, GLM_INCLUDE,
        RDKIT_INCLUDE, GTEST_INCLUDE,
    ])

    clipper_libraries = " ".join(f'-l{l}' for l in [
        "clipper-core", "clipper-ccp4", "clipper-cif", "clipper-cns",
        "clipper-contrib", "clipper-minimol", "clipper-mmdb", "clipper-phs",
    ])
    rdkit_libraries = " ".join(f'-lRDKit{l}' for l in [
        "GraphMol", "SmilesParse", "FileParsers", "RDGeneral",
        "RDStreams", "RDGeometryLib", "SubstructMatch", "Depictor",
        "MolTransforms", "RDInchiLib",
    ])

    return (
        f'{CXX} -std=c++20 -fno-access-control "{test_cc.absolute()}" -o "{output_bin.absolute()}" '
        f'{includes} '
        f'-Wl,-rpath,{AUTOBUILD_LIB} '
        f'-Wl,-rpath,{COOT_BUILD_DIR} '
        f'-L "{COOT_BUILD_DIR}" -lcootapi '
        f'-L "{AUTOBUILD_LIB}" {clipper_libraries} {rdkit_libraries} -l{MMDB_API_NAME} -lstdc++ '
        f'-L "{GTEST_LIB_DIR}" -lgtest -lgtest_main -lm -no-pie'
    )



def compile_test_cc(test_cc: Path, output_bin: Path) -> tuple[bool, str]:
    """Compile test_cc. Returns (success, compiler output)."""
    cmd = make_test_compile_cmd(test_cc, output_bin)

    script = output_bin.parent / "compile.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=str(test_cc.parent), timeout=180)
    except subprocess.TimeoutExpired:
        return False, "[compile_test_cc] timed out after 180s"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def _spawn_and_wait(cmd: list[str], cwd: str, timeout: int) -> tuple[int | None, str, str]:
    """Spawn cmd in its own process group, wait up to timeout. Returns
    (returncode, stdout, stderr); returncode is None on timeout."""
    import os
    import signal

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=cwd, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None, "", ""


def run_test_binary(test_bin: Path, attempts: int = 2) -> tuple[bool, str]:
    """Run a compiled test binary. Returns (all tests passed, output).

    Retries once on timeout — the hang is non-deterministic and a fresh
    invocation usually completes in milliseconds.
    """
    cmd = ["stdbuf", "-oL", "-eL", str(test_bin.absolute())]
    cwd = str(test_bin.parent)
    last_out = ""
    for attempt in range(1, attempts + 1):
        rc, stdout, stderr = _spawn_and_wait(cmd, cwd, timeout=20)
        if rc is None:
            last_out = f"[run_test_binary] timed out after 20s (attempt {attempt}/{attempts})"
            continue
        try:
            (test_bin.parent / "run.exit").write_text(str(rc))
        except OSError:
            pass
        return rc == 0, (stdout + stderr).strip()
    return False, last_out


def write_compile_script(test_subdir: Path) -> Path:
    """Write compile_test.sh into test_subdir and make it executable."""
    test_cc  = test_subdir / "test.cc"
    test_bin = test_subdir / "test"
    cmd = make_test_compile_cmd(test_cc, test_bin)
    script = test_subdir / "compile_test.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    return script

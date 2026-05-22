"""Compilation helpers for oracle.cc programs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..db import PROJECT_ROOT

# ── constants (mirrored from mmdb-refactor-ui/backend/config.py) ─────────────
CXX            = "clang++"
AUTOBUILD_LIB  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/lib"
COOT_BUILD_DIR = "/lmb/home/jdialpuri/Development/coot-dev/build-linux"
COOT_API_DIR   = AUTOBUILD_LIB  # kept for test/compile.py import
MMDB_API_NAME  = "mmdb2"        # kept for test/compile.py import
GEMMI_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
CLIPPER_INCLUDE = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
BOOST_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
MMDB_INCLUDE   = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
GSL_INCLUDE    = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
PNG_INCLUDE    = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
GLM_INCLUDE    = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
RDKIT_INCLUDE  = "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include/rdkit"


def make_compile_cmd(oracle_cc: Path, output_bin: Path) -> str:
    includes = [PROJECT_ROOT, GEMMI_INCLUDE, RDKIT_INCLUDE]
    includes = " ".join(f'-I"{i}"' for i in includes)

    clipper_libraries = " ".join(f'-l{l}' for l in [
        "clipper-core", "clipper-ccp4", "clipper-cif", "clipper-cns",
        "clipper-contrib", "clipper-minimol", "clipper-mmdb", "clipper-phs",
    ])
    rdkit_libraries = " ".join(f'-lRDKit{l}' for l in [
        "GraphMol", "SmilesParse", "FileParsers", "RDGeneral",
        "RDStreams", "RDGeometryLib", "SubstructMatch", "Depictor",
        "MolTransforms", "RDInchiLib",
    ])
    # gsl_libraries = "-lgsl -lgslcblas"

    return (
        f'{CXX} -std=c++20 -fno-access-control "{oracle_cc}" -o "{output_bin}" '
        f'{includes} '
        f'-Wl,-rpath,{AUTOBUILD_LIB} '
        f'-Wl,-rpath,{COOT_BUILD_DIR} '
        f'-L "{COOT_BUILD_DIR}" -lcootapi '
        f'-L "{AUTOBUILD_LIB}" {clipper_libraries} {rdkit_libraries} -lmmdb2 -lstdc++'
    )


def write_compile_script(out_dir: Path) -> Path:
    """Write compile.sh into out_dir and make it executable. Returns the path."""
    oracle_cc  = out_dir / "oracle.cc"
    output_bin = out_dir / "oracle"
    cmd = make_compile_cmd(oracle_cc, output_bin)

    script = out_dir / "compile.sh"
    script.write_text(f"#!/bin/sh\nset -e\n{cmd}\n")
    script.chmod(0o755)
    return script


def compile_oracle(out_dir: Path) -> tuple[bool, str]:
    """Run compile.sh in out_dir. Returns (success, output)."""
    script = out_dir / "compile.sh"
    if not script.exists():
        write_compile_script(out_dir)

    compile_log = out_dir / "compile.log"
    with open(compile_log, "w") as f:
        try:
            proc = subprocess.run(
                ["sh", str(script)],
                text=True,
                stdout=f,
                stderr=f,
                cwd=str(out_dir),
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            f.write("\n[compile_oracle] timed out after 180s\n")
            return False
    return proc.returncode == 0


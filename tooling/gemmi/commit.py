"""Commit a completed gemmi port into the coot source tree.

After a port passes verification, the generated files are copied into the
coot repo under a `gemmi/` subdirectory that mirrors the original function's
source location.  A git commit is created with a co-author trailer naming
the agent.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

from ..db import PROJECT_ROOT, get_function

COOT_REPO = Path(PROJECT_ROOT)
AGENT_COAUTHOR = "Co-authored-by: CootAgent <282556670+CootAgent@users.noreply.github.com>"


def _source_dir_relative(
    conn: sqlite3.Connection,
    function_qname: str,
    sig_hash: str | None = None,
) -> Path:
    """Return the directory of the function's source file, relative to COOT_REPO."""
    row = get_function(conn, function_qname, sig_hash)
    if row is None:
        raise RuntimeError(f"Function not found in DB: {function_qname}")
    abs_path = Path(row["file"])
    try:
        rel = abs_path.relative_to(COOT_REPO)
    except ValueError:
        raise RuntimeError(
            f"Source file {abs_path} is not inside the coot repo ({COOT_REPO})"
        )
    return rel.parent


def commit_gemmi_port(
    conn: sqlite3.Connection,
    function_qname: str,
    gemmi_subdir: Path,
    sig_hash: str | None = None,
) -> None:
    """Copy port files into the coot repo and create a git commit.

    Files written:
      <coot_repo>/<source_dir>/gemmi/<sanitized_qname>/function.hh
      <coot_repo>/<source_dir>/gemmi/<sanitized_qname>/function.cc   (if present)
      <coot_repo>/<source_dir>/gemmi/<sanitized_qname>/test.cc

    For overloaded functions, pass `sig_hash` so the destination dir is
    suffixed with `__<hash>` and overloads don't clobber each other.
    """
    src_rel_dir = _source_dir_relative(conn, function_qname, sig_hash)

    # Sanitise the qualified name the same way the batch runner does.
    sanitized = function_qname.replace("::", "__").replace(" ", "_")
    if sig_hash:
        sanitized = f"{sanitized}__{sig_hash}"
    dest_dir = COOT_REPO / src_rel_dir / "gemmi" / sanitized
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = ["function.hh", "test.cc"]
    if (gemmi_subdir / "function.cc").exists():
        files_to_copy.append("function.cc")

    copied: list[Path] = []
    for fname in files_to_copy:
        src = gemmi_subdir / fname
        dst = dest_dir / fname
        shutil.copy2(src, dst)
        copied.append(dst)

    # Stage the new files.
    rel_paths = [str(p.relative_to(COOT_REPO)) for p in copied]
    subprocess.run(
        ["git", "add", "--"] + rel_paths,
        cwd=str(COOT_REPO),
        check=True,
        timeout=60,
    )

    short_name = function_qname.rsplit("::", 1)[-1]
    commit_msg = (
        f"Add gemmi port: {function_qname}\n"
        f"\n"
        f"Ports {short_name} from MMDB to gemmi. Generated and verified\n"
        f"by the coot-tooling pipeline.\n"
        f"\n"
        f"{AGENT_COAUTHOR}"
    )
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=str(COOT_REPO),
        check=True,
        timeout=60,
    )

    subprocess.run(
        ["git", "push"],
        cwd=str(COOT_REPO),
        check=True,
        timeout=120,
    )

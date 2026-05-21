#!/usr/bin/env python3
"""Commit a completed gemmi port into the coot source tree.

Usage:
    python commit-port.py <function_qname>

Example:
    python commit-port.py "coot::molecule_t::get_number_of_atoms"
"""
import sys
from pathlib import Path

from tooling.db import connect
from tooling.gemmi.commit import commit_gemmi_port

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

function_qname = sys.argv[1]
sanitized = function_qname.replace("::", "__").replace(" ", "_")
gemmi_subdir = Path("generated-tests") / sanitized / "gemmi"

if not gemmi_subdir.exists():
    print(f"Error: {gemmi_subdir} does not exist", file=sys.stderr)
    sys.exit(1)

conn = connect()
try:
    commit_gemmi_port(conn, function_qname, gemmi_subdir)
    print(f"Committed: {function_qname}")
finally:
    conn.close()

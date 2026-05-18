"""Pre-compile static name verification — catches the dominant
'method does not exist' / 'name not declared' failures BEFORE burning a compile.

Walks each generated source file, extracts every qualified name in a tracked
namespace, and cross-references against:
  * code_graph.db (functions and types tables)
  * gemmi/.symbol_index.json (gemmi header symbol map)

Findings include near-match suggestions when the symbol index has a similar
name, mirroring `include_for_symbol`'s suggestion behavior.
"""
from __future__ import annotations

import re
import sqlite3

from ..db import get_type
from .cheat_lookup import _load_index

# Strip comments and string literals before token extraction.
_LINE_COMMENT_RE  = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_RE        = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)
_CHAR_RE          = re.compile(r"'(?:[^'\\]|\\.)'")

# Qualified identifier: at least one `::` plus a tail component.
_QNAME_RE = re.compile(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\b")

# Namespaces whose names we check.
_TRACKED_NAMESPACES = ("gemmi", "mmdb", "coot", "clipper", "ccp4")

# Tail names that occur on every container — too common to flag even when the
# parent type isn't indexed. Avoids ~all false positives on STL-like methods.
_TAIL_ALLOWLIST: frozenset[str] = frozenset({
    "size", "length", "data", "begin", "end", "cbegin", "cend", "rbegin",
    "rend", "clear", "empty", "swap", "front", "back", "at", "find",
    "insert", "erase", "push_back", "emplace_back", "emplace", "reserve",
    "resize", "count", "contains", "operator", "operator_eq", "operator_ne",
    "operator_lt", "operator_le", "operator_gt", "operator_ge",
})


def _strip_noncode(source: str) -> str:
    s = _BLOCK_COMMENT_RE.sub(" ", source)
    s = _LINE_COMMENT_RE.sub(" ", s)
    s = _STRING_RE.sub('""', s)
    s = _CHAR_RE.sub("''", s)
    return s


def extract_qualified_names(source: str) -> list[str]:
    clean = _strip_noncode(source)
    seen: set[str] = set()
    out: list[str] = []
    for m in _QNAME_RE.finditer(clean):
        name = m.group(1)
        root = name.split("::", 1)[0]
        if root not in _TRACKED_NAMESPACES:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _exists_in_gemmi_index(name: str) -> bool:
    if not name.startswith("gemmi::"):
        return False
    tail = name.rsplit("::", 1)[-1]
    if tail in _TAIL_ALLOWLIST:
        return True
    return tail in _load_index()


def _exists_in_db(conn: sqlite3.Connection, name: str) -> bool:
    # Exact match — type table is namespace-scoped already.
    row = conn.execute(
        "SELECT 1 FROM types WHERE qualified_name = ? LIMIT 1",
        (name,),
    ).fetchone()
    if row:
        return True
    row = conn.execute(
        "SELECT 1 FROM functions WHERE qualified_name = ? LIMIT 1",
        (name,),
    ).fetchone()
    if row:
        return True
    if "::" not in name:
        return False
    tail = name.rsplit("::", 1)[-1]
    if tail in _TAIL_ALLOWLIST:
        return True
    # Suffix match scoped to the query's ROOT namespace so we recover
    # inherited methods (`mmdb::Manager::ReadCoorFile` lives on
    # `mmdb::Root`) without smuggling wrong-namespace tails through
    # (`gemmi::Cell` must not pass just because `clipper::Cell` exists).
    root = name.split("::", 1)[0]
    row = conn.execute(
        "SELECT 1 FROM functions "
        "WHERE qualified_name LIKE ? AND qualified_name LIKE ? LIMIT 1",
        (f"{root}::%::{tail}", f"{root}::%"),
    ).fetchone()
    return row is not None


def _is_known(conn: sqlite3.Connection, name: str, _depth: int = 0) -> bool:
    if _exists_in_db(conn, name):
        return True
    if _exists_in_gemmi_index(name):
        return True
    # Enum values / static members: `gemmi::PolymerType::Unknown` — if the
    # parent type is known, accept the tail without flagging.
    if _depth == 0 and name.count("::") >= 2:
        parent = name.rsplit("::", 1)[0]
        return _is_known(conn, parent, _depth=1)
    return False


def _suggest_near_gemmi(name: str) -> list[str]:
    if not name.startswith("gemmi::"):
        return []
    tail = name.rsplit("::", 1)[-1]
    if len(tail) < 4:
        return []
    tail_lower = tail.lower()
    # Bias suggestions to the same case-style as the query: `Real3` is a type
    # so we want type-shaped candidates (capitalized), not lowercase field
    # names that happen to share two letters.
    same_case_first = tail[:1].isupper()
    # Rank candidates by (longest common prefix with query, shortest length-diff).
    scored: list[tuple[int, int, str]] = []
    for s in _load_index():
        if same_case_first != s[:1].isupper():
            continue
        # Require at least 3 chars of substring overlap to avoid 'A'/'r' noise.
        sl = s.lower()
        overlap = 0
        for k in range(min(len(sl), len(tail_lower)), 2, -1):
            if any(sl[i:i + k] in tail_lower for i in range(len(sl) - k + 1)):
                overlap = k
                break
        if overlap < 3:
            continue
        len_diff = abs(len(s) - len(tail))
        if len_diff > max(4, len(tail) // 2):
            continue
        scored.append((-overlap, len_diff, s))
    scored.sort()
    return [f"gemmi::{s}" for _, _, s in scored[:5]]


def check_names(source: str, conn: sqlite3.Connection) -> list[dict]:
    findings: list[dict] = []
    seen_pair: set[tuple[str, str]] = set()
    for name in extract_qualified_names(source):
        root = name.split("::", 1)[0]
        tail = name.rsplit("::", 1)[-1]
        # Allow freshly generated `_gemmi` ports — they don't exist in the DB
        # yet by definition.
        if tail.endswith("_gemmi"):
            continue
        if _is_known(conn, name):
            continue
        if (root, tail) in seen_pair:
            continue
        seen_pair.add((root, tail))
        findings.append({
            "name": name,
            "root": root,
            "suggestions": _suggest_near_gemmi(name),
        })
    return findings


def check_no_mmdb(source: str) -> list[str]:
    """Return all mmdb:: tokens in `source`. Used for function.hh/.cc where
    any mmdb reference is a port-correctness bug."""
    clean = _strip_noncode(source)
    out: list[str] = []
    seen: set[str] = set()
    for m in _QNAME_RE.finditer(clean):
        name = m.group(1)
        if not name.startswith("mmdb::"):
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    # `<mmdb...>` includes survive comment/string stripping; catch separately.
    for m in re.finditer(r"#\s*include\s*<(mmdb[^>]*)>", source):
        inc = f"<{m.group(1)}>"
        if inc not in seen:
            seen.add(inc)
            out.append(inc)
    return out


def name_check_findings(
    function_hh: str,
    test_cc: str,
    function_cc: str | None,
    conn: sqlite3.Connection,
) -> list[str]:
    out: list[str] = []
    for label, body in (
        ("function.hh", function_hh),
        ("function.cc", function_cc),
        ("test.cc",     test_cc),
    ):
        if not body:
            continue
        for f in check_names(body, conn):
            msg = f"{label}: `{f['name']}` — not declared in any indexed header"
            if f["suggestions"]:
                msg += (" (did you mean "
                        + ", ".join(f"`{s}`" for s in f["suggestions"][:3])
                        + "?)")
            out.append(msg)
        if label in ("function.hh", "function.cc"):
            for ref in check_no_mmdb(body):
                out.append(
                    f"{label}: MMDB reference `{ref}` — the port header/body "
                    "must be MMDB-free"
                )
    return out


def name_check_report(
    function_hh: str,
    test_cc: str,
    function_cc: str | None,
    conn: sqlite3.Connection,
) -> str:
    findings = name_check_findings(function_hh, test_cc, function_cc, conn)
    if not findings:
        return ""
    return (
        "Name check FAILED (this does not count against your compile budget). "
        "The names below are not declared in the code graph DB or the gemmi "
        "symbol index. Either you mistyped a name, or it does not exist. "
        "Resolve correct names with `find_header`, `lookup_type`, or "
        "`include_for_symbol` before retrying.\n\n"
        + "\n".join(f"  - {f}" for f in findings)
    )


def has_name_findings(
    function_hh: str,
    test_cc: str,
    function_cc: str | None,
    conn: sqlite3.Connection,
) -> bool:
    return bool(name_check_findings(function_hh, test_cc, function_cc, conn))

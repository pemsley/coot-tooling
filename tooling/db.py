"""
Database access layer — thin wrappers around code_graph.db queries.
All functions accept an open sqlite3.Connection with row_factory = sqlite3.Row.
"""
import hashlib
import re
import sqlite3
from pathlib import Path

DB_PATH      = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
PROJECT_ROOT = "/lmb/home/jdialpuri/Development/coot-dev/coot"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── signature normalization ───────────────────────────────────────────────────
#
# A qualified_name on its own is ambiguous for overloaded functions. We derive
# a stable "normalized signature" string from display_name (the libclang-supplied
# `ret-type name(params)` form) and hash it to a short suffix used to disambiguate
# output directories and DB lookups.
#
# Normalization rules — designed to coalesce spelling/whitespace/namespace
# differences that libclang produces for the SAME overload (e.g. seeing it
# from a header vs. a definition):
#   • strip the return type (everything before the parameter list);
#   • strip the trailing parameter name from each parameter;
#   • strip leading namespace prefixes from each token ('coot::Cartesian' →
#     'Cartesian'); libclang varies on whether it qualifies types depending on
#     which translation unit it parsed;
#   • normalize whitespace around '&', '*';
#   • preserve a trailing 'const' qualifier on member functions.


def _split_params(s: str) -> list[str]:
    """Split a comma-separated parameter list, respecting <> and () nesting."""
    out: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[start:i])
            start = i + 1
    last = s[start:].strip()
    if last:
        out.append(last)
    return out


def _normalize_param(param: str) -> str:
    p = param.strip()
    if not p:
        return p
    # Drop default value if any: 'int n = 5' → 'int n'.
    p = re.sub(r"\s*=\s*[^,]+$", "", p)
    # Strip leading namespace prefixes from every type token.
    p = re.sub(r"\b([a-zA-Z_]\w*::)+", "", p)
    # Drop the trailing parameter name. The name is the last bare identifier
    # that's not glued to a '&' or '*'. Only strip when preceded by whitespace
    # (otherwise we'd eat e.g. 'int' for a nameless 'int' param).
    p = re.sub(r"(\s+)[a-zA-Z_]\w*\s*$", "", p)
    # Normalize whitespace around pointer/reference qualifiers.
    p = re.sub(r"\s*([*&])\s*", r" \1 ", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def _normalize_signature(display_name: str | None) -> str:
    """Return a canonical parameter-list string for an overload.

    Two display_names that describe the same overload (modulo param names and
    namespace spelling) produce the same output. Used as the input to
    `sig_hash` so directories and DB rows can be matched per-overload.

    Falls back to the trimmed input when no parameter list is detectable.
    """
    if not display_name:
        return ""
    s = display_name
    # Find the outermost ( ) — guard against template angle brackets and
    # parens inside default args by tracking depth.
    depth = 0
    open_paren = -1
    close_paren = -1
    for i, c in enumerate(s):
        if c == "<":
            depth += 1
        elif c == ">" and depth > 0:
            depth -= 1
        elif c == "(" and depth == 0:
            open_paren = i
            break
    if open_paren == -1:
        return s.strip()
    depth = 0
    for i in range(open_paren, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                close_paren = i
                break
    if close_paren == -1:
        return s.strip()

    params_str = s[open_paren + 1: close_paren]
    tail = s[close_paren + 1:].strip()
    const_suffix = " const" if re.search(r"\bconst\b", tail) else ""

    params = _split_params(params_str)
    normalized = [_normalize_param(p) for p in params]
    return f"({', '.join(normalized)}){const_suffix}"


def sig_hash(display_name: str | None) -> str | None:
    """Return a short hex hash of the normalized signature, or None if empty.

    None means "no signature info available" — used as a sentinel for the
    non-overloaded code path so existing output directories don't change.
    """
    norm = _normalize_signature(display_name)
    if not norm:
        return None
    return hashlib.sha1(norm.encode()).hexdigest()[:6]


def get_function_overloads(
    conn: sqlite3.Connection, qname: str,
) -> list[sqlite3.Row]:
    """Return one row per distinct overload of `qname`, definition rows preferred.

    Each row carries (id, qualified_name, display_name, source_code, comment,
    access, file). Decl rows are kept only when no definition exists for the
    same normalized signature.
    """
    rows = conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.source_code, f.comment,
               f.access, f.is_definition, fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.qualified_name = ?
        ORDER BY f.is_definition DESC, f.line_start
    """, (qname,)).fetchall()
    by_sig: dict[str, sqlite3.Row] = {}
    for r in rows:
        norm = _normalize_signature(r["display_name"])
        # First row for a given normalized sig wins — and because we sort
        # by is_definition DESC, that's the definition when one exists.
        if norm not in by_sig:
            by_sig[norm] = r
    return list(by_sig.values())


def get_function(
    conn: sqlite3.Connection,
    qname: str,
    sig_hash_: str | None = None,
) -> sqlite3.Row | None:
    """Look up a function by qualified name, optionally pinned to an overload.

    When `sig_hash_` is None, returns whichever row sqlite picks first
    (definition preferred) — fine for non-overloaded names but ambiguous when
    multiple signatures share the qname.

    When `sig_hash_` is set, scans all rows for `qname` and returns the one
    whose normalized display_name hashes to that value.
    """
    if sig_hash_ is not None:
        for r in get_function_overloads(conn, qname):
            if sig_hash(r["display_name"]) == sig_hash_:
                return r
        # The sig_hash didn't match any overload of this exact qname. Fall
        # through to the unpinned lookup so callers that pass a stale hash
        # still get *something* back rather than None.

    row = conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.source_code, f.comment,
               f.access, fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.qualified_name = ?
        ORDER BY f.is_definition DESC
        LIMIT 1
    """, (qname,)).fetchone()
    if row:
        return row
    # Fall back to suffix match (e.g. "rdkit_mol" → "coot::rdkit_mol")
    short = qname.rsplit("::", 1)[-1]
    return conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.source_code, f.comment,
               f.access, fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.qualified_name LIKE ?
        ORDER BY f.is_definition DESC
        LIMIT 1
    """, (f"%::{short}",)).fetchone()


def get_containing_class(conn: sqlite3.Connection, qname: str) -> sqlite3.Row | None:
    if "::" not in qname:
        return None
    parent = qname.rsplit("::", 1)[0]
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name = ?
        LIMIT 1
    """, (parent,)).fetchone()


def get_used_types(conn: sqlite3.Connection, function_id: int) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM uses_type u
        JOIN types t  ON t.qualified_name = u.type_qualified_name
        JOIN files fi ON fi.id = t.file_id
        WHERE u.function_id = ?
    """, (function_id,)).fetchall()


def get_called_qnames(conn: sqlite3.Connection, function_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT callee_qualified_name FROM calls WHERE caller_id = ?",
        (function_id,)
    ).fetchall()
    return [r[0] for r in rows]


def get_type(conn: sqlite3.Connection, type_qname: str) -> sqlite3.Row | None:
    """Look up a type by exact or suffix-matched qualified name."""
    row = conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name = ?
        LIMIT 1
    """, (type_qname,)).fetchone()
    if row:
        return row
    # Fall back to matching on the last component (e.g. "Residue" → "mmdb::Residue")
    short = type_qname.rsplit("::", 1)[-1]
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name LIKE ?
        LIMIT 1
    """, (f"%::{short}",)).fetchone()


def get_types_matching(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    """Return every type matching `name` — either the exact qualified name or,
    for bare names with no "::", every row whose qualified_name ends in "::name".

    Used to detect ambiguity (e.g. 'Residue' matches both mmdb::Residue and
    gemmi::Residue) so the agent can be asked to qualify its lookup.
    """
    if "::" in name:
        return conn.execute("""
            SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
            FROM types t JOIN files fi ON fi.id = t.file_id
            WHERE t.qualified_name = ?
        """, (name,)).fetchall()
    return conn.execute("""
        SELECT t.qualified_name, t.kind, t.summary, fi.path AS file
        FROM types t JOIN files fi ON fi.id = t.file_id
        WHERE t.qualified_name = ? OR t.qualified_name LIKE ?
        ORDER BY t.qualified_name
    """, (name, f"%::{name}")).fetchall()


def get_type_methods(conn: sqlite3.Connection, type_qname: str) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT display_name, comment
        FROM functions
        WHERE qualified_name LIKE ?
          AND kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE')
        ORDER BY line_start
    """, (f"{type_qname}::%",)).fetchall()


def get_class_functions(
    conn: sqlite3.Connection,
    class_qname: str,
    mmdb_only: bool = False,
) -> list[tuple[str, str | None]]:
    """Return (qualified_name, sig_hash) for every overload of every method.

    Each overload appears as its own entry. `sig_hash` is None when the method
    is not overloaded (so existing single-overload output dirs stay unchanged);
    otherwise it's a short hex disambiguator derived from the parameter list.

    If `mmdb_only` is True, only methods that use at least one `mmdb::` type
    are returned — applied to the underlying rows, so a class method with
    multiple overloads where only one touches MMDB returns only that overload.
    """
    if mmdb_only:
        rows = conn.execute("""
            SELECT DISTINCT f.qualified_name, f.display_name, f.line_start
            FROM functions f
            JOIN uses_type u ON u.function_id = f.id
            WHERE f.qualified_name LIKE ?
              AND f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
              AND u.type_qualified_name LIKE 'mmdb::%'
            ORDER BY f.line_start
        """, (f"{class_qname}::%",)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT qualified_name, display_name, line_start
            FROM functions
            WHERE qualified_name LIKE ?
              AND kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
            ORDER BY line_start
        """, (f"{class_qname}::%",)).fetchall()
    return _per_overload_entries(rows)


def _per_overload_entries(
    rows: list[sqlite3.Row],
) -> list[tuple[str, str | None]]:
    """Collapse decl/def variants of the same overload, then attach sig_hash.

    The same (qname, normalized_sig) pair may appear in `rows` multiple times
    when libclang has both a declaration and a definition row. We keep only
    one per pair, in first-seen order. Once collapsed, we mark a qname's
    entries with sig_hash only if the qname has >1 distinct overload — names
    that have a single overload keep `None` so legacy output dirs are
    preserved.
    """
    seen: set[tuple[str, str]] = set()
    by_qname: dict[str, list[str]] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        qn = r["qualified_name"]
        norm = _normalize_signature(r["display_name"])
        key = (qn, norm)
        if key in seen:
            continue
        seen.add(key)
        by_qname.setdefault(qn, []).append(norm)
        order.append(key)
    out: list[tuple[str, str | None]] = []
    for qn, norm in order:
        if len(by_qname[qn]) <= 1:
            out.append((qn, None))
        else:
            out.append((qn, hashlib.sha1(norm.encode()).hexdigest()[:6]))
    return out


def get_class_methods_with_access(
    conn: sqlite3.Connection,
    class_qname: str,
) -> list[tuple[str, str | None]]:
    """Return [(qualified_name, access)] for every method in a class.

    Access is one of 'public', 'private', 'protected', or None when libclang
    couldn't determine it. The result preserves declaration order. When the
    same qualified_name appears multiple times (decl + definition + override),
    the most-restrictive non-null access wins so the agent always sees the
    visibility that actually matters at the call site.
    """
    rows = conn.execute("""
        SELECT qualified_name, access
        FROM functions
        WHERE qualified_name LIKE ?
          AND kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR',
                       'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
        ORDER BY line_start
    """, (f"{class_qname}::%",)).fetchall()

    # Coalesce duplicates, keeping the most-restrictive seen access.
    rank = {"public": 0, "protected": 1, "private": 2, None: -1}
    seen: dict[str, str | None] = {}
    order: list[str] = []
    for qn, acc in rows:
        if qn not in seen:
            seen[qn] = acc
            order.append(qn)
        else:
            if rank.get(acc, -1) > rank.get(seen[qn], -1):
                seen[qn] = acc
    return [(qn, seen[qn]) for qn in order]


def get_constructor_callers(
    conn: sqlite3.Connection,
    type_qname: str,
    limit: int = 5,
) -> list[sqlite3.Row]:
    """Return callers of type_qname's constructor, shortest source first."""
    short = type_qname.rsplit("::", 1)[-1]
    ctor_qname = f"{type_qname}::{short}"
    return conn.execute("""
        SELECT DISTINCT f.qualified_name, f.display_name, f.source_code, f.comment,
               fi.path AS file
        FROM calls c
        JOIN functions f  ON f.id = c.caller_id
        JOIN files fi     ON fi.id = f.file_id
        WHERE c.callee_qualified_name = ?
          AND f.is_definition = 1
          AND f.source_code IS NOT NULL
          AND f.source_code != ''
        ORDER BY LENGTH(f.source_code) ASC
        LIMIT ?
    """, (ctor_qname, limit)).fetchall()


def get_file_functions(
    conn: sqlite3.Connection,
    file_path: str,
    mmdb_only: bool = False,
) -> list[tuple[str, str | None]]:
    """Return (qualified_name, sig_hash) for every overload defined in a source file.

    file_path may be an absolute path or a suffix of the stored path
    (e.g. "src/coot/molecule.cc" will match the full stored path).
    If mmdb_only is True, only return overloads that use at least one
    mmdb:: type. See `get_class_functions` for the sig_hash convention.
    """
    if mmdb_only:
        rows = conn.execute("""
            SELECT DISTINCT f.qualified_name, f.display_name, f.line_start
            FROM functions f
            JOIN files fi ON fi.id = f.file_id
            JOIN uses_type u ON u.function_id = f.id
            WHERE (fi.path = ? OR fi.path LIKE ?)
              AND f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR',
                             'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
              AND f.is_definition = 1
              AND u.type_qualified_name LIKE 'mmdb::%'
            ORDER BY f.line_start
        """, (file_path, f"%/{file_path}")).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT f.qualified_name, f.display_name, f.line_start
            FROM functions f
            JOIN files fi ON fi.id = f.file_id
            WHERE (fi.path = ? OR fi.path LIKE ?)
              AND f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR',
                             'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
              AND f.is_definition = 1
            ORDER BY f.line_start
        """, (file_path, f"%/{file_path}")).fetchall()
    return _per_overload_entries(rows)


def expand_with_callee_deps(
    conn: sqlite3.Connection,
    seeds: list[tuple[str, str | None]],
    mmdb_only: bool = False,
) -> list[tuple[str, str | None]]:
    """BFS-expand seeds to include every transitive callee that
    exists as a function row in the DB (i.e. is portable in-codebase).

    The call edge table only carries callee qualified names, not signatures.
    So when a callee qname has multiple overloads, every overload is pulled
    in — we can't tell statically which overload was meant, and porting all
    of them keeps downstream gemmi compilation safe.

    If `mmdb_only` is True, transitive callees are filtered to overloads
    that use at least one mmdb:: type. The original seeds are always
    retained regardless of the filter.
    """
    if not seeds:
        return []
    seen_qnames: set[str] = {q for q, _ in seeds}
    result: dict[tuple[str, str | None], None] = {s: None for s in seeds}
    frontier: list[str] = list(seen_qnames)
    while frontier:
        placeholders = ",".join("?" * len(frontier))
        if mmdb_only:
            rows = conn.execute(f"""
                SELECT DISTINCT c.callee_qualified_name
                FROM calls c
                JOIN functions caller ON caller.id = c.caller_id
                JOIN functions callee ON callee.qualified_name = c.callee_qualified_name
                JOIN uses_type u ON u.function_id = callee.id
                WHERE caller.qualified_name IN ({placeholders})
                  AND u.type_qualified_name LIKE 'mmdb::%'
            """, frontier).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT DISTINCT c.callee_qualified_name
                FROM calls c
                JOIN functions caller ON caller.id = c.caller_id
                JOIN functions callee ON callee.qualified_name = c.callee_qualified_name
                WHERE caller.qualified_name IN ({placeholders})
            """, frontier).fetchall()
        new_qnames: list[str] = []
        for r in rows:
            qn = r[0]
            if qn in seen_qnames:
                continue
            seen_qnames.add(qn)
            new_qnames.append(qn)
            overloads = get_function_overloads(conn, qn)
            if len(overloads) <= 1:
                # No ambiguity — preserve the legacy unsuffixed output dir.
                result[(qn, None)] = None
            else:
                for overload in overloads:
                    result[(qn, sig_hash(overload["display_name"]))] = None
        frontier = new_qnames
    return sorted(result.keys(), key=lambda t: (t[0], t[1] or ""))


def get_internal_call_deps(
    conn: sqlite3.Connection,
    entries: list[tuple[str, str | None]],
) -> dict[tuple[str, str | None], set[tuple[str, str | None]]]:
    """For every (qname, sig_hash) in the batch, return the entries it calls
    that are ALSO in the batch. Self-calls (direct recursion) are ignored.

    Call edges in the DB are recorded by callee qname only — we don't know
    which overload was actually called. So if a caller invokes `foo` and
    multiple `foo` overloads are in the batch, the caller depends on ALL of
    them. This is conservative but correct: when batching topologically the
    caller waits until every plausible callee port has been generated.

    Every input entry is present as a key, possibly with an empty set.
    """
    if not entries:
        return {}
    qname_to_entries: dict[str, list[tuple[str, str | None]]] = {}
    for e in entries:
        qname_to_entries.setdefault(e[0], []).append(e)
    qnames = list(qname_to_entries.keys())
    placeholders = ",".join("?" * len(qnames))
    rows = conn.execute(f"""
        SELECT DISTINCT f.qualified_name, c.callee_qualified_name
        FROM calls c
        JOIN functions f ON f.id = c.caller_id
        WHERE f.qualified_name IN ({placeholders})
          AND c.callee_qualified_name IN ({placeholders})
    """, (*qnames, *qnames)).fetchall()
    deps: dict[tuple[str, str | None], set[tuple[str, str | None]]] = {
        e: set() for e in entries
    }
    for caller_qn, callee_qn in rows:
        if caller_qn == callee_qn:
            continue
        callee_entries = qname_to_entries.get(callee_qn, [])
        for caller_entry in qname_to_entries.get(caller_qn, []):
            for callee_entry in callee_entries:
                if callee_entry != caller_entry:
                    deps[caller_entry].add(callee_entry)
    return deps


def get_callers_with_source(
    conn: sqlite3.Connection,
    function_id: int,
    limit: int = 2,
) -> list[sqlite3.Row]:
    """Return callers that have source code, shortest first (easier to read)."""
    return conn.execute("""
        SELECT DISTINCT f.qualified_name, f.display_name, f.source_code, f.comment,
               fi.path AS file
        FROM calls c
        JOIN functions f  ON f.id = c.caller_id
        JOIN files fi     ON fi.id = f.file_id
        WHERE c.callee_qualified_name = (
            SELECT qualified_name FROM functions WHERE id = ?
        )
          AND c.caller_id != ?
          AND f.is_definition = 1
          AND f.source_code IS NOT NULL
          AND f.source_code != ''
        ORDER BY LENGTH(f.source_code) ASC
        LIMIT ?
    """, (function_id, function_id, limit)).fetchall()

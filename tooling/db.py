"""
Database access layer — thin wrappers around code_graph.db queries.
All functions accept an open sqlite3.Connection with row_factory = sqlite3.Row.
"""
import sqlite3
from pathlib import Path

DB_PATH      = Path(__file__).parent.parent / "ast-data" / "code_graph.db"
PROJECT_ROOT = "/lmb/home/jdialpuri/Development/coot-dev/coot"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_function(conn: sqlite3.Connection, qname: str) -> sqlite3.Row | None:
    return conn.execute("""
        SELECT f.id, f.qualified_name, f.display_name, f.source_code, f.comment,
               f.access, fi.path AS file
        FROM functions f JOIN files fi ON fi.id = f.file_id
        WHERE f.qualified_name = ?
        ORDER BY f.is_definition DESC
        LIMIT 1
    """, (qname,)).fetchone()


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
) -> list[str]:
    """Return qualified names of all methods in a class (definitions preferred, declarations as fallback).

    If mmdb_only is True, only return methods that use at least one mmdb:: type.
    """
    if mmdb_only:
        rows = conn.execute("""
            SELECT DISTINCT f.qualified_name
            FROM functions f
            JOIN uses_type u ON u.function_id = f.id
            WHERE f.qualified_name LIKE ?
              AND f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
              AND u.type_qualified_name LIKE 'mmdb::%'
            ORDER BY f.line_start
        """, (f"{class_qname}::%",)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT qualified_name
            FROM functions
            WHERE qualified_name LIKE ?
              AND kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR', 'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
            ORDER BY line_start
        """, (f"{class_qname}::%",)).fetchall()
    return [r[0] for r in rows]


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
) -> list[str]:
    """Return qualified names of all functions/methods defined in a source file.

    file_path may be an absolute path or a suffix of the stored path
    (e.g. "src/coot/molecule.cc" will match the full stored path).
    If mmdb_only is True, only return functions that use at least one mmdb:: type.
    """
    if mmdb_only:
        rows = conn.execute("""
            SELECT DISTINCT f.qualified_name
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
            SELECT DISTINCT f.qualified_name
            FROM functions f
            JOIN files fi ON fi.id = f.file_id
            WHERE (fi.path = ? OR fi.path LIKE ?)
              AND f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR',
                             'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
              AND f.is_definition = 1
            ORDER BY f.line_start
        """, (file_path, f"%/{file_path}")).fetchall()
    return [r[0] for r in rows]


def expand_with_callee_deps(
    conn: sqlite3.Connection,
    seed_qnames: list[str],
    mmdb_only: bool = False,
) -> list[str]:
    """BFS-expand seed_qnames to include every transitive callee that
    exists as a function row in the DB (i.e. is portable in-codebase).

    Use this when a target function calls helpers that haven't been ported
    yet — adding them to the batch and combining with topo ordering means
    they're converted callees-first, so the target's gemmi port can link
    against existing gemmi versions of its helpers.

    If `mmdb_only` is True, transitive callees are filtered to those that
    use at least one mmdb:: type. The original seeds are always retained
    regardless of the filter.
    """
    if not seed_qnames:
        return []
    result: set[str] = set(seed_qnames)
    frontier: list[str] = list(seed_qnames)
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
        new = [r[0] for r in rows if r[0] not in result]
        result.update(new)
        frontier = new
    return sorted(result)


def get_internal_call_deps(
    conn: sqlite3.Connection, qnames: list[str],
) -> dict[str, set[str]]:
    """For every qname in the batch, return the subset of qnames it calls
    that are ALSO in the batch. Self-calls (direct recursion) are ignored.

    Result shape: {caller_qname: {callee_qname, ...}}.
    Every input qname is present as a key, possibly with an empty set.
    """
    if not qnames:
        return {}
    placeholders = ",".join("?" * len(qnames))
    rows = conn.execute(f"""
        SELECT DISTINCT f.qualified_name, c.callee_qualified_name
        FROM calls c
        JOIN functions f ON f.id = c.caller_id
        WHERE f.qualified_name IN ({placeholders})
          AND c.callee_qualified_name IN ({placeholders})
    """, (*qnames, *qnames)).fetchall()
    deps: dict[str, set[str]] = {q: set() for q in qnames}
    for caller, callee in rows:
        if caller != callee:
            deps[caller].add(callee)
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
          AND f.is_definition = 1
          AND f.source_code IS NOT NULL
          AND f.source_code != ''
        ORDER BY LENGTH(f.source_code) ASC
        LIMIT ?
    """, (function_id, limit)).fetchall()

"""
Prompt rendering — assembles context from the DB into a prompt for the LLM.
"""
from __future__ import annotations

import sqlite3
import re
from pathlib import Path

from ..db import (
    get_function,
    get_containing_class,
    get_used_types,
    get_called_qnames,
    get_type,
    get_type_methods,
    get_callers_with_source,
    get_constructor_callers,
    PROJECT_ROOT,
)


_ACCESS_LABELS = ("public:", "protected:", "private:")


def caller_class_fields(conn: sqlite3.Connection, caller_qname: str) -> str | None:
    """Return a terse listing of field declarations for the class that contains
    caller_qname, grouped by access level.

    All fields (public, protected, private) are shown so the agent knows the
    names available — oracle.cc is compiled with -fno-access-control and can
    touch any of them. Methods are filtered out — only data members survive.
    Returns None if there are no fields.
    """
    cls = get_containing_class(conn, caller_qname)
    if not cls or not cls["summary"]:
        return None

    current_access = "public"
    buckets: dict[str, list[str]] = {"public": [], "protected": [], "private": []}
    for line in cls["summary"].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _ACCESS_LABELS:
            current_access = stripped[:-1]
            continue
        if stripped.startswith(("class ", "struct ", "};")):
            continue
        if "(" in stripped:
            continue  # method signature, not a field
        buckets[current_access].append(line)

    all_fields: list[str] = []
    for access in ("public", "protected", "private"):
        if buckets[access]:
            all_fields.append(f"  // {access}:")
            all_fields.extend(buckets[access])

    if not all_fields:
        return None
    return f"// Fields of {cls['qualified_name']}:\n" + "\n".join(all_fields)

# mmdb::Manager key methods are inherited from mmdb::Root / mmdb::CoorMngrRoot
# and don't appear on Manager directly, so we use a hardcoded setup pattern.
MMDB_MANAGER_SNIPPET = """\
// MMDB pointer typedef convention — every class Foo generates via DefineClass(Foo):
//   PFoo  = Foo*     PPFoo  = Foo**    RFoo  = Foo&    RPFoo  = Foo*&
// e.g. PAtom = Atom*,  PPAtom = Atom**,  PChain = Chain*,  PResidue = Residue*
//

// mmdb::Manager — load and navigate a PDB file:
//   mmdb::Manager *mol = new mmdb::Manager();
//   mol->ReadCoorFile("structure.pdb");          // load PDB
//   int nModels = mol->GetNumberOfModels();
//   mmdb::Model *model = mol->GetModel(1);        // 1-indexed
//   // selection API (alternative to manual traversal):
//   int selHnd = mol->NewSelection();
//   mol->Select(selHnd, mmdb::STYPE_RESIDUE, "//A/10", mmdb::SKEY_NEW);
//   mmdb::PPResidue selRes; int nSelRes;
//   mol->GetSelIndex(selHnd, selRes, nSelRes);
//   mol->DeleteSelection(selHnd);\
"""

# Hierarchy levels below Manager — methods are direct, so we pull from the DB.
# Each entry: (qualified_name, nav_methods_to_show)
MMDB_HIERARCHY: list[tuple[str, set[str]]] = [
    ("mmdb::Model", {
        "GetNumberOfChains", "GetChain", "GetModelNum",
    }),
    ("mmdb::Chain", {
        "GetNumberOfResidues", "GetResidue", "GetChainID",
    }),
    ("mmdb::Residue", {
        "GetNumberOfAtoms", "GetAtom",
        "GetResName", "GetSeqNum", "GetChainID", "GetInsCode",
    }),
    ("mmdb::Atom", {
        "GetAtomName", "GetElement", "GetChainID", "GetResName", "GetSeqNum",
    }),
]

_MMDB_ORDER = {"mmdb::Manager": 0}
_MMDB_ORDER.update({qname: i + 1 for i, (qname, _) in enumerate(MMDB_HIERARCHY)})

INCLUDE_ROOTS = [
    PROJECT_ROOT,
    "/lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include"
]

_TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test-data"


def make_oracle_instructions(pdb_path: str, pdb_note: str = "") -> str:
    """Build the ORACLE_INSTRUCTIONS block with the given PDB path.

    pdb_note is injected after the PDB line when the choice was ambiguous —
    it lists all available PDB files so the LLM can override the default.
    """
    pdb_line = f"       PDB: {pdb_path}"
    if pdb_note:
        pdb_line += (
            "\n       (or choose a more appropriate file from the list below —\n"
            f"        the selected path is only a default)\n{pdb_note}"
        )
    return f"""\
Write a complete, compilable C++ program (oracle.cc) that observes the inputs
and outputs of the function marked FUNCTION TO OBSERVE below.

Requirements:
  1. Be self-contained — hardcode the test file paths below, do not use argc/argv.
{pdb_line}
       MTZ: {_TEST_DATA_DIR}/example.mtz
  2. Load the structure using the hardcoded path.
  3. Navigate the structure to reach a valid receiver/input for the function.
  4. Call the function.
  5. Print every input value and every meaningful output value using this format:
       INPUT  <name>: <value>
       OUTPUT <name>: <value>

ACCESS RULES (oracle.cc is compiled with `-fno-access-control`):
  * You may call ANY method (public, protected, or private) and read or write
    ANY field on any object. C++ access checks are disabled for oracle.cc.
  * Prefer the public API where one exists — it is usually cleaner — but if
    a private method or member is the most direct way to set up or observe
    behaviour, just call it directly.
  * Private/protected members may be hidden from the terse type summaries
    below. Use `lookup_type` or `read_file` to inspect them when needed.

Use the EXAMPLE CALLERS to understand how the function is typically invoked and
what objects are needed. Only use types and methods shown in the context below.\
"""


# Backward-compat constant used by external code that imports ORACLE_INSTRUCTIONS.
ORACLE_INSTRUCTIONS = make_oracle_instructions(str(_TEST_DATA_DIR / "example.pdb"))


def _to_include(path: str) -> str:
    for root in INCLUDE_ROOTS:
        if path.startswith(root + "/"):
            return path[len(root) + 1:]
    return path


def _short_name(qname: str) -> str:
    return qname.rsplit("::", 1)[-1]


OVERRIDES_DIR = Path(__file__).parent / "overrides"


def _load_override(type_qname: str, pdb_path: str | None = None) -> str | None:
    """Return the contents of an override file for type_qname, or None.

    Files are named by replacing '::' with '__', e.g.:
      molecules_container_t       → overrides/molecules_container_t.cc
      mmdb::Residue               → overrides/mmdb__Residue.cc

    pdb_path: full path to the example PDB file to substitute for @PDB_PATH@.
    Falls back to the default example.pdb when None.
    """
    stem = type_qname.replace("::", "__")
    path = OVERRIDES_DIR / f"{stem}.cc"
    if not path.exists():
        return None
    resolved_pdb = pdb_path or str(_TEST_DATA_DIR / "example.pdb")
    return (
        path.read_text()
        .replace("@TEST_DATA_DIR@", str(_TEST_DATA_DIR))
        .replace("@PDB_PATH@", resolved_pdb)
    )


def _render_type(
    conn: sqlite3.Connection,
    type_qname: str,
    summary: str,
    called_methods: set[str] | None,
    compact: bool = False,
) -> str:
    """Render a type summary with inline method comments.

    compact=True (oracle mode): omit all fields; show only constructors and
    called methods so the prompt stays focused on what the oracle needs to call.

    Members inside `private:` / `protected:` sections are suppressed — oracle.cc
    is external code and cannot call them. The access labels themselves are
    preserved so the agent sees where the cut-off is.
    """
    method_rows = get_type_methods(conn, type_qname)
    comment_map = {r["display_name"]: r["comment"] or "" for r in method_rows}
    class_short  = _short_name(type_qname)

    out_lines: list[str] = []
    elided = 0
    current_access = "public"  # safest default; will be corrected by labels
    hidden_nonpublic = 0

    for line in summary.splitlines():
        stripped      = line.strip()
        candidate     = stripped.rstrip(";")

        # Access-specifier line → always emit, update state.
        if stripped in _ACCESS_LABELS:
            current_access = stripped[:-1]
            out_lines.append(line)
            continue

        # Detect method lines by signature syntax (parentheses) rather than
        # comment_map membership — methods not yet in the functions table would
        # otherwise be misidentified as fields and dropped in compact mode.
        is_method     = "(" in candidate
        is_structural = not stripped or stripped.startswith(("class ", "struct ", "};"))
        bare_name     = candidate.split("(")[0].strip()
        is_ctor       = bare_name == class_short

        # Suppress anything non-public — oracle.cc can't reach it.
        if current_access != "public" and not is_structural:
            hidden_nonpublic += 1
            continue

        if compact and not is_method and not is_structural:
            continue  # drop field lines

        # called_methods=None → show all methods (used for return types)
        # called_methods=set() → show only constructors (used for containing class)
        if is_method and called_methods is not None and bare_name not in called_methods and not is_ctor:
            elided += 1
            continue

        comment = comment_map.get(candidate, "")
        if is_method and comment:
            out_lines.append(f"{line}  // {comment}")
        else:
            out_lines.append(line)

    if out_lines and out_lines[-1].strip() == "};":
        if elided:
            out_lines.insert(-1, f"  // ... ({elided} more public methods)")
        if hidden_nonpublic:
            out_lines.insert(-1, f"  // ... ({hidden_nonpublic} non-public members hidden — oracle cannot access)")

    return "\n".join(out_lines)


def _mmdb_navigation_section(
    conn: sqlite3.Connection,
    involved_types: set[str],
    headers: dict[str, str],
) -> str | None:
    """Return a rendered MMDB hierarchy section, or None if MMDB is not involved.

    Always includes the Manager setup snippet, then renders DB-derived summaries
    for Model → Chain → Residue → Atom down to the deepest level needed.
    """
    all_mmdb = {"mmdb::Manager"} | {qname for qname, _ in MMDB_HIERARCHY}
    if not involved_types & all_mmdb:
        return None

    present = [qname for qname, _ in MMDB_HIERARCHY if qname in involved_types]
    cutoff_name = max(present, key=lambda q: _MMDB_ORDER[q]) if present else None

    lines = [MMDB_MANAGER_SNIPPET]

    for qname, nav_methods in MMDB_HIERARCHY:
        type_row = get_type(conn, qname)
        if type_row:
            inc = _to_include(type_row["file"])
            if inc not in headers:
                headers[inc] = f"MMDB hierarchy {qname}"
        lines.append(f"\n// {qname}")
        if type_row:
            lines.append(_render_type(conn, qname, type_row["summary"] or "", nav_methods, compact=True))
        if cutoff_name and qname == cutoff_name:
            break

    return "\n".join(lines)


def _extract_return_type(source_code: str, function_qname: str) -> str:
    """Parse the return type from function source, stripped of decorators."""
    fn_name = re.escape(function_qname.rsplit("::", 1)[-1])
    match = re.match(
        r'^([\w\s:<>*&,]+?)\s*\n\s*(?:[\w:]+::)+' + fn_name + r'\s*\(',
        source_code,
        re.DOTALL,
    )
    if not match:
        return ""
    raw = match.group(1)
    raw = re.sub(r'\b(const|virtual|static|inline|explicit|override)\b', '', raw)
    raw = re.sub(r'[*&]', '', raw)
    return raw.strip()


def build_oracle_prompt(
    conn: sqlite3.Connection,
    function_qname: str,
    pdb_file: str = "example.pdb",
    pdb_note: str = "",
) -> str | None:
    fn = get_function(conn, function_qname)
    if not fn:
        return None

    # Map  type_qname -> {bare method names}  called by this function
    called_by_type: dict[str, set[str]] = {}
    for qname in get_called_qnames(conn, fn["id"]):
        if "::" in qname:
            parent, method = qname.rsplit("::", 1)
            called_by_type.setdefault(parent, set()).add(method)

    headers: dict[str, str] = {}

    # Containing class
    containing_class = None
    cls = get_containing_class(conn, function_qname)
    if cls:
        containing_class = dict(cls)
        headers[_to_include(cls["file"])] = f"containing class {cls['qualified_name']}"

    # Types used in the function body
    used_types: list[dict] = []
    for t in get_used_types(conn, fn["id"]):
        used_types.append(dict(t))
        inc = _to_include(t["file"])
        if inc not in headers:
            headers[inc] = f"{t['kind']} {t['qualified_name']}"

    # Return type — show all its methods so the oracle can inspect the output
    return_type_row = None
    ret_type_name = _extract_return_type(fn["source_code"] or "", function_qname)
    if ret_type_name:
        return_type_row = get_type(conn, ret_type_name)
        if return_type_row:
            inc = _to_include(return_type_row["file"])
            if inc not in headers:
                headers[inc] = f"return type {return_type_row['qualified_name']}"

    # Callers (example usage)
    callers = get_callers_with_source(conn, fn["id"], limit=3)

    # ---- Assemble context block ----
    # Each code chunk is wrapped in its own ```cpp fence, with plain-text
    # markdown headings between them so section structure is visually
    # distinct from C++ content.
    parts: list[str] = []

    def section(heading: str, code: str, *, note: str | None = None) -> None:
        parts.append(f"## {heading}")
        if note:
            parts.append(note)
        parts.append(f"```cpp\n{code.rstrip()}\n```")

    # Includes
    section("Includes", "\n".join(f'#include "{inc}"' for inc in sorted(headers)))

    # Containing class
    if containing_class:
        section(
            f"Containing class: `{containing_class['qualified_name']}`",
            _render_type(
                conn,
                containing_class["qualified_name"],
                containing_class["summary"] or "",
                called_by_type.get(containing_class["qualified_name"], set()),
                compact=True,
            ),
        )

    # Containing class constructor callers — shows how to instantiate this class.
    # A hand-curated override file takes precedence over the automated DB lookup.
    if containing_class:
        cls_qname = containing_class["qualified_name"]
        full_pdb_path = str(_TEST_DATA_DIR / pdb_file)
        override = _load_override(cls_qname, pdb_path=full_pdb_path)
        if override:
            section(f"`{cls_qname}` construction (curated)", override)
        else:
            ctor_callers = get_constructor_callers(conn, cls_qname)
            if ctor_callers:
                parts.append(f"## `{cls_qname}` constructor callers")
                for ctor_caller in ctor_callers:
                    rel = ctor_caller["file"].replace(PROJECT_ROOT + "/", "")
                    parts.append(f"**{rel}**")
                    if ctor_caller["comment"]:
                        parts.append(f"_{ctor_caller['comment']}_")
                    parts.append(f"```cpp\n{ctor_caller['source_code'].rstrip()}\n```")

    # Types used in the function body
    if used_types:
        rendered_types: list[tuple[dict, str]] = []
        for t in used_types:
            if containing_class and t["qualified_name"] == containing_class["qualified_name"]:
                continue
            if return_type_row and t["qualified_name"] == return_type_row["qualified_name"]:
                continue
            rendered = _render_type(
                conn,
                t["qualified_name"],
                t["summary"] or "",
                called_by_type.get(t["qualified_name"], set()),
                compact=True,
            )
            rendered_types.append((t, rendered))

        if rendered_types:
            parts.append("## Types used in function")
            for t, rendered in rendered_types:
                parts.append(f"**[{t['kind']}] `{t['qualified_name']}`**")
                parts.append(f"```cpp\n{rendered.rstrip()}\n```")

    # Return type
    if return_type_row:
        section(
            f"Return type: `{return_type_row['qualified_name']}`",
            _render_type(
                conn,
                return_type_row["qualified_name"],
                return_type_row["summary"] or "",
                called_methods=None,   # show all methods — these are the oracle's output accessors
                compact=True,
            ),
        )

    # MMDB hierarchy — always show the traversal path when MMDB types are involved
    all_type_qnames: set[str] = {t["qualified_name"] for t in used_types}
    if containing_class:
        all_type_qnames.add(containing_class["qualified_name"])
    if return_type_row:
        all_type_qnames.add(return_type_row["qualified_name"])

    mmdb_section = _mmdb_navigation_section(conn, all_type_qnames, headers)
    if mmdb_section:
        section("MMDB navigation hierarchy", mmdb_section)

    # Example callers
    if callers:
        parts.append("## Example callers")
        parts.append("_Reference only — do NOT copy verbatim. See ACCESS RULES above._")
        target_class = containing_class["qualified_name"] if containing_class else None
        for i, caller in enumerate(callers, 1):
            rel = caller["file"].replace(PROJECT_ROOT + "/", "")
            caller_cls = get_containing_class(conn, caller["qualified_name"])
            caller_cls_qname = caller_cls["qualified_name"] if caller_cls else None
            in_class = (
                target_class is not None
                and caller_cls_qname == target_class
            )
            access_note = (
                "in-class member — has PRIVATE access, oracle does NOT"
                if in_class else
                "external caller — public API only"
            )

            parts.append(f"### Caller {i}/{len(callers)}: `{caller['qualified_name']}`")
            meta = [
                f"- **File:** `{rel}`",
            ]
            if caller_cls_qname:
                meta.append(f"- **Class:** `{caller_cls_qname}`")
            meta.append(f"- **Access:** {access_note}")
            if caller["comment"]:
                meta.append(f"- **Doc:** {caller['comment']}")
            parts.append("\n".join(meta))

            fields = caller_class_fields(conn, caller["qualified_name"])
            if fields:
                parts.append(f"```cpp\n{fields.rstrip()}\n```")

            parts.append(f"```cpp\n{caller['source_code'].rstrip()}\n```")

    # Function to observe
    parts.append("## Function to observe")
    if fn["comment"]:
        parts.append(f"_{fn['comment']}_")
    parts.append(
        "```cpp\n"
        + (fn["source_code"] or f"// (no source available) {fn['display_name']}").rstrip()
        + "\n```"
    )

    context_block = "\n\n".join(parts)

    instructions = make_oracle_instructions(str(_TEST_DATA_DIR / pdb_file), pdb_note)
    return f"{instructions}\n\n{context_block}\n"

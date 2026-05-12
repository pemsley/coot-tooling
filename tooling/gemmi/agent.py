"""Combined agentic port: function.hh (+ optional function.cc) + test.cc in one session.

The original MMDB function source and its MMDB-based test are both supplied.
The agent produces a gemmi equivalent of the function AND a gemmi version of
the test that exercises it — compiled and linked as a single unit so
signatures agree by construction.

Frozen: the RHS literal (expected value) of every multi-arg EXPECT_* / ASSERT_*
in the original test. LHS accessors may be rewritten when the type changes.
"""
from __future__ import annotations

import json
import re
import sqlite3
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from ..oracle.agent import (
    OLLAMA_CHAT_URL, TOOLS, _dispatch,
    _EXTENSION_TURNS, _MAX_EXTENSIONS, _EXTENSION_PROMPT,
    _tool_resolve_includes, _has_unresolved_includes,
    _tool_grep_codebase,
    _TraceWriter,
    _chat,
    _log_llm_timing,
    _is_degenerate_thinking,
    _has_compile_intent,
    NUDGE_EVERY_N_TURNS,
    NO_COMPILE_AFTER,
)

# Format-reminder nudge (injected every NUDGE_EVERY_N_TURNS turns) — keeps
# the output-format spec near the end of the context where attention is
# strongest. The gemmi port has the strictest format requirement of the
# three agents (two or three labelled fenced blocks), so a tighter reminder
# pays for itself.
_GEMMI_NUDGE = (
    "Reminder: when you stop calling tools, your final reply must be exactly "
    "two or three fenced code blocks, labelled:\n"
    "  ```cpp:function.hh\n  ...\n  ```\n"
    "  ```cpp:function.cc          (optional — only if needed)\n  ...\n  ```\n"
    "  ```cpp:test.cc\n  ...\n  ```\n"
    "Do not summarise. Do not narrate. If you have a working draft, call "
    "write_gemmi_file NOW for function.hh and then test.cc — writing "
    "test.cc auto-compiles and validates the port."
)

_GEMMI_NO_COMPILE_NUDGE = (
    "WARNING: you have not started writing files yet. "
    "STOP looking things up. Call write_gemmi_file with your best draft "
    "of function.hh, then write_gemmi_file with test.cc — writing test.cc "
    "triggers an automatic compile and you will get real compiler errors "
    "to act on. Further file reads and lookups are now LESS useful than "
    "a failed compile. Action over analysis."
)
from ..oracle.compile import GEMMI_INCLUDE
from ..oracle.notes import load_notes, render_notes_for_prompt
from .compile import (
    MAX_COMPILE_ATTEMPTS, compile_gemmi, run_gemmi_test_binary,
    write_compile_script,
)
from .lint import gemmi_lint, lint_report
from .cheat_lookup import mmdb_to_gemmi, include_for_symbol
from ..oracle.generate import OUT_ROOT, sanitize_name

_GEMMI_NO_COMPILE_AFTER = 10

# Absolute paths to data files (pdb/cif/mtz/map/ent) inside the original test
# source. These fixtures are validated by the oracle stage and MUST carry over
# to the gemmi test verbatim — surfacing them in the prompt prevents the agent
# from spending tool calls on inspect_pdb / grep_codebase to "verify" them.
_FIXTURE_PATH_RE = re.compile(r'"(/[^"\s]+\.(?:pdb|cif|mmcif|ent|mtz|map))"')

# Patterns in the original MMDB function source that indicate the function
# reads parent-context from a Residue or Atom. gemmi types have no parent
# pointer, so these calls cannot be ported with a bare Residue*/Atom* — the
# signature must take gemmi::CRA. Detecting this deterministically at prompt
# time turns a recurring failure mode into a guardrail.
_PARENT_ACCESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"->\s*chain\b"),                # mmdb::Residue::chain (Chain*)
    re.compile(r"->\s*GetChainID\s*\("),        # any -> chain ID accessor
    re.compile(r"->\s*GetChain\s*\(\s*\)"),      # atom->GetChain() — zero-arg parent accessor
    re.compile(r"\batom\s*->\s*residue\b"),     # mmdb::Atom::residue
    re.compile(r"->\s*GetResidue\s*\(\s*\)"),   # zero-arg -> parent residue
]


def _needs_parent_context(mmdb_src: str) -> bool:
    """True if the MMDB source reads parent pointers (chain from residue, etc.).

    A True result means the gemmi port must take gemmi::CRA, not bare
    Residue*/Atom*, because gemmi has no parent back-pointers.
    """
    return any(p.search(mmdb_src) for p in _PARENT_ACCESS_PATTERNS)


# Drop callees that are noisy (operators, dtors) or definitionally outside the
# port surface (the function we're porting must not appear as its own dep).
def _coot_callees(conn: sqlite3.Connection, caller_qname: str) -> list[str]:
    """Return distinct `coot::*` callees of `caller_qname`, in DB order."""
    rows = conn.execute(
        "SELECT DISTINCT c.callee_qualified_name "
        "FROM calls c JOIN functions f ON c.caller_id = f.id "
        "WHERE f.qualified_name = ? "
        "  AND c.callee_qualified_name LIKE 'coot::%' "
        "  AND c.callee_qualified_name NOT LIKE 'coot::operator%' "
        "  AND c.callee_qualified_name NOT GLOB '*~*' "
        "  AND c.callee_qualified_name != ? ",
        (caller_qname, caller_qname),
    ).fetchall()
    return [r[0] for r in rows]


def _has_gemmi_port(callee_qname: str) -> bool:
    """True if a verified `_gemmi` port exists for `callee_qname`.

    `function.hh` is only written by `generate.py` after compile+run both
    pass, so its presence is the reliable success signal — see
    `tooling/gemmi/generate.py:_write_files`.
    """
    return (OUT_ROOT / sanitize_name(callee_qname)
            / "gemmi" / "function.hh").is_file()


def _gemmi_target_name(qname: str) -> str:
    """Mirror the rule used in `generate_gemmi_port_with_agent`."""
    parts = qname.rsplit("::", 1)
    if len(parts) == 2:
        return f"{parts[0]}::{parts[1]}_gemmi"
    return f"{qname}_gemmi"


def _all_gemmi_ports(conn: sqlite3.Connection) -> list[str]:
    """Return all coot qualified names that have a verified gemmi port.

    A port is "verified" when `<sanitized>/gemmi/function.hh` exists — see
    `_has_gemmi_port`. `sanitize_name` is lossy so we recover qnames by
    matching against the DB's distinct qualified_name set.
    """
    ported_dirs = {
        p.parent.parent.name
        for p in OUT_ROOT.glob("*/gemmi/function.hh")
    }
    if not ported_dirs:
        return []
    rows = conn.execute("SELECT DISTINCT qualified_name FROM functions").fetchall()
    by_sanitized: dict[str, str] = {}
    for (q,) in rows:
        by_sanitized.setdefault(sanitize_name(q), q)
    return sorted({by_sanitized[d] for d in ported_dirs if d in by_sanitized})


_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_NAMESPACE_OPEN_RE = re.compile(r"namespace\s+([A-Za-z_]\w*)\s*\{")
_GEMMI_DECL_RE = re.compile(r"([A-Za-z_]\w*_gemmi)\s*\(")


def _parse_gemmi_decls(header_path: Path) -> list[dict]:
    """Extract every `*_gemmi(...)` declaration from a generated header.

    Returns one dict per declaration with the real qualified name (built
    from the surrounding `namespace X {` stack) and the signature text
    (return type through closing `)`, whitespace normalised). We parse the
    source rather than predicting because the agent may have put the port
    in any namespace — `coot::molecule_t::foo` often ports to a free
    `coot::foo_gemmi`, and `_gemmi_target_name` is only a guess.
    """
    try:
        text = header_path.read_text()
    except OSError:
        return []
    text = _LINE_COMMENT_RE.sub("", text)
    text = _BLOCK_COMMENT_RE.sub("", text)

    results: list[dict] = []
    stack: list[tuple[int, str]] = []   # (brace depth when opened, ns name)
    depth = 0
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            while stack and stack[-1][0] == depth:
                stack.pop()
            depth -= 1
            i += 1
            continue
        m = _NAMESPACE_OPEN_RE.match(text, i)
        if m:
            depth += 1
            stack.append((depth, m.group(1)))
            i = m.end()
            continue
        m = _GEMMI_DECL_RE.match(text, i)
        if m:
            # Only treat this as a declaration if we're directly inside a
            # namespace (or at file scope) — NOT inside another function
            # body. Otherwise we'd pick up call-sites like
            # `gemmi::Residue* p = cid_to_residue_gemmi(...)` as decls.
            ns_depth = stack[-1][0] if stack else 0
            if depth != ns_depth:
                i = m.end()
                continue
            # Forward to the matching ')'.
            j, pdepth = m.end(), 1
            while j < n and pdepth > 0:
                if text[j] == "(":
                    pdepth += 1
                elif text[j] == ")":
                    pdepth -= 1
                j += 1
            # Backward to the start of the declaration.
            back = i
            while back > 0 and text[back - 1] not in ";{}":
                back -= 1
            signature = re.sub(r"\s+", " ", text[back:j].strip())
            ns = "::".join(name for _, name in stack)
            qname = f"{ns}::{m.group(1)}" if ns else m.group(1)
            results.append({"qname": qname, "signature": signature})
            i = j
            continue
        i += 1

    seen, out = set(), []
    for r in results:
        if r["qname"] in seen:
            continue
        seen.add(r["qname"])
        out.append(r)
    return out


def _port_entry(qname: str) -> dict:
    """Render a single port as the structured dict returned by the tool.

    `qname` here is the ORIGINAL MMDB function's qname (the directory's
    source). The actual ported callable(s) are parsed from function.hh.
    """
    header = OUT_ROOT / sanitize_name(qname) / "gemmi" / "function.hh"
    return {
        "source_qname": qname,
        "header": str(header),
        "decls": _parse_gemmi_decls(header),
    }


def _find_gemmi_ports(conn: sqlite3.Connection, name: str) -> list[dict]:
    """Match `name` (bare or qualified) against verified ports.

    Matches against the original MMDB qname (the dir's source) AND against
    the parsed gemmi declaration qnames (since the port may live in a
    different namespace than the original). Exact matches win; otherwise
    falls back to case-insensitive substring search.
    """
    name = (name or "").strip()
    if not name:
        return []
    sources = _all_gemmi_ports(conn)
    entries = [_port_entry(q) for q in sources]
    bare = name.rsplit("::", 1)[-1]

    def _matches_exact(e: dict) -> bool:
        if e["source_qname"] == name or e["source_qname"].rsplit("::", 1)[-1] == bare:
            return True
        for d in e["decls"]:
            dq = d["qname"]
            if dq == name or dq.rsplit("::", 1)[-1] == bare:
                return True
            # Also match against the bare name minus the _gemmi suffix.
            if dq.rsplit("::", 1)[-1] == f"{bare}_gemmi":
                return True
        return False

    exact = [e for e in entries if _matches_exact(e)]
    if exact:
        return exact
    lower = name.lower()

    def _matches_loose(e: dict) -> bool:
        if lower in e["source_qname"].lower():
            return True
        return any(lower in d["qname"].lower() for d in e["decls"])

    return [e for e in entries if _matches_loose(e)]


def _format_ports_for_tool(entries: list[dict]) -> str:
    """Human-readable rendering returned to the agent."""
    if not entries:
        return ("No verified gemmi port found. The callee may not be ported "
                "yet — translate inline using gemmi primitives.")
    out = []
    for e in entries:
        out.append(f"Port of `{e['source_qname']}`:")
        out.append(f'  #include "{e["header"]}"')
        if e["decls"]:
            out.append("  Call as:")
            for d in e["decls"]:
                out.append(f"    {d['qname']}")
                out.append(f"      signature: {d['signature']}")
        else:
            out.append("  (no `*_gemmi` declarations found in header)")
        out.append("")
    return "\n".join(out).rstrip()


def _transitive_ported_deps(
    conn: sqlite3.Connection, qname: str,
) -> list[str]:
    """BFS over the coot:: call graph, returning all transitively ported callees.

    Only entries with a verified `function.hh` on disk are included (i.e. the
    same condition as `_has_gemmi_port`). The start node itself is excluded.
    Order is BFS (closest deps first), deduped via a visited set.
    """
    visited: set[str] = {qname}
    queue: list[str] = [qname]
    result: list[str] = []
    while queue:
        current = queue.pop(0)
        for callee in _coot_callees(conn, current):
            if callee in visited:
                continue
            visited.add(callee)
            if _has_gemmi_port(callee):
                result.append(callee)
                queue.append(callee)
    return result


def _dep_extra_includes(conn: sqlite3.Connection, qname: str) -> list[Path]:
    """One gemmi/ dir per transitively ported dep — needed on -I so each
    dep's function.cc can resolve its own `#include "function.hh"`.
    """
    return [
        OUT_ROOT / sanitize_name(dep) / "gemmi"
        for dep in _transitive_ported_deps(conn, qname)
    ]


def _dep_extra_sources(conn: sqlite3.Connection, qname: str) -> list[Path]:
    """function.cc paths for all transitively ported deps that have one."""
    return [
        cc for dep in _transitive_ported_deps(conn, qname)
        if (cc := OUT_ROOT / sanitize_name(dep) / "gemmi" / "function.cc").is_file()
    ]


def _extract_test_fixtures(test_cc: str) -> list[str]:
    seen: list[str] = []
    for m in _FIXTURE_PATH_RE.finditer(test_cc):
        path = m.group(1)
        if path not in seen:
            seen.append(path)
    return seen


_ASSERTION_MACRO_RE = re.compile(r'\b(EXPECT_[A-Z]+|ASSERT_[A-Z]+)\s*\(')


def _split_top_level_args(arglist: str) -> list[str]:
    """Split a comma-separated argument list on top-level commas only,
    respecting (), [], {}, and "..." / '...' string/char literals.
    Angle brackets are NOT tracked because '<' / '>' are ambiguous with
    comparison and arrow (`->`) operators in C++ expressions."""
    args: list[str] = []
    depth_paren = depth_brack = depth_brace = 0
    in_str: str | None = None
    buf: list[str] = []
    i = 0
    while i < len(arglist):
        ch = arglist[i]
        if in_str:
            buf.append(ch)
            if ch == '\\' and i + 1 < len(arglist):
                buf.append(arglist[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
        elif ch == '(':
            depth_paren += 1; buf.append(ch)
        elif ch == ')':
            depth_paren -= 1; buf.append(ch)
        elif ch == '[':
            depth_brack += 1; buf.append(ch)
        elif ch == ']':
            depth_brack -= 1; buf.append(ch)
        elif ch == '{':
            depth_brace += 1; buf.append(ch)
        elif ch == '}':
            depth_brace -= 1; buf.append(ch)
        elif (ch == ',' and depth_paren == 0 and depth_brack == 0
              and depth_brace == 0):
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        args.append("".join(buf).strip())
    return args


def _extract_assertion_rhs_literals(src: str) -> list[tuple[str, str]]:
    """Return (macro_name, rhs_literal) pairs for each multi-arg EXPECT_*/ASSERT_*
    found in the source. Single-arg macros (EXPECT_TRUE, ASSERT_FALSE, ...) and
    no-op state-of-load asserts whose RHS is trivial (0, nullptr, NULL, true,
    false) are skipped — they convey no expected-value information, only sanity
    that loading succeeded, which has no MMDB↔gemmi equivalence."""
    trivial_rhs = {"0", "nullptr", "NULL", "true", "false"}
    out: list[tuple[str, str]] = []
    for m in _ASSERTION_MACRO_RE.finditer(src):
        macro = m.group(1)
        # Walk forward from the opening '(' to find the matching ')'.
        start = m.end()  # position after '('
        depth = 1
        in_str: str | None = None
        i = start
        while i < len(src) and depth > 0:
            ch = src[i]
            if in_str:
                if ch == '\\' and i + 1 < len(src):
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
            elif ch in ('"', "'"):
                in_str = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            i += 1
        if depth != 0:
            continue
        inner = src[start:i - 1]
        args = _split_top_level_args(inner)
        if len(args) < 2:
            continue
        rhs = args[-1]
        if rhs in trivial_rhs:
            continue
        out.append((macro, rhs))
    return out


def _normalize_rhs_literal(rhs: str) -> str:
    """Reduce an RHS literal to a comparable canonical form so semantically
    equivalent values match across the MMDB→gemmi port. Examples that collapse
    to the same key:
      `"H"`  ≡  `'H'`            (string vs char)
      `1.0`  ≡  `1.0f`  ≡  `1.`  (numeric suffix / trailing zeros)
      `42`   ≡  `42u`   ≡  `42L` (integer suffix)
      `  "x" ` ≡ `"x"`           (whitespace)
    Non-literal expressions fall through unchanged."""
    s = rhs.strip()
    if not s:
        return s
    # String / char literal: strip enclosing quotes (and any prefix like u8, L).
    m = re.match(r'^(?:u8|u|U|L)?(["\'])(.*)\1$', s, re.DOTALL)
    if m:
        return m.group(2)
    # Numeric: try to parse after stripping common suffixes.
    num = re.sub(r'[uUlLfF]+$', '', s)
    try:
        f = float(num)
        if f == int(f) and 'e' not in num.lower() and '.' not in num:
            return str(int(f))
        return f"{f:.6g}"
    except ValueError:
        pass
    return s


def _check_assertions_unchanged(original_test_cc: str, proposed_test_cc: str) -> str | None:
    """Return a warning message if any RHS literal frozen by the original test
    is absent from the proposed test (after semantic normalization), else None.

    Only the expected value (last arg) is checked — LHS accessors may be
    rewritten when the type changes (e.g. `res->GetSeqNum()` → `res->seqid.num.value`).
    The macro name itself is also free to change (e.g. EXPECT_STREQ → EXPECT_EQ
    when porting `const char*` → `std::string`).

    Matching is semantic: `"H"` matches `'H'`, `1.0` matches `1.0f`, `42`
    matches `42u`. If the RHS only differs cosmetically the check passes.
    Trivial RHS values (0, nullptr, true, false) are excluded — they tend to
    mark load-state checks with no gemmi analogue."""
    original = _extract_assertion_rhs_literals(original_test_cc)
    proposed_rhs_norm = {
        _normalize_rhs_literal(rhs)
        for _, rhs in _extract_assertion_rhs_literals(proposed_test_cc)
    }
    missing = sorted({
        rhs for _, rhs in original
        if _normalize_rhs_literal(rhs) not in proposed_rhs_norm
    })
    if not missing:
        return None
    sample = "\n".join(missing[:5])
    tail = f"\n  … and {len(missing) - 5} more" if len(missing) > 5 else ""
    return (
        "ASSERTION CHECK: your test.cc is missing the following expected RHS "
        "literal values from the original test:\n\n"
        + sample + tail + "\n\n"
        "The RHS should match the original — LHS accessors may change, but the "
        "expected value should be asserted somewhere. If your RHS is "
        "semantically equivalent (e.g. `\"H\"` vs `'H'`, `1.0` vs `1.0f`, an "
        "equivalent numeric expression, or a constant that evaluates to the "
        "same value) this is fine and you may proceed. Otherwise, fix your "
        "FUNCTION IMPLEMENTATION so it produces these values."
    )


GEMMI_CHEAT_SHEET = """\
## gemmi quick reference (verified against the installed headers)

Headers you will almost certainly need:
  #include <gemmi/model.hpp>      // Structure, Model, Chain, Residue, Atom, CRA
  #include <gemmi/pdb.hpp>        // read_pdb_file(path)
  #include <gemmi/mmread.hpp>     // read_structure(path)  — auto-detects format
  #include <gemmi/neighbor.hpp>   // NeighborSearch
  #include <gemmi/math.hpp>       // Vec3, Mat33, Transform
  #include <gemmi/unitcell.hpp>   // UnitCell, Position, Fractional

Loading a PDB — this is the only idiom that works:
  gemmi::Structure st = gemmi::read_pdb_file("/abs/path/to/file.pdb");
  // read_pdb_file DOES retain hydrogen atoms — element.is_hydrogen() returns
  // true for H atoms parsed from column 77-78 of the PDB ATOM record.
  // If your count is 0, the bug is in your code, not the file:
  //   ✅ atom.element.is_hydrogen()   ← correct
  //   ❌ atom.name == "H"             ← wrong (name is " H  ", "HA", "HB3"…)
  // grep_codebase does NOT search test-data/ — use read_file to inspect fixtures.

Traversal — every level is a std::vector, so you iterate, not GetXxx():
  for (gemmi::Model&   model   : st.models)
  for (gemmi::Chain&   chain   : model.chains)
  for (gemmi::Residue& residue : chain.residues)
  for (gemmi::Atom&    atom    : residue.atoms) { ... }

Most-used MMDB → gemmi accessors (use the mmdb_to_gemmi tool with method="<Name>" for any others):
  mol->GetModel(1)                → st.models[0]            // 0-indexed!
  chain->GetChainID()             → chain.name              // field, not method
  residue->GetResName()           → residue.name            // field
  residue->GetSeqNum()            → residue.seqid.num.value
  residue->GetInsCode()           → residue.seqid.icode     // char, not const char*
  // ⚠ INSERTION CODE MISMATCH: MMDB uses "" (empty string) for "no insertion
  //   code"; gemmi uses ' ' (space char), matching the raw PDB column.
  //   std::string(1, residue.seqid.icode) for a plain residue gives " ", NOT "".
  //   Always normalize before comparing:
  //     auto norm = [](const std::string& ic){ return ic.empty() ? std::string(" ") : ic; };
  //     if (norm(query_ic) == std::string(1, residue.seqid.icode)) { ... }
  atom->GetAtomName()             → atom.name               // std::string field
  atom->GetElementName()          → atom.element.name()     // "C","O" — unpadded
  atom->x, atom->y, atom->z       → atom.pos.x, atom.pos.y, atom.pos.z
  atom->occupancy                 → atom.occ
  atom->tempFactor                → atom.b_iso
  chain->GetNumberOfResidues()    → chain.residues.size()
  residue->GetNumberOfAtoms()     → residue.atoms.size()
  ... (~200 more — use the mmdb_to_gemmi tool with method="<MethodName>" for any MMDB API.
       The catalog covers Manager/Model/Chain/Residue/Atom getters,
       setters, and NO_EQUIVALENT explanations.)


NeighborSearch — distance queries against a Model:
  gemmi::NeighborSearch ns(st.models[0], st.cell, /*max_radius=*/5.0);
  ns.populate(/*include_h=*/false);   // MUST call before any find_*
  std::vector<gemmi::NeighborSearch::Mark*> hits =
      ns.find_atoms(atom.pos, /*alt=*/'\\0', /*min_dist=*/0.0, /*radius=*/4.0);
  for (auto* m : hits) {
      gemmi::CRA cra = m->to_cra(st.models[0]);   // gives chain/residue/atom
      // cra.chain, cra.residue, cra.atom are POINTERS (may be nullptr)
  }
  // find_atoms takes `const Position&`, NOT `Vec3`. Position derives from Vec3
  // but the conversion is EXPLICIT — passing a bare Vec3 fails to compile:
  //   ❌ ns.find_atoms(some_vec3, ...)              // no viable conversion
  //   ✅ ns.find_atoms(gemmi::Position(some_vec3), ...)
  //   ✅ ns.find_atoms(atom.pos, ...)               // atom.pos is already Position

ContactSearch — pairs of atoms within a radius:
  gemmi::ContactSearch cs(/*search_radius=*/4.0);
  cs.ignore = gemmi::ContactSearch::Ignore::SameResidue;  // or AdjacentResidues, SameChain, SameAsu, Nothing
  cs.setup_atomic_radii(1.0, 0.0);                        // optional, for VdW-aware filtering
  std::vector<gemmi::ContactSearch::Result> contacts = cs.find_contacts(ns);
  for (const auto& c : contacts) {
      // c.partner1, c.partner2 are CRA; c.dist_sq is double; c.image_idx is int
  }

CRA shape (gemmi/model.hpp):
  struct CRA { Chain* chain; Residue* residue; Atom* atom; };  // ALL POINTERS

There is NO top-level <gemmi.hpp>. There is NO atom.get_pos() or st.n_atoms().
To count atoms in a Structure use the free function (NOT a method on Structure):
  #include <gemmi/calculate.hpp>
  size_t total = gemmi::count_atom_sites(st);   // works for Structure/Model/Chain/Residue
  // ❌ st.n_atoms() / st.count_atom_sites()  — neither exists
  // ✅ Manual fallback: nested for-loops summing residue.atoms.size()
Vec3 operator* is component-wise; for dot product use v.dot(w); for squared
length use v.length_sq().

Extended type → header map (include the listed header for the named type):
  gemmi::Element                    → <gemmi/elem.hpp>
  gemmi::EntityType, PolymerType    → <gemmi/model.hpp>
  gemmi::ResidueId, SeqId           → <gemmi/model.hpp>
  gemmi::Fractional                 → <gemmi/unitcell.hpp>
  gemmi::Transform, Mat33           → <gemmi/math.hpp>
  gemmi::Span<T>                    → <gemmi/span.hpp>
  gemmi::SubChain                   → <gemmi/model.hpp>
  gemmi::SmallStructure             → <gemmi/small.hpp>
  gemmi::Mtz, MtzDataset            → <gemmi/mtz.hpp>
  gemmi::GridBase, Grid<T>          → <gemmi/grid.hpp>
  gemmi::Ccp4<T>                    → <gemmi/ccp4.hpp>
  gemmi::read_ccp4_map              → <gemmi/ccp4.hpp>
  gemmi::read_mtz_file              → <gemmi/mtz.hpp>
  gemmi::ChemComp, Restraints       → <gemmi/chemcomp.hpp>
  gemmi::MonLib                     → <gemmi/monlib.hpp>
  gemmi::Topo                       → <gemmi/topo.hpp>
  gemmi::DsspMaker                  → <gemmi/dssp.hpp>
  gemmi::PolyHeur                   → <gemmi/polyheur.hpp>

Everything above is in code_graph.db — lookup_type("gemmi::NeighborSearch"),
list_methods("gemmi::ContactSearch"), find_symbol("read_pdb_file"), etc. will
show the real API. Some signatures involving std::string / std::vector may
appear as "int" in the DB summary due to a libclang template-resolution quirk;
when in doubt, read the header directly with read_file or grep_codebase
(gemmi headers are at /lmb/home/jdialpuri/autobuild/Linux-hal.lmb.internal/include/gemmi/).
"""

# Anti-pattern catalog. Every entry in this list is on the list because it
# appeared in ≥3 verify-stage compile failures across the 180-function
# generated-tests/ corpus. The catalog is intentionally short and lives at the
# *end* of the system prompt where transformer attention is strongest.
GEMMI_ANTIPATTERNS = """\
## Names that DO NOT exist — never write these (data-derived from real fails)

  ❌ gemmi::Real3                  → ✅ gemmi::Vec3 (raw 3-vector) or gemmi::Position
  ❌ gemmi::vec3   (lowercase)     → ✅ gemmi::Vec3
  ❌ gemmi::Cell                   → ✅ gemmi::UnitCell
  ❌ gemmi::Element::C   (enum)    → ✅ gemmi::Element("C")
  ❌ gemmi::mat44                  → ✅ gemmi::Transform   (Mat33 + Vec3)

  ❌ Fractional.u / .v / .w        → ✅ Fractional.x / .y / .z   (inherits Vec3)
  ❌ ResidueId.num / .icode        → ✅ ResidueId.seqid.num.value / .seqid.icode
  ❌ query_ic == std::string(1, r.seqid.icode)  // breaks when query_ic="" and icode=' '
     → ✅ normalize: MMDB "" and gemmi ' ' both mean "no insertion code":
          auto norm=[](const std::string& s){ return s.empty() ? std::string(" ") : s; };
          norm(query_ic) == std::string(1, r.seqid.icode)
  ❌ Atom.alt_loc                  → ✅ Atom.altloc   (no underscore)
  ❌ Structure.links               → ✅ Structure.connections
  ❌ Structure.space_group         → ✅ Structure.spacegroup_hm
  ❌ Residue.add_atom(a)           → ✅ residue.atoms.push_back(a)
  ❌ st.setup_entities()           → ✅ gemmi::setup_entities(st)
                                       (free fn, <gemmi/polyheur.hpp>)
  ❌ residue.chain  /  atom.residue  → ✅ pass gemmi::CRA{&chain, &res, &atom}.
                                       gemmi::Residue and gemmi::Atom have NO
                                       parent pointer. CRA is the idiomatic
                                       carrier for "I need parent context":
                                         struct CRA { Chain*; Residue*; Atom* };
                                       — all pointers, all may be null. Build
                                       it during traversal:
                                         for (auto& chain : model.chains)
                                           for (auto& res : chain.residues)
                                             cra = {&chain, &res, nullptr};
                                       If your MMDB original reads `r->chain`
                                       or `atom->GetChainID()`, the gemmi
                                       signature MUST take CRA — bare
                                       Residue*/Atom* cannot recover parents.
  ❌ residue.subchain (as chain ID) → ✅ chain.name (from parent Chain).
                                       `subchain` is gemmi's auto-assigned
                                       polymer/entity label — for chain "A"
                                       it is typically "Axp" (polymer) or a
                                       similar synthetic ID. It is NOT the
                                       user-visible chain name. Comparing it
                                       to "A"/"B"/etc. will silently fail.

## Mandatory boilerplate for test.cc

```cpp
#include <gtest/gtest.h>
#include <gemmi/pdb.hpp>      // read_pdb_file
#include <gemmi/model.hpp>    // Structure, Model, Chain, Residue, Atom
#include "function.hh"

// ... TEST(...) blocks here ...

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

If you're unsure which header declares a name, use the **include_for_symbol**
tool (symbol="Foo") — authoritative answer, no grep needed. For an MMDB →
gemmi method mapping, use the **mmdb_to_gemmi** tool (method="GetSeqNum")
before grep'ing.\
"""

GEMMI_SYSTEM_PROMPT = f"""\
You are porting ONE C++ function from the MMDB API to the gemmi API AND
translating its Google Test, in the same session.

# Artifacts to produce
A. function.hh — `#pragma once`, declaration, `#include <gemmi/...>` deps.
   If the body is short, define it inline here.
B. function.cc — OPTIONAL. Only emit if the body is long or uses
   translation-unit-private helpers. Skip otherwise.
C. test.cc — the gemmi-translated Google Test. Must `#include "function.hh"`.

**The function MUST be defined somewhere — either inline in function.hh
or out-of-line in function.cc. A declaration-only function.hh with no
function.cc will fail to link (undefined reference). When in doubt,
inline the body in function.hh.**

# Naming (read this before anything else)
The user's task message states the exact target name — use it verbatim.
Rule: keep the original namespace, append `_gemmi` to the function name.

    Original:  coot::angle(...)
    Ported:    coot::angle_gemmi(...)        ✓
    Wrong:     gemmi::angle(...)             ✗  (no gemmi:: namespace)
    Wrong:     coot::gemmi::angle(...)       ✗  (no nested wrapper)
    Wrong:     coot::angle(...)              ✗  (missing _gemmi suffix)

# Test translation — what you may and may not change
You MAY rewrite the accessor on the LHS of an assertion when the type changes:

    EXPECT_EQ(res->GetSeqNum(), 42);          // MMDB form
    EXPECT_EQ(res->seqid.num.value, 42);      // gemmi form — same 42

You MAY NOT change the expected value or weaken the comparison operator.
`42` stays `42`. `EXPECT_EQ` does not become `EXPECT_NEAR`. The original
expected literals are the correctness oracle.

# Function port
- Semantics 1:1 — same output for the same input.
- The function signature must be exactly what test.cc calls. Design them together.
- function.hh and function.cc must not mention any `mmdb::` name or
  `<mmdb*>` include. If you typed one, you picked the wrong replacement
  — look it up. (coot:: and clipper:: are fine.)

# Reuse existing ports — do not re-derive what's already been ported
Before re-implementing any logic that the original delegates to a
`coot::` callee, check whether that callee has a verified gemmi port and
call it instead. A 2-line wrapper around `foo()` should port to a 2-line
wrapper around `foo_gemmi()` — not a 50-line reimplementation of foo.

- find_gemmi_port — does a verified `_gemmi` port exist for this name?
                    Pass bare ('cid_to_residue') or qualified. Returns the
                    target name + absolute header to #include.
- list_gemmi_ports — list all verified ports (optional substring filter).

Call find_gemmi_port for every `coot::` function the original invokes
BEFORE writing your body. If a port exists, your body should call it and
build on it rather than reach for MMDB primitives or gemmi primitives.

# Look up gemmi APIs, do not invent them
Use these tools BEFORE writing any gemmi name:
- mmdb_to_gemmi      — authoritative MMDB→gemmi mapping. Try this first.
- include_for_symbol — canonical #include for a known gemmi/gtest symbol.
- lookup_type, list_methods, find_header, find_symbol — DB lookups.
- grep_codebase      — search coot + gemmi headers for a usage pattern.

When lookup_type reports ambiguity, retry with the fully-qualified name.

# Link target
test.cc (+ function.cc if present) links against -lgemmi_cpp, -lgtest,
and the coot + clipper libraries. You MAY call into coot:: and clipper::
helpers where they make the port simpler — only MMDB is off-limits.

# Workflow — write_gemmi_file is the ONLY path
Build by writing files to disk with **write_gemmi_file**, in this order:
  1. function.cc (if needed)
  2. function.hh
  3. test.cc      ← writing this triggers an automatic compile + run

After an auto-compile, the response shows compile + gtest output. If
either fails, rewrite the affected file(s) with write_gemmi_file again.

If the compile log ends with "... more lines — use get_compile_errors",
call get_compile_errors before guessing at the fix.

Max {MAX_COMPILE_ATTEMPTS} compile attempts total.

# Available tools (use ONLY these — do not invent tool names)
- read_file          — read a C++ source or header file
- lookup_function    — get source + docs for a function by qualified name
- lookup_type        — get class/struct definition and method list
- list_methods       — list all method signatures in a class
- get_callers        — find functions that call a given function
- find_header        — resolve a type/function name to its header path
- resolve_includes   — check which #includes are needed for a code draft
- search_functions   — find functions by partial name
- grep_codebase      — text search across coot + gemmi headers
- get_base_classes   — list base classes of a type
- find_symbol        — locate a symbol in header files
- mmdb_to_gemmi      — authoritative MMDB→gemmi method mapping
- include_for_symbol — canonical #include for a gemmi/gtest symbol
- find_gemmi_port    — check whether a verified _gemmi port exists
- list_gemmi_ports   — list all verified gemmi ports
- write_gemmi_file   — write function.hh / function.cc / test.cc to disk
- run_gemmi_test     — re-run the last successfully compiled test binary
- get_compile_errors — return full (untruncated) compiler output

# When to stop looking up and start writing files
**Hard cap: at most 6 lookup tool calls before your first
write_gemmi_file.** Lookups include read_file, lookup_type, list_methods,
find_header, find_symbol, grep_codebase, mmdb_to_gemmi,
include_for_symbol, find_gemmi_port, list_gemmi_ports. After 6, you MUST
start writing files even if you
are uncertain — auto-compile errors are faster and more accurate than
further research. Re-reading the same file or re-looking-up the same
symbol counts and is wasted; act on what you already know.

Concretely, start writing as soon as you can name, for each MMDB call
you replaced: (a) the gemmi header that defines the replacement,
(b) the receiver type in gemmi, (c) the accessor or free function.

# Terminal condition
You are done when:
  1. The latest auto-compile after writing test.cc returned success.
  2. The gtest output in that response contains "All tests PASSED" (or
     equivalent — no FAIL lines).
  3. You have emitted a final response containing the three fenced
     blocks below in this exact order (omit function.cc if you did not
     write one).

# Final output format (ONE response)
```cpp:function.hh
... header contents ...
```

```cpp:function.cc
... only if you wrote a .cc; otherwise omit this block entirely ...
```

```cpp:test.cc
... test contents ...
```

# Reference: cheat sheet and antipatterns
{GEMMI_CHEAT_SHEET}

{GEMMI_ANTIPATTERNS}\
"""

_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "compile_gemmi",
        "description": (
            "Write the supplied sources to disk (function.hh, test.cc, and "
            "optionally function.cc) and compile them as one unit linked "
            f"against -lgemmi_cpp and -lgtest. Max {MAX_COMPILE_ATTEMPTS} attempts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "function_hh": {"type": "string",
                                "description": "Contents of function.hh"},
                "test_cc":     {"type": "string",
                                "description": "Contents of test.cc"},
                "function_cc": {"type": "string",
                                "description": "Optional contents of function.cc "
                                               "(omit for header-only)"},
            },
            "required": ["function_hh", "test_cc"],
        },
    },
}

_RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run_gemmi_test",
        "description": "Run the last compiled test binary and return GoogleTest output.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_GET_ERRORS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_compile_errors",
        "description": (
            "Return the FULL last compile log without truncation. "
            "Call this whenever an auto-compile response (triggered by "
            "write_gemmi_file test.cc) ends with '... (N more lines — use "
            "get_compile_errors)' — that footer means you are missing "
            "context that almost certainly contains the real error. "
            "Don't try to fix a compile failure from a truncated log."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_MMDB_TO_GEMMI_TOOL = {
    "type": "function",
    "function": {
        "name": "mmdb_to_gemmi",
        "description": (
            "Authoritative MMDB → gemmi method mapping, sourced from the "
            "curated cheat sheet. Pass an MMDB method like 'GetSeqNum' or "
            "'mmdb::Atom::GetAtomName'. Returns the gemmi equivalent expression "
            "(or NO_EQUIVALENT with a strategy note). Faster and more accurate "
            "than grep_codebase for known mappings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "MMDB method name, qualified or bare "
                                   "(e.g. 'GetSeqNum' or 'mmdb::Residue::GetSeqNum')",
                },
            },
            "required": ["method"],
        },
    },
}

_INCLUDE_FOR_SYMBOL_TOOL = {
    "type": "function",
    "function": {
        "name": "include_for_symbol",
        "description": (
            "Return the canonical #include directive that defines a known "
            "symbol. Works for gemmi/gtest names (e.g. 'read_pdb_file', "
            "'gemmi::Vec3', 'EXPECT_EQ') and for coot/clipper qualified "
            "names (e.g. 'coot::molecule_t', 'clipper::Coord_orth'). For "
            "coot symbols, pass the fully-qualified name. Use this BEFORE "
            "grep_codebase when you need a header for a known name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name, optionally namespaced",
                },
            },
            "required": ["symbol"],
        },
    },
}

_FIND_GEMMI_PORT_TOOL = {
    "type": "function",
    "function": {
        "name": "find_gemmi_port",
        "description": (
            "Check whether a coot function already has a verified gemmi port "
            "you can call directly. Pass a bare name ('cid_to_residue') or a "
            "qualified one ('coot::molecule_t::cid_to_residue'). Returns the "
            "target `_gemmi` name and the absolute header to #include. Call "
            "this for ANY coot function the original invokes, BEFORE writing "
            "the body — if a port exists, call it instead of re-deriving the "
            "callee's logic with gemmi primitives."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Function name, bare or qualified",
                },
            },
            "required": ["name"],
        },
    },
}

_LIST_GEMMI_PORTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_gemmi_ports",
        "description": (
            "List every coot function that already has a verified gemmi "
            "port. Optionally filter by a case-insensitive substring. Use "
            "this when you want to scan what's reusable before writing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contains": {
                    "type": "string",
                    "description": "Optional substring filter (case-insensitive)",
                },
            },
            "required": [],
        },
    },
}

_WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_gemmi_file",
        "description": (
            "Write one of the three gemmi port files to disk. "
            "Writing test.cc automatically compiles and runs the test — "
            "this is the ONLY way to compile in this stage. "
            "Order: function.cc first (if needed), then function.hh, then "
            "test.cc to trigger the build."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "enum": ["function.hh", "function.cc", "test.cc"],
                    "description": "Which file to write",
                },
                "contents": {
                    "type": "string",
                    "description": "Full contents of the file",
                },
            },
            "required": ["filename", "contents"],
        },
    },
}

# Filter inspect_pdb out — fixtures are injected into the user prompt verbatim,
# so the tool gets called wastefully (29× across the corpus despite a "do NOT
# call" instruction). Don't expose it during the gemmi stage.
_GEMMI_BASE_TOOLS = [
    t for t in TOOLS
    if (t.get("function", {}).get("name") or "") != "inspect_pdb"
]

GEMMI_TOOLS = _GEMMI_BASE_TOOLS + [
    _RUN_TOOL, _GET_ERRORS_TOOL, _WRITE_FILE_TOOL,
    _MMDB_TO_GEMMI_TOOL, _INCLUDE_FOR_SYMBOL_TOOL,
    _FIND_GEMMI_PORT_TOOL, _LIST_GEMMI_PORTS_TOOL,
]


# Truncation budgets for compile output. Calibrated against real logs:
# - Most failures fit in <3 KB / 80 lines and should ship in full.
# - g++ template-instantiation cascades blow past 1000 lines but the actual
#   diagnostic is concentrated at the first ~30 'error:' lines plus the very
#   end (linker errors live there). Show both ends and drop the middle.
_FULL_LOG_LINES = 80      # if log fits in this, send everything verbatim
_FULL_LOG_BYTES = 3000
_HEAD_LINES     = 50      # large logs: keep this many top lines
_TAIL_LINES     = 30      # ...and this many bottom lines
_ERROR_LINES    = 25      # plus this many 'error:' lines from the middle


_UNDEF_REF_RE = re.compile(
    r"undefined reference to `([^']+)'"
)


def _undefined_reference_directive(
    output: str, function_hh: str, function_cc: str | None,
) -> str | None:
    """If the linker error is 'undefined reference to coot::X_gemmi(...)',
    diagnose whether X is declared-only or just missing a body, and return
    a directive line for the agent. Returns None if not applicable.
    """
    matches = _UNDEF_REF_RE.findall(output)
    if not matches:
        return None

    # Only act on the user's target symbol(s) — anything in a coot/clipper
    # namespace ending in `_gemmi`. Std/gemmi names are real link errors
    # the model needs to fix differently (wrong library, wrong overload).
    porter_syms: list[str] = []
    for m in matches:
        head = m.split("(", 1)[0].strip()
        bare = head.rsplit("::", 1)[-1]
        if "_gemmi" in bare and head.startswith(("coot::", "clipper::")):
            porter_syms.append(head)
    if not porter_syms:
        return None

    target = porter_syms[0]
    bare = target.rsplit("::", 1)[-1]

    # Decide which fix to suggest. If function.cc wasn't written, the
    # body is missing entirely. If it was written but the symbol is still
    # undefined, the signature in .cc doesn't match the .hh declaration.
    if not function_cc:
        return (
            f"DIRECTIVE: linker can't find a body for `{target}`. You wrote "
            f"a declaration in function.hh but never defined it. Fix by EITHER:\n"
            f"  (a) adding an inline body to function.hh (best for short ports), OR\n"
            f"  (b) calling write_gemmi_file with filename=\"function.cc\" "
            f"containing the definition (signature must match function.hh exactly).\n"
            f"Then rewrite test.cc unchanged to re-trigger the compile."
        )
    if bare not in function_cc:
        return (
            f"DIRECTIVE: function.cc was written but does NOT define `{bare}` "
            f"(the linker can't find it). Check that the signature in "
            f"function.cc matches function.hh exactly — namespace, name, "
            f"argument types, and const/ref qualifiers must all match. "
            f"Rewrite function.cc with write_gemmi_file."
        )
    return (
        f"DIRECTIVE: linker can't resolve `{target}` even though function.cc "
        f"appears to mention it. The signature in function.cc almost "
        f"certainly differs from function.hh (a type, const, or ref mismatch "
        f"counts as a different symbol). Compare the two declarations "
        f"character-by-character and rewrite function.cc."
    )


def _summarise_compile_output(output: str) -> str:
    """Truncate compile output for the agent without losing the diagnostic.

    Behaviour:
      * Small logs: return verbatim.
      * Large logs: head + 'error:' lines from middle + tail, with an explicit
        pointer to get_compile_errors when content was elided.
    """
    if len(output) <= _FULL_LOG_BYTES:
        return output
    lines = output.splitlines()
    if len(lines) <= _FULL_LOG_LINES:
        return output

    head = lines[:_HEAD_LINES]
    tail = lines[-_TAIL_LINES:] if len(lines) > _HEAD_LINES + _TAIL_LINES else []
    middle_start = len(head)
    middle_end   = len(lines) - len(tail)
    middle = lines[middle_start:middle_end]

    # Pick out the most informative middle lines — anything containing 'error:'
    # or the line just before/after one (compile errors often span 2 lines).
    error_indices = [i for i, ln in enumerate(middle) if "error:" in ln]
    keep: set[int] = set()
    for i in error_indices:
        keep.update({i - 1, i, i + 1})
    keep = {i for i in keep if 0 <= i < len(middle)}
    middle_keep_lines: list[str] = []
    if keep:
        last_emitted = -2
        # Emit in order, with a separator when there's a gap.
        for i in sorted(keep)[:_ERROR_LINES * 3]:
            if i > last_emitted + 1:
                middle_keep_lines.append("...")
            middle_keep_lines.append(middle[i])
            last_emitted = i
        # Cap the kept-lines list size — 75 lines max from the middle.
        if len(middle_keep_lines) > 75:
            middle_keep_lines = middle_keep_lines[:75] + ["... (more middle elided)"]

    parts = list(head)
    if middle_keep_lines:
        parts.append(f"... ({middle_end - middle_start} middle lines — showing only "
                     "lines near 'error:' below)")
        parts.extend(middle_keep_lines)
    elif middle:
        parts.append(f"... ({len(middle)} middle lines elided — no 'error:' lines)")
    if tail:
        parts.append(f"--- last {len(tail)} lines ---")
        parts.extend(tail)
    parts.append(f"\n[full log is {len(lines)} lines / {len(output)} bytes — "
                 "call get_compile_errors for the complete output]")
    return "\n".join(parts)


def _make_tool_handlers(
    gemmi_subdir: Path,
    extra_includes: list[Path] | None = None,
    extra_sources: list[Path] | None = None,
    original_test_cc: str = "",
) -> tuple[callable, callable, callable]:
    attempts       = [0]
    last_binary    = [None]
    last_error_log = [None]
    assertion_warned = [False]

    def compile_handler(function_hh: str, test_cc: str,
                        function_cc: str | None = None) -> str:
        if attempts[0] >= MAX_COMPILE_ATTEMPTS:
            return (f"Compile limit reached ({MAX_COMPILE_ATTEMPTS}). "
                    "Output your best drafts as the final fenced blocks.")

        # Pre-flight include check across all three files — free fix cycle.
        sections: list[str] = []
        for label, body in (("function.hh", function_hh),
                            ("function.cc", function_cc),
                            ("test.cc",     test_cc)):
            if not body:
                continue
            report = _tool_resolve_includes(body)
            if _has_unresolved_includes(report):
                sections.append(f"--- {label} ---\n{report}")
        if sections:
            return (
                "Include check FAILED (this does not count against your "
                f"{MAX_COMPILE_ATTEMPTS} compile attempts). Fix the paths "
                "below and rewrite the affected file(s) with write_gemmi_file:\n"
                + "\n\n".join(sections)
            )

        # Pre-flight gemmi anti-pattern lint — also a free fix cycle. Catches
        # the recurring mistakes that would otherwise burn a compile attempt
        # (Real3, alt_loc, st.setup_entities(), missing <gtest/gtest.h>, etc).
        lint_sections: list[str] = []
        for label, body in (("function.hh", function_hh),
                            ("function.cc", function_cc),
                            ("test.cc",     test_cc)):
            if not body:
                continue
            findings = gemmi_lint(body)
            if findings:
                lint_sections.append(
                    f"--- {label} ---\n" + "\n".join(f"  - {f}" for f in findings)
                )
        if lint_sections:
            return (
                "Gemmi lint FAILED (this does not count against your "
                f"{MAX_COMPILE_ATTEMPTS} compile attempts). Fix the issues "
                "below and rewrite the affected file(s) with write_gemmi_file. "
                "These are anti-patterns the compiler would also reject:\n\n"
                + "\n\n".join(lint_sections)
            )

        attempts[0] += 1
        gemmi_subdir.mkdir(exist_ok=True)

        hh_path   = gemmi_subdir / "function.hh"
        test_path = gemmi_subdir / "test.cc"
        cc_path   = gemmi_subdir / "function.cc"
        hh_path.write_text(function_hh)
        test_path.write_text(test_cc)
        if function_cc:
            cc_path.write_text(function_cc)
            fn_cc_arg = cc_path
        else:
            if cc_path.exists():
                cc_path.unlink()
            fn_cc_arg = None

        write_compile_script(
            gemmi_subdir,
            has_function_cc=fn_cc_arg is not None,
            extra_includes=extra_includes,
            extra_sources=extra_sources,
        )

        test_bin = gemmi_subdir / "test_check"
        success, output = compile_gemmi(
            test_path, test_bin, fn_cc_arg, extra_includes, extra_sources,
        )

        compile_log = gemmi_subdir / "compile.log"
        compile_log.write_text(output)
        output = _summarise_compile_output(output)
        if success:
            last_binary[0] = test_bin
            run_ok, run_out = run_gemmi_test_binary(test_bin)
            (gemmi_subdir / "run.log").write_text(run_out)
            run_lines = run_out.splitlines()
            if len(run_lines) > 100:
                run_out = "\n".join(run_lines[:100]) + f"\n... ({len(run_lines) - 100} more lines)"
            status = "All tests PASSED." if run_ok else "Some tests FAILED — fix your FUNCTION IMPLEMENTATION (do NOT modify the EXPECT_*/ASSERT_* assertions) and recompile."
            return (
                f"Compilation succeeded (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}).\n"
                f"{status}\n{run_out}"
            )
        last_binary[0] = None
        last_error_log[0] = compile_log
        directive = _undefined_reference_directive(
            output, function_hh, function_cc,
        )
        prefix = (f"Compilation FAILED (attempt {attempts[0]}/"
                  f"{MAX_COMPILE_ATTEMPTS}):\n")
        if directive:
            prefix += directive + "\n\n"
        return prefix + output

    def run_handler() -> str:
        if last_binary[0] is None:
            return "No compiled binary — write test.cc with write_gemmi_file first to trigger a build."
        success, output = run_gemmi_test_binary(last_binary[0])
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        status = "All tests PASSED." if success else "Some tests FAILED."
        return f"{status}\n{output}"

    def get_errors_handler() -> str:
        if last_error_log[0] is None or not last_error_log[0].exists():
            return "No compile error log available."
        return last_error_log[0].read_text()

    def write_file_handler(filename: str, contents: str) -> str:
        allowed = {"function.hh", "function.cc", "test.cc"}
        if filename not in allowed:
            return f"ERROR: only {sorted(allowed)} may be written."
        gemmi_subdir.mkdir(exist_ok=True)
        (gemmi_subdir / filename).write_text(contents)

        hh   = gemmi_subdir / "function.hh"
        tc   = gemmi_subdir / "test.cc"
        cc   = gemmi_subdir / "function.cc"
        if hh.exists() and tc.exists():
            result = compile_handler(
                hh.read_text(),
                tc.read_text(),
                cc.read_text() if cc.exists() else None,
            )
            # Mid-flight assertion check: surface a missing-RHS warning ONCE
            # so the agent gets a chance to align literals before finalising.
            # Subsequent compiles stay silent — semantically equivalent RHS is
            # accepted, and we don't want to badger the model after one nudge.
            assertion_warning = ""
            if original_test_cc and not assertion_warned[0]:
                violation = _check_assertions_unchanged(
                    original_test_cc, tc.read_text()
                )
                if violation:
                    assertion_warned[0] = True
                    assertion_warning = (
                        "\n\n⚠ ASSERTION CHECK (one-time notice).\n"
                        + violation
                    )
            return (
                f"'{filename}' written. Compilation triggered automatically:"
                f"\n\n{result}{assertion_warning}"
            )
        missing = [f for f in ("function.hh", "test.cc") if not (gemmi_subdir / f).exists()]
        return f"'{filename}' written. Still waiting for: {', '.join(missing)}"

    def compiled_ok() -> bool:
        return last_binary[0] is not None

    return compile_handler, run_handler, get_errors_handler, write_file_handler, compiled_ok


_BLOCK_RE = re.compile(
    r"```(?:cpp|c\+\+)?(?::([^\n]+))?\n(.*?)```",
    re.DOTALL,
)


def _extract_drafts_from_thinking(thinking: str) -> dict[str, str]:
    """Salvage a draft from `thinking` when the stream aborted before
    `assistant_content` got a chance to be produced.

    Models often write multiple complete drafts inside a thinking block before
    going degenerate. We can't rely on labelled fences here (thinking blocks
    typically use bare ```cpp), so we fingerprint each block by content and
    keep the LAST one of each kind. Iteration is in document order, so later
    matches naturally overwrite earlier ones — i.e. the most refined draft wins.
    """
    found: dict[str, str] = {}
    for _label, body in _BLOCK_RE.findall(thinking):
        body = body.strip()
        if not body:
            continue
        # test.cc must be checked first: it usually `#include`s function.hh too,
        # so a naive function.cc check would steal it.
        if "TEST(" in body and "<gtest/gtest.h>" in body:
            found["test.cc"] = body
        elif "#pragma once" in body:
            found["function.hh"] = body
        elif '#include "function.hh"' in body and "{" in body and "}" in body:
            found["function.cc"] = body
    return found


def _extract_blocks(content: str) -> dict[str, str]:
    """Pull named fenced blocks out of the assistant's final message.

    Accepts labelled fences like ```cpp:function.hh or falls back to ordering
    (hh, cc, test) if labels are missing.
    """
    found: dict[str, str] = {}
    unlabelled: list[str] = []
    for label, body in _BLOCK_RE.findall(content):
        body = body.strip()
        label = (label or "").strip().lower()
        if "function.hh" in label or label.endswith(".hh"):
            found["function.hh"] = body
        elif "function.cc" in label or label.endswith("function.cc"):
            found["function.cc"] = body
        elif "test.cc" in label or label.endswith("test.cc"):
            found["test.cc"] = body
        else:
            unlabelled.append(body)
    if unlabelled:
        keys = ["function.hh", "function.cc", "test.cc"]
        for key in keys:
            if key not in found and unlabelled:
                found[key] = unlabelled.pop(0)
    return found


def generate_gemmi_port_with_agent(
    conn: sqlite3.Connection,
    original_function_src: str,
    function_qname: str,
    original_test_cc: str,
    gemmi_subdir: Path,
    model: str,
    verbose: bool = False,
) -> tuple[dict[str, str] | None, str]:
    """Return ({file_name: contents, ...}, trace_text) or (None, trace) on failure."""
    dep_includes = _dep_extra_includes(conn, function_qname)
    dep_sources  = _dep_extra_sources(conn, function_qname)
    compile_handler, run_handler, get_errors_handler, write_file_handler, compiled_ok = \
        _make_tool_handlers(gemmi_subdir, dep_includes, dep_sources, original_test_cc)

    assertion_dispatch_warned = [False]

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_gemmi":
            proposed_test_cc = args.get("test_cc", "")
            prefix = ""
            if (proposed_test_cc and original_test_cc
                    and not assertion_dispatch_warned[0]):
                violation = _check_assertions_unchanged(
                    original_test_cc, proposed_test_cc
                )
                if violation:
                    assertion_dispatch_warned[0] = True
                    trace_lines.append(
                        f"  → [assertion check — one-time notice]\n"
                        f"{textwrap.indent(violation, '    ')}\n"
                    )
                    prefix = (
                        "⚠ ASSERTION CHECK (one-time notice — compile still "
                        "proceeded):\n" + violation + "\n\n---\n\n"
                    )
            result = compile_handler(
                args.get("function_hh", ""),
                proposed_test_cc,
                args.get("function_cc") or None,
            )
            return prefix + result
        if name == "write_gemmi_file":
            return write_file_handler(args.get("filename", ""), args.get("contents", ""))
        if name == "run_gemmi_test":
            if not compiled_ok() and last_draft[0]:
                draft = last_draft[0]
                compile_msg = compile_handler(
                    draft.get("function.hh", ""),
                    draft.get("test.cc", ""),
                    draft.get("function.cc") or None,
                )
                if not compiled_ok():
                    return (
                        "run_gemmi_test: no compiled binary was available, so I "
                        "auto-compiled the most recent drafts you provided. "
                        "Compilation failed — fix the errors below and call "
                        "compile_gemmi with corrected code:\n" + compile_msg
                    )
                run_msg = run_handler()
                return (
                    "(auto-compiled latest drafts before running — "
                    f"{compile_msg.splitlines()[0] if compile_msg else 'compile ok'})\n"
                    + run_msg
                )
            return run_handler()
        if name == "get_compile_errors":
            return get_errors_handler()
        if name == "mmdb_to_gemmi":
            return mmdb_to_gemmi(args.get("method", ""))
        if name == "include_for_symbol":
            return include_for_symbol(args.get("symbol", ""))
        if name == "find_gemmi_port":
            return _format_ports_for_tool(
                _find_gemmi_ports(conn, args.get("name", ""))
            )
        if name == "list_gemmi_ports":
            substr = (args.get("contains") or "").lower()
            ports = _all_gemmi_ports(conn)
            if substr:
                ports = [q for q in ports if substr in q.lower()]
            if not ports:
                return "No gemmi ports match."
            return _format_ports_for_tool([_port_entry(q) for q in ports])
        # Widen grep to include the gemmi header tree — most API discovery
        # during a port needs to see gemmi usage, which isn't in PROJECT_ROOT.
        if name == "grep_codebase":
            return _tool_grep_codebase(
                args["pattern"],
                args.get("glob"),
                extra_roots=[GEMMI_INCLUDE, OUT_ROOT.parent / "test-data"],
            )
        return _dispatch(conn, name, args)

    parts: list[str] = []

    # Derive the target name: same namespace, function base name + _gemmi suffix.
    # e.g. "coot::molecule_t::angle" → target "coot::molecule_t::angle_gemmi"
    _ns_parts = function_qname.rsplit("::", 1)
    if len(_ns_parts) == 2:
        _target_name = f"{_ns_parts[0]}::{_ns_parts[1]}_gemmi"
    else:
        _target_name = f"{function_qname}_gemmi"

    parts.append("## Task")
    parts.append(
        f"Port `{function_qname}` to gemmi AND translate its MMDB test in one "
        f"pass. The ported function MUST be named **`{_target_name}`** — same "
        "namespace as the original, with `_gemmi` appended to the function "
        "name. Do NOT place it inside a `gemmi::` namespace. "
        "Design the function signature and the test's call site together. "
        "Use the tools to resolve gemmi types. Compile and run before finalising."
    )

    if _needs_parent_context(original_function_src):
        parts.append("## ⚠ Parent-context access detected in original MMDB source")
        parts.append(
            "The original function reads parent pointers (e.g. `r->chain`, "
            "`atom->GetChainID()`, `atom->residue`). **gemmi::Residue and "
            "gemmi::Atom have NO parent pointer.** Your ported signature "
            "MUST take `gemmi::CRA` (struct CRA { Chain*; Residue*; Atom* } "
            "— all pointers) instead of bare `gemmi::Residue*` / "
            "`gemmi::Atom*`. The test should construct CRAs by iterating\n\n"
            "```cpp\n"
            "for (auto& chain : model.chains)\n"
            "  for (auto& res : chain.residues)\n"
            "    gemmi::CRA cra{&chain, &res, nullptr};\n"
            "```\n\n"
            "Recover the chain name via `cra.chain->name` — do NOT use "
            "`residue->subchain` as a chain-name proxy. `subchain` is "
            "gemmi's auto-assigned polymer/entity label (e.g. \"Axp\" for "
            "chain \"A\"), not the user-visible chain ID."
        )

    # Coot dependencies — prefer ported variants over re-deriving from MMDB.
    # Without this nudge the agent re-implements every `coot::foo()` call from
    # scratch using gemmi primitives, which (a) wastes compile attempts and
    # (b) drifts in semantics from sibling ports.
    coot_callees = _coot_callees(conn, function_qname)
    if coot_callees:
        ported, unported = [], []
        for c in coot_callees:
            (ported if _has_gemmi_port(c) else unported).append(c)
        lines = [
            "## Coot dependencies — prefer the `_gemmi` variant when one exists",
            "",
            "Your function calls other `coot::` functions. For any with a verified "
            "gemmi port, **call the `_gemmi` variant directly** instead of "
            "re-deriving from MMDB primitives.",
            "",
        ]
        if ported:
            lines.append(
                "**Verified ports — include by ABSOLUTE PATH and call the "
                "real entry point shown below. The signature/namespace is "
                "parsed from the actual generated header, so use it "
                "verbatim — do NOT assume the port lives in the same "
                "namespace or has the same signature as the MMDB original. "
                "The build system automatically compiles any required dep "
                "`.cc` files.**"
            )
            for c in ported:
                hh = OUT_ROOT / sanitize_name(c) / "gemmi" / "function.hh"
                decls = _parse_gemmi_decls(hh)
                lines.append(f"  - `{c}`")
                lines.append(f'    `#include "{hh}"`')
                if decls:
                    for d in decls:
                        lines.append(f"    Call as: `{d['qname']}`")
                        lines.append(f"      `{d['signature']}`")
                else:
                    # Header exists but no `*_gemmi` decl parsed — surface
                    # the predicted target as a fallback hint.
                    lines.append(f"    Call as: `{_gemmi_target_name(c)}` "
                                 "(predicted; verify by reading the header)")
            lines.append("")
        if unported:
            lines.append(
                "**No port yet — translate inline using gemmi primitives "
                "(no `_gemmi` variant exists to call):**"
            )
            for c in unported:
                lines.append(f"  - `{c}`")
            lines.append("")
        parts.append("\n".join(lines).rstrip())

    fixtures = _extract_test_fixtures(original_test_cc)
    if fixtures:
        parts.append(
            "## Test fixtures (use these paths VERBATIM in the gemmi test — "
            "do NOT call grep_codebase to verify them)"
        )
        parts.append("\n".join(f"  - {p}" for p in fixtures))

    # Pre-fill the gtest preamble so the agent stops grep'ing for it. Real
    # data: 25+ wasted grep_codebase calls per failed run looking for gtest.h.
    parts.append("## Required test.cc preamble (paste verbatim)")
    parts.append(
        "```cpp\n"
        "#include <gtest/gtest.h>\n"
        "#include <gemmi/pdb.hpp>\n"
        "#include <gemmi/model.hpp>\n"
        '#include "function.hh"\n'
        "\n"
        "// ... TEST(...) blocks here ...\n"
        "\n"
        "int main(int argc, char** argv) {\n"
        "    ::testing::InitGoogleTest(&argc, argv);\n"
        "    return RUN_ALL_TESTS();\n"
        "}\n"
        "```"
    )

    parts.append("## Original MMDB function")
    parts.append(f"```cpp\n{original_function_src.rstrip()}\n```")

    parts.append("## Original MMDB test")
    parts.append("_FREEZE every `EXPECT_*` — keep the assertions identical._")
    parts.append(f"```cpp\n{original_test_cc.rstrip()}\n```")

    notes = load_notes(gemmi_subdir.parent / "oracle" / "notes.json")
    if notes:
        rendered = render_notes_for_prompt(notes, audience="gemmi")
        if rendered:
            parts.append("## Validated facts from oracle stage")
            parts.append(
                "_Carry these over where they still apply; treat port caveats "
                "as concrete design hints._"
            )
            parts.append(f"```\n{rendered.rstrip()}\n```")

    user_content = "\n\n".join(parts)

    messages: list[dict] = [
        {"role": "system", "content": GEMMI_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    gemmi_subdir.mkdir(parents=True, exist_ok=True)
    (gemmi_subdir / "prompt.txt").write_text(
        f"=== SYSTEM ===\n{GEMMI_SYSTEM_PROMPT}\n\n"
        f"=== USER ===\n{user_content}\n"
    )

    trace_lines = _TraceWriter(gemmi_subdir / "agent_trace.txt")
    trace_lines.append("=== GEMMI COMBINED AGENT TRACE ===\n")
    trace_lines.append(f"[user]\n{textwrap.indent(user_content, '  ')}\n")

    final_blocks: dict[str, str] | None = None
    last_draft: list[dict[str, str] | None] = [None]
    call_counts: dict[str, int] = {}
    tool_cache: dict[str, str] = {}
    REPEAT_LIMIT = 3
    NO_CACHE = {"compile_gemmi", "run_gemmi_test", "get_compile_errors", "leave_note",
                "write_gemmi_file"}
    no_compile_warned = [False]
    degen_recovered = [False]
    compile_intent_strikes = [0]
    non_compile_tool_calls = [0]
    no_tool_nudge_sent = [False]

    def _save_draft_from_compile(args: dict, compile_result: str) -> None:
        # Only save when the compile + test run actually succeeded — otherwise
        # the rescue fallback can resurrect a broken draft and the verify-stage
        # re-compile fails anyway. Detect success from the canonical phrases
        # produced by compile_handler.
        passed = ("All tests PASSED." in compile_result
                  and "Compilation succeeded" in compile_result)
        if not passed:
            return
        hh = args.get("function_hh") or ""
        tc = args.get("test_cc") or ""
        if len(hh) > 50 and len(tc) > 100 and "#include" in tc:
            draft = {
                "function.hh": hh,
                "test.cc": tc,
                **({"function.cc": args["function_cc"]}
                   if args.get("function_cc") else {}),
            }
            last_draft[0] = draft
            draft_dir = gemmi_subdir / "draft"
            draft_dir.mkdir(exist_ok=True)
            for fname, body in draft.items():
                (draft_dir / fname).write_text(body)

    def _run_tool_calls(tool_calls: list[dict]) -> list[dict]:
        results: list[dict] = []
        for call in tool_calls:
            fn_info = call.get("function", {})
            name    = fn_info.get("name", "")
            args    = fn_info.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            hash_args = {k: v for k, v in args.items()
                         if k not in ("function_hh", "function_cc", "test_cc")}
            key = f"{name}:{json.dumps(hash_args, sort_keys=True)}"
            call_counts[key] = call_counts.get(key, 0) + 1
            if name not in NO_CACHE and key in tool_cache:
                cached = tool_cache[key]
                note = (
                    "(cached — you already called this with the same arguments. "
                    "Use the answer below; do not re-query.)\n"
                )
                trace_lines.append(f"  → [cached × {call_counts[key]}] {name}({json.dumps(hash_args)})")
                trace_lines.append_call(f"[cached × {call_counts[key]}] {name}({json.dumps(hash_args)})")
                results.append({"role": "tool", "content": note + cached})
                continue
            if call_counts[key] > REPEAT_LIMIT and name not in ("compile_gemmi", "run_gemmi_test"):
                nudge = (
                    f"You have called {name} with these arguments {call_counts[key]} times. "
                    "Stop repeating — write function.hh and test.cc with write_gemmi_file now to trigger a real compile."
                )
                trace_lines.append(f"  → {name}(repeated — nudged)")
                trace_lines.append_call(f"[repeat-intercept] {name}({json.dumps(hash_args)})")
                results.append({"role": "tool", "content": nudge})
                continue
            if verbose:
                display = ({"function_hh": "...", "test_cc": "...",
                            "function_cc": "..." if args.get("function_cc") else None}
                           if name == "compile_gemmi" else args)
                print(f"  tool: {name}({display})")
            result_text = dispatch(name, args)
            # Save draft AFTER compile so we know whether it actually passed.
            # Saving before would let a failed-compile draft become the rescue
            # fallback, which then fails again at verify time.
            if name == "compile_gemmi":
                _save_draft_from_compile(args, result_text)
            if name in ("compile_gemmi", "write_gemmi_file", "run_gemmi_test"):
                non_compile_tool_calls[0] = 0
            else:
                non_compile_tool_calls[0] += 1
            result_lines = result_text.splitlines()
            if len(result_lines) > 150:
                result_text = ("\n".join(result_lines[:150])
                               + f"\n... ({len(result_lines) - 150} more lines)")
            short = json.dumps(args) if name != 'compile_gemmi' else '{...}'
            trace_lines.append(f"  → {name}({short})")
            trace_lines.append(textwrap.indent(result_text, "      ") + "\n")
            trace_lines.append_call(f"{name}({short})")
            if name not in NO_CACHE:
                tool_cache[key] = result_text
            results.append({"role": "tool", "content": result_text})
        return results

    def _is_usable(blocks: dict[str, str] | None) -> bool:
        if not blocks:
            return False
        return ("function.hh" in blocks and "test.cc" in blocks
                and "#include" in blocks["test.cc"])

    def _progress(label: str, tool_calls: list) -> None:
        if tool_calls:
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
            print(f"\r  [gemmi] {label} → {names}", flush=True)
        else:
            print(f"\r  [gemmi] {label} → done (final answer)", flush=True)

    for turn in range(50):
        print(f"  [gemmi] turn {turn + 1}/50 ...", end="", flush=True)
        data = _chat(messages, model, GEMMI_TOOLS)
        _log_llm_timing(data, stage="gemmi", turn=turn + 1, verbose=verbose, trace_lines=trace_lines)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking",  "") or ""
        assistant_content = msg.get("content",   "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                         "tool_calls": tool_calls})
        if thinking:
            trace_lines.append(f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n")

        # Degenerate-thinking guard: if this turn's thinking is pathologically
        # repetitive, force a compile with the last draft and continue with a
        # strict recovery nudge. On a second offence, abort.
        degen, diag = _is_degenerate_thinking(thinking)
        if degen:
            if degen_recovered[0]:
                print(f"\r  [gemmi] turn {turn + 1}/50 → DEGENERATE — aborting", flush=True)
                trace_lines.append(
                    f"[agent] {diag} — second degeneracy, aborting.\n"
                )
                break
            print(f"\r  [gemmi] turn {turn + 1}/50 → DEGENERATE — forcing compile", flush=True)
            degen_recovered[0] = True
            trace_lines.append(f"[agent] {diag} — forcing compile and continuing.\n")

            # Process any tool calls this turn produced before intervening.
            if tool_calls:
                trace_lines.append(f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]")
                messages.extend(_run_tool_calls(tool_calls))

            draft = last_draft[0]
            # No compiled draft yet — try to salvage one from the thinking
            # block. Models often write complete drafts in their scratchpad
            # before going degenerate; those drafts are otherwise discarded.
            if not draft:
                salvaged = _extract_drafts_from_thinking(thinking)
                if "function.hh" in salvaged and "test.cc" in salvaged:
                    trace_lines.append(
                        f"[agent] Salvaged draft from thinking: "
                        f"{sorted(salvaged.keys())}\n"
                    )
                    draft = salvaged
            if draft:
                compile_result = compile_handler(
                    draft["function.hh"],
                    draft["test.cc"],
                    draft.get("function.cc"),
                )
                trace_lines.append(
                    f"[agent] Forced compile result:\n{textwrap.indent(compile_result, '  ')}\n"
                )
                recovery_msg = (
                    f"DEGENERACY DETECTED: your thinking repeated itself excessively and was cut off.\n"
                    f"Your last draft has been compiled automatically:\n\n{compile_result}\n\n"
                    "STRICT RULES — follow exactly:\n"
                    "1. Do NOT repeat any analysis or reasoning you have already done.\n"
                    "2. If all tests passed: output the final fenced code blocks immediately.\n"
                    "3. If tests failed: fix ONLY the specific errors shown, then rewrite the affected file(s) with write_gemmi_file.\n"
                    "4. No preamble, no summary, no re-examination of APIs."
                )
            else:
                recovery_msg = (
                    "DEGENERACY DETECTED: your thinking repeated itself excessively and was cut off.\n"
                    "You have no compiled draft yet. Write your best attempt to disk NOW using "
                    "write_gemmi_file — write function.hh first, then test.cc (compilation triggers "
                    "automatically). Do NOT repeat any prior analysis."
                )
            messages.append({"role": "user", "content": recovery_msg})
            trace_lines.append(f"[degen-recovery]\n{textwrap.indent(recovery_msg, '  ')}\n")
            continue

        _progress(f"turn {turn + 1}/50", tool_calls)

        if not tool_calls:
            # If thinking expressed compile intent but the model produced no tool
            # calls AND no usable fenced output, force a compile from the last draft
            # rather than exiting — the model talked itself into compiling but then
            # stalled before making the call.
            extracted = _extract_blocks(assistant_content) or None
            if (_has_compile_intent(thinking)
                    and not _is_usable(extracted)
                    and last_draft[0]
                    and not degen_recovered[0]):
                draft = last_draft[0]
                compile_result = compile_handler(
                    draft["function.hh"],
                    draft["test.cc"],
                    draft.get("function.cc"),
                )
                trace_lines.append(
                    f"[agent] Compile intent with no tool calls — forced compile:\n"
                    f"{textwrap.indent(compile_result, '  ')}\n"
                )
                recovery_msg = (
                    "You expressed intent to compile but made no tool calls. "
                    f"Your last draft has been compiled automatically:\n\n{compile_result}\n\n"
                    "If all tests passed: output the final fenced code blocks immediately. "
                    "If tests failed: fix only the specific errors shown, then rewrite the "
                    "affected file(s) with write_gemmi_file. Do NOT re-examine any APIs."
                )
                messages.append({"role": "user", "content": recovery_msg})
                trace_lines.append(f"[intent-no-tool-recovery]\n{textwrap.indent(recovery_msg, '  ')}\n")
                continue
            trace_lines.append(f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n")
            final_blocks = extracted
            break
        trace_lines.append(f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]")
        messages.extend(_run_tool_calls(tool_calls))

        # Compile-intent-without-follow-through guard: if thinking expressed
        # intent to compile but no compile_gemmi call was actually made,
        # count the strike. Two consecutive strikes → force compile.
        had_compile_call = any(
            (tc.get("function") or {}).get("name") in ("compile_gemmi", "write_gemmi_file")
            for tc in tool_calls
        )
        if had_compile_call:
            compile_intent_strikes[0] = 0
        elif _has_compile_intent(thinking):
            compile_intent_strikes[0] += 1
            trace_lines.append(
                f"[agent] compile-intent strike {compile_intent_strikes[0]} "
                f"(thinking expressed intent but no compile_gemmi call).\n"
            )
            if compile_intent_strikes[0] >= 1 and last_draft[0]:
                draft = last_draft[0]
                compile_result = compile_handler(
                    draft["function.hh"],
                    draft["test.cc"],
                    draft.get("function.cc"),
                )
                trace_lines.append(
                    f"[agent] Forced compile (intent strikes):\n"
                    f"{textwrap.indent(compile_result, '  ')}\n"
                )
                recovery_msg = (
                    "You expressed intent to compile multiple times but kept researching instead. "
                    f"Your last draft has been compiled automatically:\n\n{compile_result}\n\n"
                    "Do NOT look up any more APIs. If tests passed, output the final blocks. "
                    "If tests failed, fix only the specific errors shown and rewrite the affected "
                    "file(s) with write_gemmi_file."
                )
                messages.append({"role": "user", "content": recovery_msg})
                trace_lines.append(f"[intent-recovery]\n{textwrap.indent(recovery_msg, '  ')}\n")
                compile_intent_strikes[0] = 0

        if (not no_compile_warned[0]
                and not any(k.startswith("compile_gemmi:") for k in call_counts)):
            turn_fired = (_GEMMI_NO_COMPILE_AFTER and (turn + 1) >= _GEMMI_NO_COMPILE_AFTER)
            tool_fired = (not no_tool_nudge_sent[0] and non_compile_tool_calls[0] >= 10)
            if turn_fired or tool_fired:
                messages.append({"role": "user", "content": _GEMMI_NO_COMPILE_NUDGE})
                trace_lines.append(
                    f"[no-compile nudge — turn {turn + 1}, "
                    f"reason={'turn-limit' if turn_fired else 'tool-count'}]\n"
                    f"{textwrap.indent(_GEMMI_NO_COMPILE_NUDGE, '  ')}\n"
                )
                no_compile_warned[0] = True
                no_tool_nudge_sent[0] = True

        if NUDGE_EVERY_N_TURNS and (turn + 1) % NUDGE_EVERY_N_TURNS == 0:
            messages.append({"role": "user", "content": _GEMMI_NUDGE})
            trace_lines.append(f"[nudge — turn {turn + 1}]\n{textwrap.indent(_GEMMI_NUDGE, '  ')}\n")
    else:
        trace_lines.append("[agent] Turn limit reached.\n")

    if not _is_usable(final_blocks) and _is_usable(last_draft[0]):
        trace_lines.append("[agent] Falling back to last compile_gemmi draft.\n")
        final_blocks = last_draft[0]
    elif not _is_usable(final_blocks):
        trace_lines.append("[agent] No usable output — issuing rescue prompt.\n")
        messages.append({"role": "user", "content": (
            "STOP. Do not call any tools. Output your best attempt NOW as "
            "three fenced blocks labelled ```cpp:function.hh, optionally "
            "```cpp:function.cc, and ```cpp:test.cc."
        )})
        try:
            data = _chat(messages, model, tools=[])
            _log_llm_timing(data, stage="gemmi", turn="rescue", verbose=verbose, trace_lines=trace_lines)
            assistant_content = (data.get("message") or {}).get("content") or ""
            trace_lines.append(f"[assistant — rescue]\n{textwrap.indent(assistant_content, '  ')}\n")
            rescued = _extract_blocks(assistant_content)
            if _is_usable(rescued):
                final_blocks = rescued
            elif _is_usable(last_draft[0]):
                final_blocks = last_draft[0]
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            trace_lines.append(f"[agent] Rescue call failed: {e}\n")
            if _is_usable(last_draft[0]):
                final_blocks = last_draft[0]

    # Final assertion check: warn if test.cc dropped or weakened an original
    # RHS literal, but do NOT discard — semantically equivalent expressions
    # are valid ports and a strict reject loses good work. The warning is
    # written to the trace so it can be reviewed after the fact.
    if final_blocks and original_test_cc:
        violation = _check_assertions_unchanged(
            original_test_cc, final_blocks.get("test.cc", "")
        )
        if violation:
            trace_lines.append(
                f"[agent] Final assertion check WARNING (output kept — verify "
                f"RHS is semantically equivalent).\n"
                f"{textwrap.indent(violation, '  ')}\n"
            )

    text = trace_lines.text()
    trace_lines.close()
    return final_blocks, text

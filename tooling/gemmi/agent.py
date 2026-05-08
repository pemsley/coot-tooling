"""Combined agentic port: function.hh (+ optional function.cc) + test.cc in one session.

The original MMDB function source and its MMDB-based test are both supplied.
The agent produces a gemmi equivalent of the function AND a gemmi version of
the test that exercises it — compiled and linked as a single unit so
signatures agree by construction.

Frozen: every EXPECT_* / ASSERT_* line from the original test.
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
    "compile_gemmi NOW to validate it before finalising."
)

_GEMMI_NO_COMPILE_NUDGE = (
    "WARNING: you have not attempted compile_gemmi yet. "
    "Stop researching and DRAFT your best function.hh + test.cc (and "
    "optionally function.cc) NOW, then call compile_gemmi. The compiler's "
    "error messages are far more useful than further speculation about "
    "gemmi APIs. Failures are expected — you have multiple retries to fix "
    "them. Action over analysis."
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

Most-used MMDB → gemmi accessors (call mmdb_to_gemmi("Name") for any others):
  mol->GetModel(1)                → st.models[0]            // 0-indexed!
  chain->GetChainID()             → chain.name              // field, not method
  residue->GetResName()           → residue.name            // field
  residue->GetSeqNum()            → residue.seqid.num.value
  residue->GetInsCode()           → residue.seqid.icode     // char, not const char*
  atom->GetAtomName()             → atom.name               // std::string field
  atom->GetElementName()          → atom.element.name()     // "C","O" — unpadded
  atom->x, atom->y, atom->z       → atom.pos.x, atom.pos.y, atom.pos.z
  atom->occupancy                 → atom.occ
  atom->tempFactor                → atom.b_iso
  chain->GetNumberOfResidues()    → chain.residues.size()
  residue->GetNumberOfAtoms()     → residue.atoms.size()
  ... (~200 more — call mmdb_to_gemmi("<MethodName>") for any MMDB API.
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

If you're unsure which header declares a name, call **include_for_symbol("Foo")**
— authoritative answer, no grep needed. For an MMDB → gemmi method mapping,
call **mmdb_to_gemmi("GetSeqNum")** before grep'ing.\
"""

GEMMI_SYSTEM_PROMPT = f"""\
You are porting ONE C++ function from the MMDB API to the gemmi API AND
translating its Google Test, in the same session.

{GEMMI_CHEAT_SHEET}

## Artifacts to produce

  A. function.hh — header with declaration and #include <gemmi/...> deps.
     Use `#pragma once`. If the body is short, put it here as `inline`.
  B. function.cc — OPTIONAL. Only emit if the body is long or uses
     translation-unit-private helpers. Otherwise omit it entirely.
  C. test.cc — the gemmi-translated Google Test, #include "function.hh".

## Rules

1. Preserve every EXPECT_* / ASSERT_* line's semantic fact — same compared
   value, same comparison operator. You MAY rewrite the left-hand-side
   accessor when the type changes (e.g. `res->GetSeqNum()` becomes
   `res->seqid.num.value`), but you MAY NOT change the expected value or
   relax the check. The original expected numbers are the correctness
   oracle.
2. Port the function semantics 1:1 — same output for the same input.
3. **Naming**: keep the original function's C++ namespace exactly as-is and
   append `_gemmi` to the function name. For example, if the original is
   `coot::angle(...)` the ported function MUST be declared and defined as
   `coot::angle_gemmi(...)`. Do NOT wrap it in a `gemmi::` namespace or any
   other namespace. The task below states the exact target name — use it
   verbatim.
4. The function signature must match what test.cc calls. Design them together.
5. Use the DB tools (lookup_type, list_methods, find_header, find_symbol)
   BEFORE writing any gemmi name. When lookup_type reports an ambiguous
   name, retry with the fully-qualified form. Do not invent APIs.
6. grep_codebase searches both coot and the gemmi header tree — use it
   when you need to see a usage pattern.
7. Link target: test.cc (+ function.cc if present) against -lgemmi_cpp and
   -lgtest. No MMDB, no clipper, no coot libraries.
8. When ready to compile, use **write_gemmi_file** to write each file to disk.
   Write function.cc first (if needed), then function.hh, then test.cc.
   Compilation is triggered automatically once function.hh and test.cc are
   both written — you do not need to call compile_gemmi separately.
   Alternatively, call compile_gemmi directly to pass all contents at once.
   If tests FAIL, fix and rewrite the affected file(s) to recompile.
   Max {MAX_COMPILE_ATTEMPTS} compile attempts total.
9. **Compile early and often.** Look up at most 3–4 APIs, then call
   compile_gemmi with your best draft — do not wait until you are certain.
   Compiler errors are faster feedback than further API research. Once you
   have reasoned through an API question, do not revisit it; act on your
   conclusion immediately.

{GEMMI_ANTIPATTERNS}

Final output format (ONE response, THREE fenced blocks in this exact order):

```cpp:function.hh
... header contents ...
```

```cpp:function.cc
... only if needed; otherwise omit this block entirely ...
```

```cpp:test.cc
... test contents ...
```

If you omit function.cc, just skip the middle block.\
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
            "Call this whenever the compile_gemmi response ends with "
            "'... (N more lines — use get_compile_errors)' — that footer is a "
            "signal you are missing context that almost certainly contains the "
            "real error. Don't try to fix a compile failure from a truncated log."
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
            "Return the canonical #include directive that defines a gemmi or "
            "gtest symbol. Pass natural forms like 'read_pdb_file', "
            "'gemmi::Vec3', 'TEST', 'EXPECT_EQ'. Use this BEFORE grep_codebase "
            "when you need a header for a known name."
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

_WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_gemmi_file",
        "description": (
            "Write one of the three gemmi port files to disk. "
            "Once both function.hh and test.cc have been written, "
            "compilation is triggered automatically — you do not need to call "
            "compile_gemmi separately. Write function.cc first (if needed), "
            "then function.hh, then test.cc to trigger the build."
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
    _COMPILE_TOOL, _RUN_TOOL, _GET_ERRORS_TOOL, _WRITE_FILE_TOOL,
    _MMDB_TO_GEMMI_TOOL, _INCLUDE_FOR_SYMBOL_TOOL,
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
) -> tuple[callable, callable, callable]:
    attempts       = [0]
    last_binary    = [None]
    last_error_log = [None]

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
                "below and call compile_gemmi again:\n"
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
                "below and call compile_gemmi again. These are anti-patterns "
                "the compiler would also reject:\n\n"
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
            status = "All tests PASSED." if run_ok else "Some tests FAILED — fix the assertions and recompile."
            return (
                f"Compilation succeeded (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}).\n"
                f"{status}\n{run_out}"
            )
        last_binary[0] = None
        last_error_log[0] = compile_log
        return f"Compilation FAILED (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}):\n{output}"

    def run_handler() -> str:
        if last_binary[0] is None:
            return "No compiled binary — call compile_gemmi first."
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
            return f"'{filename}' written. Compilation triggered automatically:\n\n{result}"
        missing = [f for f in ("function.hh", "test.cc") if not (gemmi_subdir / f).exists()]
        return f"'{filename}' written. Still waiting for: {', '.join(missing)}"

    return compile_handler, run_handler, get_errors_handler, write_file_handler


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
    compile_handler, run_handler, get_errors_handler, write_file_handler = \
        _make_tool_handlers(gemmi_subdir, dep_includes, dep_sources)

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_gemmi":
            return compile_handler(
                args.get("function_hh", ""),
                args.get("test_cc", ""),
                args.get("function_cc") or None,
            )
        if name == "write_gemmi_file":
            return write_file_handler(args.get("filename", ""), args.get("contents", ""))
        if name == "run_gemmi_test":
            return run_handler()
        if name == "get_compile_errors":
            return get_errors_handler()
        if name == "mmdb_to_gemmi":
            return mmdb_to_gemmi(args.get("method", ""))
        if name == "include_for_symbol":
            return include_for_symbol(args.get("symbol", ""))
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
                "`_gemmi` variant. The build system automatically compiles "
                "any required dep `.cc` files.**"
            )
            for c in ported:
                hh = OUT_ROOT / sanitize_name(c) / "gemmi" / "function.hh"
                lines.append(
                    f"  - `{c}` → `{_gemmi_target_name(c)}`\n"
                    f'    `#include "{hh}"`'
                )
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
                results.append({"role": "tool", "content": note + cached})
                continue
            if call_counts[key] > REPEAT_LIMIT and name not in ("compile_gemmi", "run_gemmi_test"):
                nudge = (
                    f"You have called {name} with these arguments {call_counts[key]} times. "
                    "Stop repeating — proceed to compile_gemmi with your best drafts."
                )
                trace_lines.append(f"  → {name}(repeated — nudged)")
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
            trace_lines.append(
                f"  → {name}({json.dumps(args) if name != 'compile_gemmi' else '{...}'})"
            )
            trace_lines.append(textwrap.indent(result_text, "      ") + "\n")
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

    text = trace_lines.text()
    trace_lines.close()
    return final_blocks, text

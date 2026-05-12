"""
Agentic oracle generation — gemma4 calls tools to explore the codebase on
demand rather than receiving a pre-built context dump.

The model is given the function source and a lean prompt, then iteratively
calls tools (read_file, lookup_function, lookup_type, list_methods,
get_callers, search_functions) until it has enough context to write oracle.cc.

A human-readable trace of every tool call and result is saved alongside the
oracle so you can audit what the model looked up.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import subprocess
import textwrap
from pathlib import Path

from ..db import (
    PROJECT_ROOT,
    get_function,
    get_type,
    get_types_matching,
    get_type_methods,
    get_callers_with_source,
    get_class_functions,
    get_class_methods_with_access,
    get_containing_class,
)
from .render import INCLUDE_ROOTS, _to_include, _load_override, MMDB_MANAGER_SNIPPET, caller_class_fields
from ..llm import chat as _chat


def _log_llm_timing(data: dict, *, stage: str, turn, verbose: bool, trace_lines: list) -> None:
    """Emit one LLM-call timing line to the trace (and stdout if verbose).

    `turn` may be an int or any str label (e.g. "rescue"). Silent unless the
    backend attached `_meta` to the response — older callers that bypass the
    backend wrapper are no-ops.
    """
    meta = data.get("_meta") if isinstance(data, dict) else None
    if not meta:
        return
    elapsed = meta.get("elapsed_s") or 0.0
    pt = meta.get("prompt_tokens")
    ot = meta.get("output_tokens")
    tps = f"{ot / elapsed:.0f}tok/s" if (ot and elapsed > 0) else "?tok/s"
    line = (
        f"[llm — {stage} turn {turn}] elapsed={elapsed:.1f}s "
        f"prompt={pt if pt is not None else '?'} output={ot if ot is not None else '?'} {tps}"
    )
    trace_lines.append(line + "\n")
    if verbose:
        print(f"  {line}", flush=True)


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"  # kept for backward-compat imports

# Periodic format-reminder cadence. Transformer attention skews toward the
# start and end of the context with a measurable dip in the middle ("lost in
# the middle"); after several tool-call turns the original system prompt is
# buried where the model attends to it less. Re-injecting a short reminder
# every N turns puts the format spec back near the end of the context where
# attention is strongest. Set to 0 to disable.
NUDGE_EVERY_N_TURNS = int(os.environ.get("CT_AGENT_NUDGE_EVERY", 5))

# Hard "stop researching, start compiling" nudge. If this many turns elapse
# without a single attempt at the compile tool, fire a one-shot warning that
# names the tool explicitly. Targets the failure mode where the model
# spends 12+ turns analysing instead of testing a draft. Set to 0 to disable.
NO_COMPILE_AFTER = int(os.environ.get("CT_AGENT_NO_COMPILE_AFTER", 10))

NOTES_DIR        = Path(__file__).parent / "notes"
ANSWER_MARKER    = "## Answer"
TEST_DATA_DIR    = Path(__file__).parent.parent.parent / "test-data"
GENERATED_TEST_DIR    = Path(__file__).parent.parent.parent / "generated-tests"
THIRD_PARTY_INCLUDE = str(Path(__file__).parent.parent.parent / "third-party" / "google-test" / "include")

# Paths the model is allowed to read.
ALLOWED_READ_ROOTS = [PROJECT_ROOT] + INCLUDE_ROOTS + [str(TEST_DATA_DIR), THIRD_PARTY_INCLUDE, str(GENERATED_TEST_DIR)]

def make_agent_system_prompt(pdb_path: str, pdb_note: str = "") -> str:
    """Build the agent system prompt with the given PDB path.

    pdb_note is injected after the PDB line when the file choice was ambiguous —
    it lists all available PDB files so the LLM can pick a more appropriate one.
    """
    pdb_section = f"PDB: {pdb_path}"
    if pdb_note:
        pdb_section += (
            "\n(The path above is a default — if the listed alternatives suit "
            f"the function better, use one of them instead.)\n{pdb_note}"
        )
    return f"""\
You are writing oracle.cc — a self-contained C++ program that calls ONE
function and prints its inputs and outputs to stdout in a fixed format.

# Test data
{pdb_section}
MTZ: {TEST_DATA_DIR}/example.mtz   (load this ONLY if the target function
                                    reads map/reflection data)

# Required output format
For every input passed to the function and every meaningful output, print
one line in exactly this form:

    INPUT  <name>: <value>
    OUTPUT <name>: <value>

Example (real working oracle for `coot::molecule_t::cid_to_atom`):

    #include <mmdb2/mmdb_manager.h>
    #include "api/coot-molecule.hh"
    #include "api/molecules-container.hh"

    int main() {{
        molecules_container_t mc;
        int imol = mc.read_pdb("{pdb_path}");
        if (imol < 0) {{ std::cerr << "load failed\\n"; return 1; }}

        // case 1: valid atom
        {{
            std::string cid = "//A/10/CA";
            mmdb::Atom *atom = mc[imol].cid_to_atom(cid);
            std::cout << "INPUT  cid: " << cid << std::endl;
            std::cout << "OUTPUT atom_found: " << (atom ? "true" : "false") << std::endl;
            std::cout << "OUTPUT atom_name: "
                      << (atom ? atom->GetAtomName() : "nullptr") << std::endl;
        }}

        // case 2: invalid CID — verifies the guarded path
        {{
            std::string cid = "//A/9999/N";
            mmdb::Atom *atom = mc[imol].cid_to_atom(cid);
            std::cout << "INPUT  cid: " << cid << std::endl;
            std::cout << "OUTPUT atom_found: " << (atom ? "true" : "false") << std::endl;
        }}
        return 0;
    }}

Cover 2–3 cases (typical + edge) using this pattern. Adapt the receiver
construction to your target function — `molecules_container_t` + `mc[imol]`
is the standard entry point for `coot::molecule_t` methods.

# Proving the function actually ran (read this carefully)
Before you write code, read the function source and identify what it
returns or mutates. Then:
- Returns a value → print it as OUTPUT.
- Void + mutates a container → print container.size() (or a key element)
  BEFORE and AFTER the call. **If before == after, the function did
  nothing and your inputs failed its preconditions** — fix the inputs,
  don't accept a no-op as success.
- Has guard clauses (`if (!x) return;`) → trace them and construct inputs
  that pass every guard to reach the core logic.

run_oracle warns when no OUTPUT lines are produced. Treat that warning
as a hard failure and fix the inputs or the prints.

# Private/protected access
oracle.cc is compiled with `-fno-access-control` — write private and
protected member access as if it were public. No friend declarations, no
casts, no workarounds.

# Workflow — terminal condition
You are done when ALL of these hold:
  1. compile_oracle returned success on your latest code.
  2. run_oracle returned success and the output contains at least one
     INPUT line and at least one OUTPUT line, with BEFORE != AFTER if the
     function is void-mutating.
  3. You have emitted the final program in a single ```cpp fenced block.

Stop after the first run that meets (1) and (2). Do not refine further
for style — the output is the artifact, not the code.

# Available tools (use ONLY these — do not invent tool names)
- read_file          — read a C++ source or header file
- lookup_function    — get source + docs for a function by qualified name
- lookup_type        — get class/struct definition and method list
- list_methods       — list all method signatures in a class
- get_callers        — find functions that call a given function
- find_header        — resolve a type/function name to its header path
- resolve_includes   — check which #includes are needed for a code draft
- search_functions   — find functions by partial name
- grep_codebase      — text search across the source tree
- inspect_pdb        — inspect chains/residues in the PDB file
- get_base_classes   — list base classes of a type
- find_symbol        — locate a symbol in header files
- leave_note         — record a domain question for future reference
- compile_oracle     — compile the current oracle.cc draft
- run_oracle         — run the compiled oracle binary

# Tool ordering
- resolve_includes on your draft before the first compile.
- compile_oracle as soon as you can name the receiver type and one
  meaningful call site. Compiler errors beat further speculation.
- run_oracle once it compiles. Fix prints/inputs and recompile if needed.

# When to stop looking up and start writing
**Hard cap: at most 6 lookup tool calls before your first compile_oracle.**
Lookups include read_file, lookup_function, lookup_type, list_methods,
get_callers, search_functions, grep_codebase. After 6 you MUST draft
oracle.cc and call compile_oracle even if you are uncertain — the
compiler's errors and run_oracle's output are far more useful than
further reading. Re-reading the same file or re-looking-up the same
symbol counts and is wasted; act on what you already know.

Concretely, start writing as soon as you can name: (a) the receiver
construction pattern (typically `molecules_container_t mc; int imol =
mc.read_pdb(...)`), (b) one input that should pass the guards, (c) the
return type or mutation you'll print as OUTPUT. Edge cases can be added
after the first compile + run succeeds.

Write idiomatic C++ (no malloc/printf/raw C arrays unless the function
itself returns them).\
"""


# Backward-compat constant — used by code that imports AGENT_SYSTEM_PROMPT directly.
AGENT_SYSTEM_PROMPT = make_agent_system_prompt(str(TEST_DATA_DIR / "example.pdb"))

_MAX_COMPILE_ATTEMPTS = 20
_EXTENSION_TURNS = 20
_MAX_EXTENSIONS  = 2
_DEGEN_RECOVERY_TURNS = 4

_DEGEN_RECOVERY_NUDGE = (
    "Your thinking just entered an infinite loop — analysis paralysis. "
    "STOP. You have already gathered enough context to write a working oracle. "
    "Call compile_oracle NOW with your best attempt at the code. "
    "Research tools are disabled for the next {n} turns; you may only use "
    "compile_oracle and run_oracle. Write the code and compile it — "
    "compiler errors are far more useful than further speculation."
)

_DEGEN_RECOVERY_NUDGE_POST_COMPILE = (
    "Your thinking just entered an infinite loop — analysis paralysis. "
    "STOP. Your oracle already compiles. The only remaining task is to fix "
    "the runtime output so it produces INPUT/OUTPUT lines. "
    "Use any tools you need to understand why the output is wrong, then "
    "update the print statements, call compile_oracle with the revised code, "
    "and call run_oracle to verify."
)
_EXTENSION_PROMPT = (
    "You have used all available turns. "
    "If you still need to compile, run, or fix errors, respond with tool calls "
    "and you will receive {n} more turns. "
    "If you are done, output the final program in a ```cpp block now."
)

# Format-reminder nudges (injected every NUDGE_EVERY_N_TURNS turns). Kept
# short — they're appended several times during a long run.
_ORACLE_NUDGE = (
    "Reminder: when you stop calling tools, your final reply must be a "
    "single ```cpp fenced block containing the complete oracle.cc. "
    "If you have a working draft, call compile_oracle now to validate it."
)

_ORACLE_NO_COMPILE_NUDGE = (
    "WARNING: you have not attempted compile_oracle yet. "
    "Stop researching and DRAFT your best oracle.cc now, then call "
    "compile_oracle. The compiler's error messages are far more useful "
    "than further speculation. Failures are expected — you have multiple "
    "retries to fix them. Action over analysis."
)

# ── tool schema ───────────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a C++ source or header file. Returns up to 300 lines by default. "
                "If the file is truncated, call again with 'offset' to read the next chunk. "
                "Use 'limit' to request fewer lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string",  "description": "Absolute file path"},
                    "offset": {"type": "integer", "description": "First line to return (0-based). Default 0."},
                    "limit":  {"type": "integer", "description": "Maximum lines to return. Default 300."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_function",
            "description": (
                "Return source code and documentation for a function by its "
                "fully-qualified name, e.g. 'coot::molecule_t::cid_to_residue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"},
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_type",
            "description": (
                "Return the class/struct definition and method list for a type. "
                "Accepts short names like 'Residue' or qualified names like "
                "'mmdb::Residue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_methods",
            "description": "List all method signatures in a C++ class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                },
                "required": ["class_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": (
                "Return example functions that call the given function, showing "
                "real usage patterns and construction of receiver objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"},
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_header",
            "description": (
                "Given a type or function name, return its absolute file path "
                "and the #include directive to use. Call this before read_file "
                "when you need to inspect a header but don't know its path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Type or function qualified name"},
                },
                "required": ["name"],
            },
        },
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "leave_note",
    #         "description": (
    #             "When you are uncertain about something domain-specific and have to guess "
    #             "(e.g. what a valid CID string looks like, which chain to use, expected "
    #             "value ranges), call this to record your question. The user will be notified "
    #             "and can write an answer that will be available in future runs."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "topic": {
    #                     "type": "string",
    #                     "description": "Short slug for the note, e.g. 'cid_format' or 'chain_id'",
    #                 },
    #                 "question": {
    #                     "type": "string",
    #                     "description": "What you are uncertain about and what would help you.",
    #                 },
    #             },
    #             "required": ["topic", "question"],
    #         },
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": "resolve_includes",
            "description": (
                "Check every #include \"...\" directive in your current draft. "
                "Reports which paths resolve correctly and, for those that do not, "
                "searches the coot source tree for a file with the same name and "
                "returns the correct #include path. Call this before finalising "
                "oracle.cc to catch wrong or missing headers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The current C++ draft (full file or just the #include lines)",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_functions",
            "description": (
                "Search for functions whose qualified name contains a substring. "
                "Useful when you don't know the exact name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_fragment": {"type": "string"},
                },
                "required": ["name_fragment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_codebase",
            "description": (
                "Search the coot source tree for a regex pattern. "
                "Returns matching lines with file path and line number. "
                "Use this to find how a type or variable is used, locate a definition, "
                "or discover valid values for a parameter. "
                "Optionally restrict to files matching a glob (e.g. '*.hh', '*.cc')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Restrict search to files matching this glob, e.g. '*.hh'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_pdb",
            "description": (
                "Report what's actually in the selected test PDB: chain IDs, "
                "residue ranges per chain, residue types, and a sample of atom names. "
                "Use this to pick valid inputs (CIDs, chain IDs, residue numbers) "
                "instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {
                        "type": "string",
                        "description": "Optional chain ID. If given, lists every residue in that chain.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_base_classes",
            "description": (
                "Walk the inheritance chain of a type and list methods declared on each "
                "base class. Use when a type appears to be missing a method — it may be "
                "inherited (common with mmdb::Manager → Root, mmdb::Model → Residue, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Type qualified name, e.g. 'mmdb::Manager'",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": (
                "Find the definition of a constant, enum value, macro, or typedef by name. "
                "Returns the defining line(s) and file. Use when you know a symbol's name "
                "(e.g. 'SKEY_NEW', 'STYPE_RESIDUE') but not its value or header."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Exact symbol name"},
                },
                "required": ["symbol"],
            },
        },
    },
]


_COMPILE_ORACLE_TOOL = {
    "type": "function",
    "function": {
        "name": "compile_oracle",
        "description": (
            "Write the supplied C++ code as oracle.cc and attempt to compile it. "
            "Returns compiler output. Fix any errors and call again until it succeeds. "
            f"Maximum {_MAX_COMPILE_ATTEMPTS} attempts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete C++ oracle source to compile",
                },
            },
            "required": ["code"],
        },
    },
}

_RUN_ORACLE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_oracle",
        "description": (
            "Run the last successfully compiled oracle binary and return its output. "
            "Verify that INPUT/OUTPUT lines are printed correctly. "
            "Only callable after a successful compile_oracle."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

ORACLE_TOOLS = TOOLS + [_COMPILE_ORACLE_TOOL, _RUN_ORACLE_TOOL]


class _TraceWriter:
    """Buffered trace that also streams to disk on each append.

    Drop-in for the previous `list[str]` pattern: only `append`/`extend` and
    `text()` are needed. The on-disk file is line-buffered and flushed after
    every write so `tail -f` shows progress live during a run.
    """

    def __init__(self, path: Path | None = None, calls_path: Path | None = None) -> None:
        self._lines: list[str] = []
        self._fp = None
        self._calls_fp = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = path.open("w", buffering=1)
        if calls_path is None and path is not None:
            calls_path = path.with_name("agent_calls.txt")
        if calls_path is not None:
            calls_path.parent.mkdir(parents=True, exist_ok=True)
            self._calls_fp = calls_path.open("w", buffering=1)

    def append(self, line: str) -> None:
        self._lines.append(line)
        if self._fp is not None:
            self._fp.write(line + "\n")
            self._fp.flush()

    def extend(self, lines) -> None:
        for line in lines:
            self.append(line)

    def append_call(self, line: str) -> None:
        """Append a one-liner to the calls-only trace (not the main trace)."""
        if self._calls_fp is not None:
            self._calls_fp.write(line + "\n")
            self._calls_fp.flush()

    def text(self) -> str:
        return "\n".join(self._lines)

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        if self._calls_fp is not None:
            self._calls_fp.close()
            self._calls_fp = None


def _is_degenerate_thinking(
    thinking: str,
    min_lines: int = 30,
    min_line_len: int = 25,
    min_count: int = 8,
    min_ratio: float = 0.10,
    # Block-level repetition: a run of N consecutive substantive lines
    # that appears more than max_block_repeats times signals a loop even
    # when no single line individually crosses the per-line threshold.
    block_ngram_size: int = 5,
    max_block_repeats: int = 3,
) -> tuple[bool, str]:
    """Detect pathological repetition in a thinking block.

    Three checks:
    0. Stream-abort marker: if the streaming guard in llm.py already aborted
       this turn, treat as ground truth. The streaming guard catches
       paraphrastic loops (variations of "Let me proceed... Actually...") that
       fall through the line-frequency check below because no single literal
       line repeats often enough.
    1. Single-line: a substantive line (≥min_line_len chars) appears
       ≥min_count times AND ≥min_ratio of all substantive lines.
    2. Block-level: a tuple of block_ngram_size consecutive substantive
       lines appears more than max_block_repeats times, catching the case
       where the model re-writes entire paragraphs rather than single lines.
    """
    from ..llm import STREAM_ABORT_MARKER
    if STREAM_ABORT_MARKER in thinking:
        return True, "Streaming-time degeneracy guard fired during this turn."
    from collections import Counter
    # Strip fenced code blocks before analysis — repeated constant initialisers,
    # parallel EXPECT_EQ lines, and array literals inside ```...``` are legitimate
    # code, not degeneracy. Also strip unclosed fences (agent cut off mid-block).
    prose = re.sub(r"```.*?```", "", thinking, flags=re.DOTALL)
    prose = re.sub(r"```.*$", "", prose, flags=re.DOTALL)
    lines = [l.strip() for l in prose.splitlines()
             if len(l.strip()) >= min_line_len]
    if len(lines) < min_lines:
        return False, ""

    # Check 1: single-line frequency
    counter = Counter(lines)
    top_line, top_count = counter.most_common(1)[0]
    ratio = top_count / len(lines)
    if top_count >= min_count and ratio >= min_ratio:
        snippet = (top_line[:80] + "…") if len(top_line) > 80 else top_line
        return True, (
            f"Degenerate thinking detected: most-frequent line repeated "
            f"{top_count}× ({ratio:.0%} of {len(lines)} substantive lines). "
            f"Sample: {snippet!r}"
        )

    # Check 2: block-level N-gram repetition
    if len(lines) >= block_ngram_size:
        ngrams = Counter(
            tuple(lines[i:i + block_ngram_size])
            for i in range(len(lines) - block_ngram_size + 1)
        )
        top_ngram, top_ng_count = ngrams.most_common(1)[0]
        if top_ng_count > max_block_repeats:
            snippet = " | ".join(s[:40] for s in top_ngram[:2])
            return True, (
                f"Degenerate thinking detected: {block_ngram_size}-line block "
                f"repeated {top_ng_count}× (threshold {max_block_repeats}). "
                f"Sample: {snippet!r}…"
            )

    return False, ""


def _has_compile_intent(thinking: str) -> bool:
    """Return True if thinking expresses an intent to compile but no tool call followed.

    Matches phrases like "let me compile", "I'll call compile_gemmi", "time to
    compile", "draft and compile" — the signal that the model decided to compile
    then talked itself out of it before making the actual tool call.
    """
    return bool(re.search(
        r"let me (?:just |now )?(?:compile|draft|call compile)"
        r"|i(?:'ll| will) (?:compile|call compile|draft)"
        r"|(?:time|ready) to compile"
        r"|call compile_gemmi now"
        r"|draft.*and.*compile"
        r"|compile.*now",
        thinking,
        re.IGNORECASE,
    ))


# ── tool implementations ──────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def _suggest_paths_by_basename(basename: str, max_hits: int = 5) -> list[str]:
    """Scan ALLOWED_READ_ROOTS for files matching `basename`. Used to
    recover from ENOENT in read_file when the model strips a subdir.
    """
    if not basename or "/" in basename or "\\" in basename:
        return []
    seen: set[str] = set()
    hits: list[str] = []
    for root in ALLOWED_READ_ROOTS:
        root_p = Path(root)
        if not root_p.is_dir():
            continue
        try:
            for p in root_p.rglob(basename):
                if not p.is_file():
                    continue
                resolved = str(p.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                hits.append(str(p))
                if len(hits) >= max_hits:
                    return hits
        except OSError:
            continue
    return hits


def _tool_read_file(path: str, offset: int = 0, limit: int = 300) -> str:
    real = Path(path).resolve()
    if not any(str(real).startswith(root) for root in ALLOWED_READ_ROOTS):
        return f"ERROR: path '{path}' is outside allowed roots. {ALLOWED_READ_ROOTS=} {real=}"
    # Some models pass numeric args as strings (e.g. '0.0'). Coerce defensively.
    try:
        offset = int(float(offset))
        limit = int(float(limit))
    except (TypeError, ValueError):
        return f"ERROR: offset/limit must be numeric, got offset={offset!r} limit={limit!r}"
    try:
        text = real.read_text(errors="replace")
    except FileNotFoundError as e:
        suggestions = _suggest_paths_by_basename(real.name)
        msg = f"ERROR: {e}"
        if suggestions:
            shown = "\n".join(f"  {p}" for p in suggestions)
            msg += (
                f"\nDid you mean one of these (same basename '{real.name}')?\n"
                f"{shown}"
            )
        return msg
    except OSError as e:
        return f"ERROR: {e}"
    lines = text.splitlines()
    total = len(lines)
    chunk = lines[offset:offset + limit]
    result = "\n".join(chunk)
    remaining = total - (offset + len(chunk))
    if remaining > 0:
        result += f"\n\n... ({remaining} more lines, call with offset={offset + len(chunk)} to continue)"
    return result


def _tool_lookup_function(conn: sqlite3.Connection, qualified_name: str) -> str:
    row = get_function(conn, qualified_name)
    if not row:
        return f"Function '{qualified_name}' not found in DB."
    parts = []
    if row["comment"]:
        parts.append(f"// {row['comment']}")
    parts.append(row["source_code"] or "(no source)")
    return "\n".join(parts)


# Hint banner appended to lookup_type output for the gemmi types that the
# agent confuses most often. Targets the recurring "non-pointer type" and
# "no member named X" errors visible in generated-tests/*/gemmi/compile.log.
_GEMMI_TYPE_HINTS: dict[str, str] = {
    "gemmi::Structure": (
        "VALUE TYPE — st is `gemmi::Structure`, NOT `gemmi::Structure*`.\n"
        "  Children are values too: st.models[0] is a Model (not Model*).\n"
        "  Common fields: name, cell, spacegroup_hm, models (vector<Model>),\n"
        "  connections (NOT links), assemblies, raw_remarks, helices, sheets,\n"
        "  cispeps. NO `space_group` field — use `spacegroup_hm`."
    ),
    "gemmi::Model": (
        "VALUE TYPE — model is `Model`, NOT `Model*`. Stored in st.models.\n"
        "  Children: chains (vector<Chain>). NO Model::n_atoms() — use model.all().\n"
        "  Sheets/helices/cispeps/connections live on Structure, NOT Model."
    ),
    "gemmi::Chain": (
        "VALUE TYPE — chain is `Chain`, NOT `Chain*`. Stored in model.chains.\n"
        "  Fields: name (NOT GetChainID()), residues (vector<Residue>).\n"
        "  Residue lookup: chain.find_residue(ResidueId), find_or_add_residue."
    ),
    "gemmi::Residue": (
        "VALUE TYPE — residue is `Residue`, NOT `Residue*` (unless from\n"
        "  find_residue → returns nullable pointer). Stored in chain.residues.\n"
        "  NO parent pointer: residue.chain DOES NOT EXIST. Track Chain*\n"
        "  alongside, or iterate chains→residues. Inherits ResidueId:\n"
        "  residue.name, residue.seqid.num.value, residue.seqid.icode,\n"
        "  residue.subchain, residue.entity_id, residue.atoms, residue.het_flag.\n"
        "  NO Residue::add_atom() — use residue.atoms.push_back(atom)."
    ),
    "gemmi::Atom": (
        "VALUE TYPE — atom is `Atom`, NOT `Atom*`. Stored in residue.atoms.\n"
        "  Fields: name (std::string), element (gemmi::Element), pos (Position\n"
        "  with .x/.y/.z), occ, b_iso, altloc (NOT alt_loc — no underscore),\n"
        "  charge (signed char), serial, flag.\n"
        "  Construct element via gemmi::Element(\"C\"), NOT gemmi::Element::C."
    ),
    "gemmi::Fractional": (
        "Inherits from Vec3 — uses .x/.y/.z. NO .u/.v/.w members."
    ),
    "gemmi::ResidueId": (
        "Sequence number lives at .seqid.num.value (SeqId nests OptionalInt).\n"
        "  Insertion code at .seqid.icode (char). NO .num or .icode directly.\n"
        "  ⚠ MMDB uses \"\" (empty string) for no insertion code; gemmi uses ' ' (space).\n"
        "  std::string(1, icode) for an ordinary residue gives \" \", NOT \"\".\n"
        "  Normalize before comparing: treat \"\" as equivalent to \" \"."
    ),
    "molecules_container_t": (
        "GLOBAL NAMESPACE — write `molecules_container_t mc;`, NOT\n"
        "  `coot::molecules_container_t`. The class is declared in the global\n"
        "  namespace in api/molecules-container.hh. Adding a `coot::` prefix\n"
        "  fails with: \"no type named 'molecules_container_t' in namespace 'coot'\"."
    ),
}


def _tool_lookup_type(conn: sqlite3.Connection, name: str) -> str:
    # Disambiguate bare names: 'Residue' exists in mmdb, gemmi, coot and more;
    # silently picking one arbitrary hit is how the agent ends up inventing
    # methods on the wrong type. When multiple candidates exist, show them
    # and let the next call use a qualified name.
    candidates = get_types_matching(conn, name)
    if len(candidates) > 1:
        lines = [
            f"Type '{name}' is ambiguous — {len(candidates)} matches. "
            "Call lookup_type again with a fully-qualified name:",
        ]
        for c in candidates:
            lines.append(f"  {c['qualified_name']:40s}  ({c['file']})")
        return "\n".join(lines)
    row = candidates[0] if candidates else get_type(conn, name)
    if not row:
        return f"Type '{name}' not found in DB."
    methods = get_type_methods(conn, row["qualified_name"])
    lines = [f"// {row['kind']} {row['qualified_name']}  ({row['file']})"]
    # Prepend the hint where applicable — placed above the summary so it is
    # visible even when the model only reads the first chunk of the response.
    hint = _GEMMI_TYPE_HINTS.get(row["qualified_name"])
    if hint:
        lines.append("// HINT — common pitfalls:")
        for hint_line in hint.splitlines():
            lines.append(f"//   {hint_line}")
    lines.append(row["summary"] or "(no summary)")
    if methods:
        lines.append("\n// Methods:")
        for m in methods:
            comment = f"  // {m['comment']}" if m["comment"] else ""
            lines.append(f"  {m['display_name']}{comment}")
    return "\n".join(lines)


def _tool_list_methods(conn: sqlite3.Connection, class_name: str) -> str:
    """List methods of a class, with access modifiers prefixed.

    The visibility marker matters: without it, the model picks an interesting-
    looking method, hits a "private member" compile error, and has to grep
    around to recover. Showing [public]/[private]/[protected] up front lets
    the model rule out non-callable methods immediately.
    """
    rows = get_class_methods_with_access(conn, class_name)
    if not rows:
        return f"No methods found for '{class_name}'."
    out: list[str] = []
    for qn, access in rows:
        marker = f"[{access}]" if access else "[?]"
        out.append(f"{marker:<11} {qn}")
    out.append("")
    out.append(
        "All members ([public], [protected], [private]) are callable from "
        "oracle.cc — it is compiled with -fno-access-control. Just call them "
        "directly."
    )
    return "\n".join(out)


def _tool_get_callers(conn: sqlite3.Connection, qualified_name: str) -> str:
    fn = get_function(conn, qualified_name)
    if not fn:
        return f"Function '{qualified_name}' not found."
    callers = get_callers_with_source(conn, fn["id"], limit=3)
    if not callers:
        return "No callers found."
    parts = []
    for c in callers:
        rel = c["file"].replace(PROJECT_ROOT + "/", "")
        parts.append(f"// {rel}")
        if c["comment"]:
            parts.append(f"// {c['comment']}")
        parts.append(c["source_code"].rstrip())
    return "\n\n".join(parts)


def _tool_find_header(conn: sqlite3.Connection, name: str) -> str:
    file_path = None
    row = get_type(conn, name)
    if row:
        file_path = row["file"]
    else:
        fn = get_function(conn, name)
        if fn:
            file_path = fn["file"]
    if not file_path:
        return f"No type or function '{name}' found in DB."
    include = _to_include(file_path)
    return f'Absolute path: {file_path}\n#include "{include}"'


def _tool_leave_note(topic: str, question: str) -> str:
    NOTES_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9_]", "_", topic.lower()).strip("_")
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        path.write_text(
            f"# {topic}\n\n"
            f"## Question\n{question}\n\n"
            f"{ANSWER_MARKER}\n<!-- Fill in your answer here -->\n"
        )
        return f"Note created: {path}"
    return f"Note already exists: {path} (not overwritten)"


def _load_notes() -> str | None:
    """Return all notes as a single context block, or None if the dir is empty."""
    if not NOTES_DIR.exists():
        return None
    notes = sorted(NOTES_DIR.glob("*.md"))
    if not notes:
        return None
    parts = []
    for p in notes:
        parts.append(f"=== {p.stem} ===\n{p.read_text().strip()}")
    return "\n\n".join(parts)


def _unanswered_notes() -> list[Path]:
    """Return notes that have no answer written yet."""
    if not NOTES_DIR.exists():
        return []
    unanswered = []
    for p in sorted(NOTES_DIR.glob("*.md")):
        text = p.read_text()
        marker_pos = text.find(ANSWER_MARKER)
        if marker_pos == -1:
            continue
        answer_body = text[marker_pos + len(ANSWER_MARKER):].strip()
        if not answer_body or answer_body.startswith("<!--"):
            unanswered.append(p)
    return unanswered


_INCLUDE_RE = re.compile(r'#include\s+([<"])([^">]+)[">]')

# C++ and C standard-library headers come from the compiler's built-in
# include path, not from ALLOWED_READ_ROOTS. Without this allowlist the
# resolver flags them as MISSING — and worse, can produce dangerous
# "WRONG PATH" suggestions when rglob picks up an unrelated project file
# that shares the name (e.g. `<string>` matched against
# numpy/f2py/tests/src/string). The list covers C++17/20 + C-compatibility
# headers + POSIX system headers commonly used in this codebase.
_STDLIB_HEADERS: frozenset[str] = frozenset({
    # C++ containers / utilities
    "algorithm", "any", "array", "atomic", "barrier", "bit", "bitset",
    "charconv", "chrono", "compare", "complex", "concepts",
    "condition_variable", "coroutine", "deque", "exception", "execution",
    "expected", "filesystem", "format", "forward_list", "fstream",
    "functional", "future", "generator", "initializer_list", "iomanip",
    "ios", "iosfwd", "iostream", "istream", "iterator", "latch", "limits",
    "list", "locale", "map", "memory", "memory_resource", "mutex", "new",
    "numbers", "numeric", "optional", "ostream", "print", "queue", "random",
    "ranges", "ratio", "regex", "scoped_allocator", "semaphore", "set",
    "shared_mutex", "source_location", "span", "sstream", "stack",
    "stacktrace", "stdexcept", "stop_token", "streambuf", "string",
    "string_view", "syncstream", "system_error", "thread", "tuple",
    "type_traits", "typeindex", "typeinfo", "unordered_map", "unordered_set",
    "utility", "valarray", "variant", "vector", "version",
    # C-compatibility (cXXX)
    "cassert", "cctype", "cerrno", "cfenv", "cfloat", "cinttypes",
    "climits", "clocale", "cmath", "csetjmp", "csignal", "cstdarg",
    "cstdbool", "cstddef", "cstdint", "cstdio", "cstdlib", "cstring",
    "ctime", "cuchar", "cwchar", "cwctype",
    # C standard library (legacy .h)
    "assert.h", "complex.h", "ctype.h", "errno.h", "fenv.h", "float.h",
    "inttypes.h", "iso646.h", "limits.h", "locale.h", "math.h", "setjmp.h",
    "signal.h", "stdalign.h", "stdarg.h", "stdatomic.h", "stdbool.h",
    "stddef.h", "stdint.h", "stdio.h", "stdlib.h", "stdnoreturn.h",
    "string.h", "tgmath.h", "threads.h", "time.h", "uchar.h", "wchar.h",
    "wctype.h",
    # POSIX / system headers commonly needed
    "dirent.h", "fcntl.h", "getopt.h", "libgen.h", "pthread.h", "regex.h",
    "sched.h", "semaphore.h", "strings.h", "syslog.h", "termios.h",
    "unistd.h", "utime.h",
})


def _fmt_include(path: str, delim: str) -> str:
    return f'#include "{path}"' if delim == '"' else f'#include <{path}>'


def _tool_resolve_includes(code: str) -> str:
    """Verify every #include in `code`. Covers both "..." and <...> forms."""
    includes = _INCLUDE_RE.findall(code)
    if not includes:
        return "No #include directives found in the supplied code."

    lines: list[str] = []
    for delim, inc in includes:
        shown = _fmt_include(inc, delim)

        # Short-circuit C++/C/POSIX standard-library headers. They live in
        # the compiler's built-in include path, not ALLOWED_READ_ROOTS, so
        # without this check they spuriously fail. A bracket form with no
        # path separator and a name in the stdlib allowlist is always OK.
        # GCC implementation sub-directories (bits/, ext/, tr1/) are also
        # compiler-internal and must be accepted without a source-tree search.
        if delim == "<" and "/" not in inc and inc in _STDLIB_HEADERS:
            lines.append(f'OK  {shown}  (C/C++ standard library)')
            continue
        _GCC_INTERNAL_PREFIXES = ("bits/", "ext/", "tr1/", "experimental/")
        if delim == "<" and any(inc.startswith(p) for p in _GCC_INTERNAL_PREFIXES):
            lines.append(f'OK  {shown}  (GCC compiler-internal header)')
            continue

        # Sibling files we generate ourselves (function.hh, function.cc,
        # test.cc, oracle.cc) live alongside the file being compiled — they
        # don't exist in any allowed source root yet at lint time, so accept
        # them unconditionally. Any OTHER bare-filename quoted include is
        # checked against the source tree below.
        _GENERATED_SIBLINGS = {"function.hh", "function.cc", "test.cc", "oracle.cc"}
        if delim == '"' and "/" not in inc and inc in _GENERATED_SIBLINGS:
            lines.append(f'OK  {shown}  (generated sibling file)')
            continue

        # Try resolving from each allowed root directly.
        found_at: list[str] = []
        for root in ALLOWED_READ_ROOTS:
            candidate = Path(root) / inc
            if candidate.exists():
                found_at.append(str(candidate))

        if found_at:
            lines.append(f'OK  {shown}  →  {found_at[0]}')
            continue

        # Not found at the given path — search by filename across the tree.
        # Skip common junk dirs that produce dangerous false positives —
        # e.g. rglob("string") would otherwise match
        # ".venv/lib/python3.13/site-packages/numpy/f2py/tests/src/string"
        # and the model would obediently change its #include to that path,
        # breaking the build.
        _SKIP_DIR_PARTS = {".venv", ".git", "node_modules", "__pycache__",
                           "build", "_build", "dist", ".tox", ".cache"}
        filename = Path(inc).name
        matches: list[Path] = []
        for root in ALLOWED_READ_ROOTS:
            for candidate in Path(root).rglob(filename):
                if not _SKIP_DIR_PARTS.intersection(candidate.parts):
                    matches.append(candidate)
        matches = sorted(set(matches))[:5]

        if not matches:
            lines.append(f'MISSING  {shown}  (no file named {filename!r} found in source tree)')
            continue

        lines.append(f'WRONG PATH  {shown}')
        for m in matches:
            for root in ALLOWED_READ_ROOTS:
                try:
                    rel = m.relative_to(root)
                    lines.append(f'  use instead:  {_fmt_include(str(rel), delim)}  (at {m})')
                    break
                except ValueError:
                    continue
            else:
                lines.append(f'  found at:  {m}  (no standard include root covers this)')

    return "\n".join(lines)


def _has_unresolved_includes(report: str) -> bool:
    return any(line.startswith(("WRONG PATH", "MISSING")) for line in report.splitlines())


def _tool_search_functions(conn: sqlite3.Connection, name_fragment: str) -> str:
    rows = conn.execute("""
        SELECT DISTINCT qualified_name FROM functions
        WHERE qualified_name LIKE ?
        LIMIT 30
    """, (f"%{name_fragment}%",)).fetchall()
    if rows:
        return "\n".join(r[0] for r in rows)

    # Zero hits — common cause is a wrong namespace prefix (e.g. searching
    # 'coot::molecules_container_t::' when the class is actually global).
    # Retry with just the tail after the last '::' and, if THAT matches,
    # return the results with a heads-up.
    if "::" in name_fragment:
        tail = name_fragment.rsplit("::", 1)[-1] or name_fragment.rsplit("::", 2)[-2]
        if tail and tail != name_fragment:
            retry = conn.execute("""
                SELECT DISTINCT qualified_name FROM functions
                WHERE qualified_name LIKE ?
                LIMIT 30
            """, (f"%{tail}%",)).fetchall()
            if retry:
                return (
                    f"No matches for '{name_fragment}'. The namespace prefix "
                    f"may be wrong — showing matches for '{tail}' instead "
                    f"(verify the qualified name before using):\n"
                    + "\n".join(r[0] for r in retry)
                )
    return f"No functions matching '{name_fragment}'."


_GREP_EXTS = {".cc", ".cpp", ".cxx", ".c", ".h", ".hh", ".hpp", ".hxx"}
_SKIP_DIRS = {".git", ".claude", "build", "node_modules", "__pycache__", ".venv"}

import shutil as _shutil
_RG_BIN = _shutil.which("rg")


def _grep_files(pattern: str, roots: list[str], glob: str | None, max_matches: int) -> list[str]:
    """Grep `roots` for `pattern`, preferring ripgrep when available.

    Returns up to max_matches lines formatted as 'path:line:content'.
    Excludes VCS/build/worktree directories regardless of backend.
    """
    if _RG_BIN:
        cmd = [_RG_BIN, "--line-number", "--with-filename", "--no-heading",
               "--max-count", str(max_matches), pattern]
        for d in _SKIP_DIRS:
            cmd += ["--glob", f"!**/{d}/**"]
        if glob:
            # Ensure path-like globs (containing '/') are matched anywhere in
            # the tree, not just at the root.
            rg_glob = glob if glob.startswith("**/") else f"**/{glob}"
            cmd += ["--glob", rg_glob]
        cmd += [r for r in roots if Path(r).is_dir()]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return ["ERROR: search timed out."]
        if proc.returncode not in (0, 1):  # 1 = no matches, >1 = error
            err = (proc.stderr or "").strip().splitlines()[:1]
            if err:
                return [f"ERROR: {err[0]}"]
        return proc.stdout.splitlines()[:max_matches]

    # Python fallback
    import fnmatch
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return [f"ERROR: invalid regex: {e}"]
    results: list[str] = []
    for root in roots:
        root_p = Path(root)
        if not root_p.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root_p):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                path = Path(dirpath) / fname
                if glob:
                    # Match against full relative path so "dir/file.cc" globs work.
                    rel = str(path.relative_to(root_p))
                    if not fnmatch.fnmatch(rel, glob) and not fnmatch.fnmatch(fname, glob):
                        continue
                elif path.suffix not in _GREP_EXTS:
                    continue
                try:
                    with path.open(errors="replace") as fh:
                        for n, line in enumerate(fh, 1):
                            if rx.search(line):
                                results.append(f"{path}:{n}:{line.rstrip()}")
                                if len(results) >= max_matches:
                                    return results
                except OSError:
                    continue
    return results


# Patterns the agent reaches for when it's run out of ideas. Each of these
# would match thousands of unrelated lines and tells the model nothing it
# couldn't get from a more focused tool. The agent's job at this point is
# usually visibility/structure analysis — for which list_methods, lookup_type,
# or get_base_classes is the right tool, not text search.
_VAGUE_GREP_PATTERNS: dict[str, str] = {
    "public:":     "use list_methods — it now reports access modifiers directly",
    "private:":    "use list_methods — it now reports access modifiers directly",
    "protected:":  "use list_methods — it now reports access modifiers directly",
    "class ":      "use lookup_type or search_functions for a specific name",
    "struct ":     "use lookup_type or search_functions for a specific name",
    "namespace ":  "use lookup_function with the qualified name you suspect",
    "void ":       "too generic — supply a function or method name to grep for",
    "int ":        "too generic — supply a function or method name to grep for",
    "auto ":       "too generic — supply a function or method name to grep for",
    "return ":     "too generic — supply a function or method name to grep for",
    "include":     "use find_header or resolve_includes",
    "#include":    "use find_header or resolve_includes",
}


def _tool_grep_codebase(
    pattern: str,
    glob: str | None = None,
    extra_roots: list[str] | None = None,
) -> str:
    # Reject patterns that match basically everything — they waste tokens
    # and stall the agent in "I'm out of ideas" mode without useful signal.
    stripped = pattern.strip()
    if stripped in _VAGUE_GREP_PATTERNS:
        return (
            f"Refusing pattern '{pattern}' — it would match thousands of "
            f"unrelated lines across the codebase. "
            f"{_VAGUE_GREP_PATTERNS[stripped]}."
        )

    MAX = 101
    roots = [PROJECT_ROOT] + (extra_roots or [])
    matches = _grep_files(pattern, roots, glob, MAX)
    if not matches:
        return f"No matches for pattern '{pattern}'."
    if matches[0].startswith("ERROR:"):
        return matches[0]
    if len(matches) >= MAX:
        return "\n".join(matches[:MAX - 1]) + f"\n... (more matches found, refine your pattern)"
    return "\n".join(matches)


def _tool_inspect_pdb(chain: str | None = None, pdb_path: Path | None = None) -> str:
    """Parse the selected example PDB and summarise its contents.

    Without 'chain': list chain IDs, residue count, and residue range per chain.
    With 'chain': list every (seq_num, ins_code, res_name) in that chain plus
    a sample of atom names.
    """
    if pdb_path is None:
        pdb_path = TEST_DATA_DIR / "example.pdb"
    if not pdb_path.exists():
        return f"ERROR: {pdb_path} not found."

    # chain_id -> list[(seq_num, ins_code, res_name, atom_name)]
    from collections import defaultdict
    records: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)

    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            atom_name = line[12:16].strip()
            res_name  = line[17:20].strip()
            chain_id  = line[21:22].strip() or "_"
            seq_num   = int(line[22:26])
            ins_code  = line[26:27].strip()
        except (ValueError, IndexError):
            continue
        records[chain_id].append((seq_num, ins_code, res_name, atom_name))

    if not records:
        return f"No ATOM/HETATM records found in {pdb_path.name}."

    if chain:
        rows = records.get(chain)
        if not rows:
            return f"Chain '{chain}' not found. Available chains: {sorted(records)}"
        residues: dict[tuple[int, str], tuple[str, set[str]]] = {}
        for seq, ins, res, atom in rows:
            key = (seq, ins)
            if key not in residues:
                residues[key] = (res, set())
            residues[key][1].add(atom)
        lines = [f"Chain '{chain}' — {len(residues)} residues:"]
        for (seq, ins), (res, atoms) in sorted(residues.items()):
            sample = ", ".join(sorted(atoms)[:6])
            if len(atoms) > 6:
                sample += f", ... ({len(atoms)} atoms)"
            ins_str = ins or ""
            lines.append(f"  {res} {seq}{ins_str}  atoms: {sample}")
            if len(lines) > 200:
                lines.append(f"  ... (truncated, {len(residues)} total residues)")
                break
        return "\n".join(lines)

    lines = [f"{pdb_path.name} — {len(records)} chain(s):"]
    for cid in sorted(records):
        rows = records[cid]
        seqs = sorted({(s, i) for s, i, _, _ in rows})
        residues_by_id: dict[tuple[int, str], str] = {}
        for s, i, r, _ in rows:
            residues_by_id.setdefault((s, i), r)
        res_types = sorted({r for r in residues_by_id.values()})
        first, last = seqs[0], seqs[-1]
        first_str = f"{first[0]}{first[1]}"
        last_str  = f"{last[0]}{last[1]}"
        lines.append(
            f"  chain '{cid}': {len(residues_by_id)} residues, "
            f"seq {first_str}..{last_str}, types: {', '.join(res_types[:10])}"
            + (f", ... ({len(res_types)} total)" if len(res_types) > 10 else "")
        )
    lines.append("\nCall inspect_pdb(chain='X') to list every residue in a chain.")
    return "\n".join(lines)


def _tool_get_base_classes(conn: sqlite3.Connection, name: str) -> str:
    """Walk the inheritance chain and list methods on each base class."""
    _BASE_RE = re.compile(r"^(?:class|struct)\s+\S+\s*:\s*(.+?)\s*\{")
    _ACCESS_RE = re.compile(r"^(public|protected|private|virtual)\s+")

    def parse_bases(summary: str) -> list[str]:
        if not summary:
            return []
        first = summary.splitlines()[0]
        m = _BASE_RE.match(first)
        if not m:
            return []
        bases = []
        for part in m.group(1).split(","):
            p = part.strip()
            while _ACCESS_RE.match(p):
                p = _ACCESS_RE.sub("", p).strip()
            if p:
                bases.append(p)
        return bases

    seen: set[str] = set()
    out: list[str] = []
    queue: list[tuple[str, int]] = [(name, 0)]

    while queue:
        qname, depth = queue.pop(0)
        if qname in seen or depth > 4:
            continue
        seen.add(qname)

        row = get_type(conn, qname)
        if not row:
            out.append(f"{'  ' * depth}{qname}  (not in DB)")
            continue

        resolved = row["qualified_name"]
        out.append(f"{'  ' * depth}{resolved}  ({row['file']})")

        methods = get_type_methods(conn, resolved)
        if methods:
            for m in methods[:15]:
                comment = f"  // {m['comment']}" if m["comment"] else ""
                out.append(f"{'  ' * depth}  {m['display_name']}{comment}")
            if len(methods) > 15:
                out.append(f"{'  ' * depth}  ... ({len(methods) - 15} more methods)")

        for base in parse_bases(row["summary"] or ""):
            queue.append((base, depth + 1))

    if len(out) == 1:
        out.append("  (no base classes found)")
    return "\n".join(out)


def _tool_find_symbol(symbol: str) -> str:
    """Locate the definition of a constant / enum value / macro / typedef."""
    if not re.match(r"^\w+$", symbol):
        return f"ERROR: symbol '{symbol}' must be a plain identifier."

    patterns = [
        rf"#define\s+{symbol}\b",
        rf"\b{symbol}\s*=",
        rf"\btypedef\b[^;]*\b{symbol}\s*;",
        rf"^\s*{symbol}\s*,?\s*(//.*)?$",
    ]
    combined = "|".join(f"(?:{p})" for p in patterns)

    roots = [PROJECT_ROOT] + [r for r in INCLUDE_ROOTS if r != PROJECT_ROOT]
    MAX = 41
    matches = _grep_files(combined, roots, glob=None, max_matches=MAX)
    if not matches:
        return f"No definition found for '{symbol}'."
    if matches[0].startswith("ERROR:"):
        return matches[0]
    if len(matches) >= MAX:
        return "\n".join(matches[:MAX - 1]) + f"\n... (more matches)"
    return "\n".join(matches)


def _make_oracle_tool_handlers(oracle_out: Path) -> tuple[callable, callable, callable]:
    """Return (compile_handler, run_handler, compiled_ok) for the oracle agent loop.

    compiled_ok() returns True once at least one compile_oracle call has succeeded.
    """
    from .compile import write_compile_script

    attempts    = [0]
    last_binary = [None]

    def compile_handler(code: str) -> str:
        if attempts[0] >= _MAX_COMPILE_ATTEMPTS:
            return (
                f"Compile limit reached ({_MAX_COMPILE_ATTEMPTS} attempts). "
                "Output your best draft as the final ```cpp block."
            )

        # Pre-flight: verify every #include before spending a compile attempt.
        # If any are WRONG PATH / MISSING, return the report and do NOT count
        # this as an attempt — the agent gets a free fix cycle.
        include_report = _tool_resolve_includes(code)
        if _has_unresolved_includes(include_report):
            return (
                "Include check FAILED (this does not count against your "
                f"{_MAX_COMPILE_ATTEMPTS} compile attempts). Fix the paths "
                "below and call compile_oracle again:\n"
                + include_report
            )

        attempts[0] += 1
        oracle_out.mkdir(parents=True, exist_ok=True)
        oracle_cc = oracle_out / "oracle.cc"
        oracle_bin = oracle_out / "oracle"
        oracle_cc.write_text(code)
        write_compile_script(oracle_out)
        try:
            proc = subprocess.run(
                ["sh", str(oracle_out / "compile.sh")],
                capture_output=True, text=True, cwd=str(oracle_out),
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            last_binary[0] = None
            return (
                f"Compilation FAILED (attempt {attempts[0]}/{_MAX_COMPILE_ATTEMPTS}): "
                "timed out after 180s — likely a runaway template instantiation. "
                "Simplify includes or types and try again."
            )
        output = (proc.stdout + proc.stderr).strip()
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        if proc.returncode == 0:
            last_binary[0] = oracle_bin
            return f"Compilation succeeded (attempt {attempts[0]}/{_MAX_COMPILE_ATTEMPTS})."
        last_binary[0] = None
        msg = f"Compilation FAILED (attempt {attempts[0]}/{_MAX_COMPILE_ATTEMPTS}):\n{output}"
        if (
            "no type named 'molecules_container_t' in namespace 'coot'" in output
            or "coot::molecules_container_t" in output
        ):
            msg += (
                "\n\nFIX: `molecules_container_t` is in the GLOBAL namespace, "
                "not `coot::`. Replace every `coot::molecules_container_t` with "
                "`molecules_container_t` and recompile."
            )
        return msg

    def run_handler() -> str:
        if last_binary[0] is None:
            return "No compiled binary available — call compile_oracle first."
        try:
            proc = subprocess.run(
                [str(last_binary[0].absolute())],
                capture_output=True, text=True, cwd=str(oracle_out),
                timeout=20,
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            return (
                "FAILED (timed out after 20s — the binary did not exit). "
                "Likely an infinite loop or a blocking call. Partial output:\n"
                + partial[-2000:]
            )
        output = (proc.stdout + proc.stderr).strip()
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        result = f"{status}\n{output}"
        has_output_lines = any(l.startswith("OUTPUT") for l in lines)
        if proc.returncode == 0 and not has_output_lines:
            result += (
                "\n\nWARNING: no OUTPUT lines detected. The function ran but "
                "produced no observable output — it likely returned early or its "
                "core logic was never reached. Re-read the function source, "
                "identify what conditions are needed to reach the main body, "
                "set up those inputs, add OUTPUT prints that capture the side "
                "effects, and call compile_oracle again."
            )
        return result

    def compiled_ok() -> bool:
        return last_binary[0] is not None

    return compile_handler, run_handler, compiled_ok


def _dispatch(
    conn: sqlite3.Connection,
    name: str,
    args: dict,
    pdb_path: Path | None = None,
) -> str:
    if name == "read_file":
        return _tool_read_file(args["path"], args.get("offset", 0), args.get("limit", 300))
    if name == "lookup_function":
        return _tool_lookup_function(conn, args["qualified_name"])
    if name == "lookup_type":
        return _tool_lookup_type(conn, args["name"])
    if name == "list_methods":
        return _tool_list_methods(conn, args["class_name"])
    if name == "get_callers":
        return _tool_get_callers(conn, args["qualified_name"])
    if name == "find_header":
        return _tool_find_header(conn, args["name"])
    if name == "resolve_includes":
        return _tool_resolve_includes(args["code"])
    if name == "search_functions":
        return _tool_search_functions(conn, args["name_fragment"])
    if name == "grep_codebase":
        return _tool_grep_codebase(args["pattern"], args.get("glob"))
    if name == "inspect_pdb":
        return _tool_inspect_pdb(args.get("chain"), pdb_path=pdb_path)
    if name == "get_base_classes":
        return _tool_get_base_classes(conn, args["name"])
    if name == "find_symbol":
        return _tool_find_symbol(args["symbol"])
    if name == "leave_note":
        return _tool_leave_note(args["topic"], args["question"])
    valid = (
        "read_file, lookup_function, lookup_type, list_methods, get_callers, "
        "find_header, resolve_includes, search_functions, grep_codebase, "
        "inspect_pdb, get_base_classes, find_symbol, leave_note, "
        "compile_oracle, run_oracle"
    )
    return (
        f"ERROR: '{name}' is not a valid tool. "
        f"Valid tools: {valid}. "
        "Do not retry this call — use one of the listed tools instead."
    )


# ── Ollama chat API ───────────────────────────────────────────────────────────

# ── agent loop ────────────────────────────────────────────────────────────────

def generate_with_agent(
    conn: sqlite3.Connection,
    function_qname: str,
    model: str,
    oracle_out: Path | None = None,
    verbose: bool = False,
    pdb_file: str = "example.pdb",
    pdb_note: str = "",
) -> tuple[str | None, str]:
    """Run the agentic oracle generation loop.

    Returns (oracle_code, trace_text).
    oracle_code is None if the function wasn't found or the model produced nothing.
    trace_text is a human-readable log of every tool call and result.

    pdb_file: filename (relative to test-data/) of the example PDB to use.
    pdb_note: injected into the prompt when the choice was ambiguous, listing
              all available PDB files so the LLM can choose.
    """
    fn = get_function(conn, function_qname)
    if not fn:
        return None, "Function not found in DB."

    # Resolve caller class context once so we can tag in-class members.
    target_class = function_qname.rsplit("::", 1)[0] if "::" in function_qname else None
    target_is_private = fn["access"] in ("private", "protected")

    parts: list[str] = []

    # If the target function itself is private/protected, emit an upfront
    # directive so the agent doesn't waste turns discovering it can't call it.
    if target_is_private:
        parts.append(
            f"## Note: `{function_qname}` is `{fn['access']}`\n\n"
            f"oracle.cc is compiled with `-fno-access-control`, so you can call "
            f"this {fn['access']} member and read/write any {fn['access']} member "
            f"variables directly. Just call the function as if it were public."
        )

    parts.append(f"## Function to observe: `{function_qname}`")
    meta = [f"- **File:** `{fn['file']}`"]
    if fn["comment"]:
        meta.append(f"- **Doc:** {fn['comment']}")
    parts.append("\n".join(meta))
    parts.append(f"```cpp\n{(fn['source_code'] or '(no source)').rstrip()}\n```")

    # Always include the MMDB Manager usage snippet — ReadCoorFile is an
    # external library not in the DB, so the agent can't find it via tools.
    parts.append("## MMDB usage")
    parts.append(f"```cpp\n{MMDB_MANAGER_SNIPPET.rstrip()}\n```")

    # Inject callers up-front so the agent sees real usage before reasoning
    # about unknown parameter types.
    callers = get_callers_with_source(conn, fn["id"], limit=3)
    if callers:
        parts.append("## Example callers")
        parts.append(
            "_Reference only. oracle.cc is compiled with `-fno-access-control`, "
            "so it can access any private/protected members the same way these "
            "in-class callers do._"
        )
        for i, c in enumerate(callers, 1):
            rel = c["file"].replace(PROJECT_ROOT + "/", "")
            caller_cls = get_containing_class(conn, c["qualified_name"])
            caller_cls_qname = caller_cls["qualified_name"] if caller_cls else None
            in_class = target_class is not None and caller_cls_qname == target_class
            access_note = (
                "in-class member — accesses private state directly (oracle.cc can do the same via -fno-access-control)"
                if in_class else
                "external caller"
            )
            parts.append(f"### Caller {i}/{len(callers)}: `{c['qualified_name']}`")
            caller_meta = [f"- **File:** `{rel}`"]
            if caller_cls_qname:
                caller_meta.append(f"- **Class:** `{caller_cls_qname}`")
            caller_meta.append(f"- **Access:** {access_note}")
            if c["comment"]:
                caller_meta.append(f"- **Doc:** {c['comment']}")
            parts.append("\n".join(caller_meta))

            fields = caller_class_fields(conn, c["qualified_name"])
            if fields:
                parts.append(f"```cpp\n{fields.rstrip()}\n```")
            parts.append(f"```cpp\n{c['source_code'].rstrip()}\n```")

    # Load any curated notes (questions + answers) left by previous runs.
    notes_context = _load_notes()
    if notes_context:
        parts.append("## Domain notes")
        parts.append(f"```\n{notes_context.rstrip()}\n```")

    # Attach the curated construction snippet for the containing class if one exists.
    selected_pdb_path = TEST_DATA_DIR / pdb_file
    if target_class:
        override = _load_override(target_class, pdb_path=str(selected_pdb_path))
        if override:
            parts.append(f"## How to construct `{target_class}`")
            parts.append(f"```cpp\n{override.rstrip()}\n```")

    user_content = "\n\n".join(parts)

    system_prompt = make_agent_system_prompt(str(selected_pdb_path), pdb_note)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    if oracle_out is not None:
        oracle_out.mkdir(parents=True, exist_ok=True)
        (oracle_out / "prompt.txt").write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n"
            f"=== USER ===\n{user_content}\n"
        )

    trace_lines = _TraceWriter(
        (oracle_out / "agent_trace.txt") if oracle_out is not None else None
    )
    trace_lines.append(f"=== AGENT TRACE: {function_qname} ===\n")
    trace_lines.append(f"[user]\n{textwrap.indent(user_content, '  ')}\n")

    compile_handler, run_handler, compiled_ok = (
        _make_oracle_tool_handlers(oracle_out) if oracle_out else (None, None, lambda: False)
    )

    def dispatch(name: str, args: dict) -> str:
        if name == "compile_oracle" and compile_handler:
            code = args.get("code")
            if code is None:
                return "Error: compile_oracle called without a 'code' argument."
            return compile_handler(code)
        if name == "run_oracle" and run_handler:
            if not compiled_ok() and last_draft[0]:
                compile_msg = compile_handler(last_draft[0])
                if not compiled_ok():
                    return (
                        "run_oracle: no compiled binary was available, so I "
                        "auto-compiled the most recent draft you provided. "
                        "Compilation failed — fix the errors below and call "
                        "compile_oracle with corrected code:\n" + compile_msg
                    )
                run_msg = run_handler()
                return (
                    "(auto-compiled latest draft before running — "
                    f"{compile_msg.splitlines()[0] if compile_msg else 'compile ok'})\n"
                    + run_msg
                )
            return run_handler()
        if name in ("compile_oracle", "run_oracle"):
            return "Tool unavailable — no output directory configured."
        return _dispatch(conn, name, args, pdb_path=selected_pdb_path)

    tools = ORACLE_TOOLS if oracle_out else TOOLS
    oracle_code: str | None = None
    last_draft: list[str | None] = [None]    # any cpp draft we've ever seen (fallback)
    call_counts: dict[str, int] = {}         # repeat-call detection
    tool_cache: dict[str, str] = {}          # memoize read-only lookups within a session
    REPEAT_LIMIT = 3
    no_compile_warned = [False]              # one-shot guard for the no-compile nudge
    degen_recoveries = [0]                   # how many times we've nudged out of a degen loop
    # Tools whose output may differ across calls or which have side effects —
    # never serve from cache.
    NO_CACHE = {"compile_oracle", "run_oracle", "leave_note"}

    def _save_draft(code: str) -> None:
        if code and len(code) > 100 and "#include" in code:
            last_draft[0] = code

    def _run_tool_calls(tool_calls: list, label: str = "") -> list[dict]:
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

            # Snapshot drafts — any code passed to compile_oracle is worth keeping.
            if name == "compile_oracle" and isinstance(args.get("code"), str):
                _save_draft(args["code"])

            # Repeat-call detection: hash on (name, non-code args) so compile
            # attempts with different code aren't counted as repeats.
            hash_args = {k: v for k, v in args.items() if k != "code"}
            key = f"{name}:{json.dumps(hash_args, sort_keys=True)}"
            call_counts[key] = call_counts.get(key, 0) + 1

            # Memoize read-only lookups: identical (tool, args) returns the
            # prior result with a "(cached)" header so the agent recognises
            # it has already seen this. Saves dispatch cost AND, more
            # importantly, signals "stop asking the same thing".
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

            if call_counts[key] > REPEAT_LIMIT and name not in ("compile_oracle", "run_oracle"):
                nudge = (
                    f"You have already called {name} with these arguments "
                    f"{call_counts[key]} times. Stop repeating — use the information "
                    "you already have. Draft oracle.cc and call compile_oracle now."
                )
                trace_lines.append(f"  → [repeat-intercept × {call_counts[key]}] {name}({json.dumps(hash_args)})")
                trace_lines.append(textwrap.indent(nudge, "      ") + "\n")
                trace_lines.append_call(f"[repeat-intercept × {call_counts[key]}] {name}({json.dumps(hash_args)})")
                results.append({"role": "tool", "content": nudge})
                continue

            if verbose:
                print(f"  tool: {name}({args})")
            result = dispatch(name, args)
            result_lines = result.splitlines()
            if len(result_lines) > 150:
                result = "\n".join(result_lines[:150]) + f"\n... ({len(result_lines) - 150} more lines)"
            trace_lines.append(f"  → {name}({json.dumps(args)})")
            trace_lines.append(textwrap.indent(result, "      ") + "\n")
            trace_lines.append_call(f"{name}({json.dumps(args)})")
            if name not in NO_CACHE:
                tool_cache[key] = result
            results.append({"role": "tool", "content": result})
        return results

    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", content, re.DOTALL)
        code = m.group(1).strip() if m else content.strip()
        _save_draft(code)
        return code

    def _progress(label: str, tool_calls: list) -> None:
        if tool_calls:
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
            print(f"\r  [oracle] {label} → {names}", flush=True)
        else:
            print(f"\r  [oracle] {label} → done (final answer)", flush=True)

    for turn in range(20):
        print(f"  [oracle] turn {turn + 1}/20 ...", end="", flush=True)
        data = _chat(messages, model, tools)
        _log_llm_timing(data, stage="oracle", turn=turn + 1, verbose=verbose, trace_lines=trace_lines)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking", "") or ""
        assistant_content = msg.get("content",  "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                         "tool_calls": tool_calls})

        if thinking:
            if verbose:
                print(f"\n[thinking]\n{textwrap.indent(thinking, '  ')}\n")
            trace_lines.append(f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n")

        # Degenerate-thinking guard: pathologically repetitive thinking
        # means the model burned its output window on garbage without emitting
        # a tool call or final answer. Inject a targeted nudge and continue
        # the loop so the model can recover naturally with all tools intact.
        # After two failed recoveries, give up and let rescue handle it.
        degen, diag = _is_degenerate_thinking(thinking)
        if degen:
            degen_recoveries[0] += 1
            if degen_recoveries[0] > 2:
                print(f"\r  [oracle] turn {turn + 1}/20 → DEGENERATE — rescue", flush=True)
                trace_lines.append(f"[agent] {diag} — recovery exhausted, going to rescue.\n")
                break
            print(f"\r  [oracle] turn {turn + 1}/20 → DEGENERATE — nudging ({degen_recoveries[0]})", flush=True)
            trace_lines.append(
                f"[agent] {diag} — injecting recovery nudge "
                f"(attempt {degen_recoveries[0]}).\n"
            )
            if messages and messages[-1].get("role") == "assistant":
                messages[-1] = {"role": "assistant",
                                "content": "[thinking loop truncated]",
                                "tool_calls": []}
            template = (
                _DEGEN_RECOVERY_NUDGE_POST_COMPILE if compiled_ok()
                else _DEGEN_RECOVERY_NUDGE
            )
            recovery_msg = template.format(n=_DEGEN_RECOVERY_TURNS)
            messages.append({"role": "user", "content": recovery_msg})
            trace_lines.append(f"[recovery nudge]\n{textwrap.indent(recovery_msg, '  ')}\n")
            continue

        _progress(f"turn {turn + 1}/20", tool_calls)

        if not tool_calls:
            trace_lines.append(f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n")
            oracle_code = _extract_code(assistant_content)
            break

        trace_lines.append(f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]")
        messages.extend(_run_tool_calls(tool_calls))

        # One-shot "you haven't compiled anything" warning. Fires when the
        # threshold is reached without a single compile_oracle attempt — the
        # model is researching instead of testing.
        if (NO_COMPILE_AFTER and not no_compile_warned[0]
                and (turn + 1) >= NO_COMPILE_AFTER
                and not any(k.startswith("compile_oracle:") for k in call_counts)):
            messages.append({"role": "user", "content": _ORACLE_NO_COMPILE_NUDGE})
            trace_lines.append(f"[no-compile nudge — turn {turn + 1}]\n{textwrap.indent(_ORACLE_NO_COMPILE_NUDGE, '  ')}\n")
            no_compile_warned[0] = True

        if NUDGE_EVERY_N_TURNS and (turn + 1) % NUDGE_EVERY_N_TURNS == 0:
            messages.append({"role": "user", "content": _ORACLE_NUDGE})
            trace_lines.append(f"[nudge — turn {turn + 1}]\n{textwrap.indent(_ORACLE_NUDGE, '  ')}\n")

    else:
        # All 20 turns used — ask once if more time is needed.
        trace_lines.append("[agent] Turn limit reached — asking for extension.\n")
        messages.append({"role": "user",
                         "content": _EXTENSION_PROMPT.format(n=_EXTENSION_TURNS)})

        print(f"  [oracle] extension check ...", end="", flush=True)
        data = _chat(messages, model, tools)
        _log_llm_timing(data, stage="oracle", turn="extension", verbose=verbose, trace_lines=trace_lines)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking", "") or ""
        assistant_content = msg.get("content",  "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                         "tool_calls": tool_calls})

        if thinking:
            trace_lines.append(f"[thinking — extension]\n{textwrap.indent(thinking, '  ')}\n")

        if not tool_calls:
            print(f"\r  [oracle] extension check → declined", flush=True)
            trace_lines.append(f"[assistant — final (declined extension)]\n{textwrap.indent(assistant_content, '  ')}\n")
            oracle_code = _extract_code(assistant_content)
        else:
            print(f"\r  [oracle] extension check → granted", flush=True)
            trace_lines.append(f"[agent] Extension granted ({_EXTENSION_TURNS} more turns).\n")
            messages.extend(_run_tool_calls(tool_calls))

            for ext_turn in range(_EXTENSION_TURNS):
                print(f"  [oracle] ext turn {ext_turn + 1}/{_EXTENSION_TURNS} ...", end="", flush=True)
                data = _chat(messages, model, tools)
                _log_llm_timing(data, stage="oracle", turn=f"ext {ext_turn + 1}", verbose=verbose, trace_lines=trace_lines)
                msg  = data.get("message", {})
                tool_calls        = msg.get("tool_calls") or []
                thinking          = msg.get("thinking", "") or ""
                assistant_content = msg.get("content",  "") or ""
                messages.append({"role": "assistant", "content": assistant_content,
                                 "tool_calls": tool_calls})

                if thinking:
                    trace_lines.append(f"[thinking — ext turn {ext_turn + 1}]\n{textwrap.indent(thinking, '  ')}\n")

                _progress(f"ext turn {ext_turn + 1}/{_EXTENSION_TURNS}", tool_calls)

                if not tool_calls:
                    trace_lines.append(f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n")
                    oracle_code = _extract_code(assistant_content)
                    break

                trace_lines.append(f"[assistant — ext turn {ext_turn + 1}, {len(tool_calls)} tool call(s)]")
                messages.extend(_run_tool_calls(tool_calls))

                if (NO_COMPILE_AFTER and not no_compile_warned[0]
                        and not any(k.startswith("compile_oracle:") for k in call_counts)):
                    messages.append({"role": "user", "content": _ORACLE_NO_COMPILE_NUDGE})
                    trace_lines.append(f"[no-compile nudge — ext turn {ext_turn + 1}]\n{textwrap.indent(_ORACLE_NO_COMPILE_NUDGE, '  ')}\n")
                    no_compile_warned[0] = True

                if NUDGE_EVERY_N_TURNS and (ext_turn + 1) % NUDGE_EVERY_N_TURNS == 0:
                    messages.append({"role": "user", "content": _ORACLE_NUDGE})
                    trace_lines.append(f"[nudge — ext turn {ext_turn + 1}]\n{textwrap.indent(_ORACLE_NUDGE, '  ')}\n")
            else:
                trace_lines.append("[agent] Extension exhausted without final answer.\n")

    # ── Rescue: recover a draft if the final turn produced nothing usable ────
    def _is_usable(code: str | None) -> bool:
        return bool(code and len(code) > 100 and "#include" in code)

    if not _is_usable(oracle_code):
        if _is_usable(last_draft[0]):
            trace_lines.append("[rescue] final turn had no code block — reusing last seen draft.\n")
            oracle_code = last_draft[0]
        else:
            trace_lines.append("[rescue] no draft found — one-shot asking for final output.\n")
            messages.append({"role": "user", "content": (
                "STOP. Do not call any tools. Output your best attempt at oracle.cc "
                "NOW inside a single ```cpp block. This is your last chance — "
                "partial or imperfect code is better than nothing."
            )})
            try:
                data = _chat(messages, model, tools=[])
                _log_llm_timing(data, stage="oracle", turn="rescue", verbose=verbose, trace_lines=trace_lines)
                assistant_content = (data.get("message") or {}).get("content") or ""
                trace_lines.append(f"[rescue — response]\n{textwrap.indent(assistant_content, '  ')}\n")
                rescued = _extract_code(assistant_content)
                if _is_usable(rescued):
                    oracle_code = rescued
                elif _is_usable(last_draft[0]):
                    oracle_code = last_draft[0]
            except (urllib.error.URLError, json.JSONDecodeError) as e:
                trace_lines.append(f"[rescue] failed: {e}\n")
                if _is_usable(last_draft[0]):
                    oracle_code = last_draft[0]

    # Notify about any notes that still need an answer.
    unanswered = _unanswered_notes()
    if unanswered:
        trace_lines.append("\n[notes] Unanswered questions — please fill in:")
        for p in unanswered:
            trace_lines.append(f"  {p}")
        if verbose:
            print("\n*** Unanswered notes — please fill in an answer: ***")
            for p in unanswered:
                print(f"  {p}")

    text = trace_lines.text()
    trace_lines.close()
    return oracle_code, text

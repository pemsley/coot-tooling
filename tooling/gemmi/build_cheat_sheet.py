#!/usr/bin/env python3
"""
Build/extend the MMDB→gemmi cheat sheet by querying the LLM for each public
method on the core MMDB hierarchy classes that isn't already covered.

Usage:
  python3 -m tooling.gemmi.build_cheat_sheet [--model MODEL] [--dry-run] [--db PATH]

Candidates are read directly from code_graph.db (public CXX_METHOD rows for
mmdb::Manager / Model / Chain / Residue / Atom). Each unmapped method gets its
own mini agentic loop where the LLM can call grep_codebase, lookup_type,
read_file, and find_symbol to look up the real gemmi API before answering.

If the model is uncertain (CONFIDENCE: LOW), you are prompted to type the
gemmi equivalent yourself. Confirmed mappings are injected into the Key
MMDB→gemmi accessor map section of GEMMI_CHEAT_SHEET in
tooling/gemmi/agent.py.

Results are cached in --cache-dir (default: tooling/gemmi/.cheat_cache/) so
interrupted sessions can be resumed without re-querying the LLM.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ..db import DB_PATH
from ..oracle.agent import (
    _tool_read_file,
    _tool_grep_codebase,
    _tool_lookup_type,
    _tool_find_symbol,
    _is_degenerate_thinking,
)
from ..llm import SAMPLING_PARAMS, OLLAMA_CONTEXT_TOKENS, OLLAMA_MAX_TOKENS
from ..ollama import chat_url
from ..oracle.compile import GEMMI_INCLUDE
from .agent import GEMMI_CHEAT_SHEET

# ---------------------------------------------------------------------------
# Core MMDB classes we want to map
# ---------------------------------------------------------------------------
CORE_CLASSES = [
    "mmdb::Manager",
    "mmdb::Model",
    "mmdb::Chain",
    "mmdb::Residue",
    "mmdb::Atom",
    "mmdb::CoorManager",
    "mmdb::SelManager",
    "mmdb::Cryst",
    "mmdb::SymOps",
]

# Method names too internal/obscure to bother mapping.
_SKIP_PATTERNS = re.compile(
    r"^(ConvertPDB|ConvertCIF|ConvertDB|Write|MakeAtomName|PutCIF"
    r"|MakePDB|isInSelection|SetShift|PutUniqueID|GetUniqueID"
    r"|UDData|Copy\b|Delete\b|FreeMemory|InitEntry|CheckID"
    r"|GetIndex|SetFlag|ResetFlag|MakeChain|RegisterModel"
    r"|AddAtom|AddChain|AddResidue|RemoveAtom|DetachAtom|InsertAtom"
    r"|Trim|Clean|Sort|Reorder)"
)

_DEGEN_CHECK_INTERVAL = 2000


def _get_candidates(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (qualified_name, representative_display_name) for public methods.

    Overloads share a qualified_name; we keep the shortest display_name as
    the representative so the LLM sees a clean signature.
    """
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    for cls in CORE_CLASSES:
        rows = conn.execute(
            """
            SELECT qualified_name, display_name
            FROM functions
            WHERE qualified_name LIKE ? || '::%'
              AND kind = 'CXX_METHOD'
              AND (access = 'public' OR access IS NULL)
            ORDER BY length(display_name), display_name
            """,
            (cls,),
        ).fetchall()
        for qn, dn in rows:
            if qn in seen:
                continue
            method_name = qn.rsplit("::", 1)[-1]
            # if _SKIP_PATTERNS.match(method_name):
            #     continue
            seen.add(qn)
            results.append((qn, dn))
    return results


# ---------------------------------------------------------------------------
# Cheat-sheet helpers
# ---------------------------------------------------------------------------
_AGENT_PY = Path(__file__).parent / "agent.py"
_DEFAULT_CACHE_DIR = Path(__file__).parent / ".cheat_cache"

# The last pre-existing accessor-map line — new entries are inserted after it.
_INSERT_AFTER = "  residue->GetNumberOfAtoms()     → residue.atoms.size()"


def _already_covered(qualified_name: str) -> bool:
    """Return True if the method name already appears in the cheat sheet."""
    method_name = qualified_name.rsplit("::", 1)[-1]
    return bool(method_name) and method_name in GEMMI_CHEAT_SHEET


_CLASS_RECEIVER = {
    "mmdb::Manager":     "mol",
    "mmdb::Model":       "model",
    "mmdb::Chain":       "chain",
    "mmdb::Residue":     "residue",
    "mmdb::Atom":        "atom",
    "mmdb::CoorManager": "mol",
    "mmdb::SelManager":  "mol",
    "mmdb::Cryst":       "cryst",
    "mmdb::SymOps":      "symops",
}


def _mmdb_expr(qualified_name: str, display_name: str) -> str:
    """Build 'receiver->MethodName(params)' from DB fields."""
    cls = qualified_name.rsplit("::", 1)[0]
    receiver = _CLASS_RECEIVER.get(cls, "obj")
    method_name = qualified_name.rsplit("::", 1)[-1]
    idx = display_name.find(method_name)
    sig = display_name[idx:] if idx >= 0 else display_name
    return f"{receiver}->{sig}"


def _format_entry(mmdb_expr: str, gemmi_expr: str, note: str) -> str:
    left = f"  {mmdb_expr}"
    mid = f"{left:<40} → {gemmi_expr}"
    if note:
        return f"{mid:<70} // {note}"
    return mid


def _inject_entries(new_lines: list[str]) -> None:
    text = _AGENT_PY.read_text()
    if _INSERT_AFTER not in text:
        sys.exit(
            f"ERROR: insertion sentinel not found in {_AGENT_PY}:\n  {_INSERT_AFTER!r}"
        )
    block = "\n".join(new_lines)
    updated = text.replace(_INSERT_AFTER, _INSERT_AFTER + "\n" + block, 1)
    _AGENT_PY.write_text(updated)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, qualified_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", qualified_name)
    return cache_dir / f"{safe}.json"


def _load_cache(cache_dir: Path, qualified_name: str) -> dict | None:
    """Return the cached result dict, or None if not cached."""
    p = _cache_path(cache_dir, qualified_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(cache_dir: Path, qualified_name: str, entry: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, qualified_name)
    p.write_text(json.dumps(entry, indent=2))


# ---------------------------------------------------------------------------
# Streaming Ollama chat
# ---------------------------------------------------------------------------

def _stream_chat(
    messages: list[dict],
    model: str,
    tools: list[dict],
    verbose: bool = False,
) -> dict:
    """Like oracle._chat but streams content tokens to stdout as they arrive.

    In verbose mode, thinking tokens are also printed with a dim prefix.
    Tool calls are always printed when they arrive.
    """
    ollama_options = {
        "num_ctx": OLLAMA_CONTEXT_TOKENS,
        "num_predict": OLLAMA_MAX_TOKENS,
        **SAMPLING_PARAMS,
    }
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "tools":    tools,
        "stream":   True,
        "think":    True,
        "options":  ollama_options,
    }).encode()
    req = urllib.request.Request(
        chat_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    accumulated_thinking   = ""
    accumulated_content    = ""
    accumulated_tool_calls: list = []
    last_degen_check       = 0
    in_thinking            = False
    in_content             = False

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            msg   = chunk.get("message", {})

            thinking_tok = msg.get("thinking") or ""
            content_tok  = msg.get("content")  or ""

            if thinking_tok:
                accumulated_thinking += thinking_tok
                if verbose:
                    if not in_thinking:
                        sys.stdout.write("  \033[2m[thinking] ")
                        in_thinking = True
                    sys.stdout.write(thinking_tok)
                    sys.stdout.flush()

            if content_tok:
                if in_thinking:
                    sys.stdout.write("\033[0m\n")
                    in_thinking = False
                if not in_content:
                    sys.stdout.write("  ")
                    in_content = True
                accumulated_content += content_tok
                sys.stdout.write(content_tok)
                sys.stdout.flush()

            if msg.get("tool_calls"):
                accumulated_tool_calls = msg["tool_calls"]

            if chunk.get("done"):
                break

            new_len = len(accumulated_thinking)
            if new_len - last_degen_check >= _DEGEN_CHECK_INTERVAL:
                last_degen_check = new_len
                if _is_degenerate_thinking(accumulated_thinking)[0]:
                    break

    # Tidy up the line after streaming
    if in_thinking or in_content:
        sys.stdout.write("\033[0m\n")
        sys.stdout.flush()

    return {
        "message": {
            "role":       "assistant",
            "thinking":   accumulated_thinking,
            "content":    accumulated_content,
            "tool_calls": accumulated_tool_calls,
        }
    }


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------
_GEMMI_SEARCH_ROOT = str(Path(GEMMI_INCLUDE).parent)  # .../include

CHEAT_SHEET_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "grep_codebase",
            "description": (
                "Search the gemmi header tree for a regex pattern. "
                "Returns matching lines with file path and line number. "
                "Use this to find struct/class members, function signatures, "
                "or constant values in the gemmi API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "glob":    {"type": "string",
                                "description": "Restrict to files matching this glob, e.g. '*.hpp'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_type",
            "description": (
                "Return the class/struct definition and member list for a gemmi type. "
                "Accepts short names like 'Chain' or qualified names like 'gemmi::Chain'."
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
            "name": "read_file",
            "description": (
                "Read a gemmi header file. Returns up to 300 lines. "
                "Use offset/limit to page through large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string",  "description": "Absolute file path"},
                    "offset": {"type": "integer", "description": "First line (0-based)"},
                    "limit":  {"type": "integer", "description": "Max lines to return"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": (
                "Find a constant, enum value, typedef, or function by name in gemmi headers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
    },
]

_SYSTEM = f"""\
You are a C++ expert in both the MMDB2 library (used in CCP4/Coot) and the
gemmi structural-biology library.

Your job: given one MMDB API method, find the equivalent gemmi idiom by
searching the gemmi headers with the provided tools, then output your answer.

## Current cheat sheet (already confirmed mappings — do not repeat these):

{GEMMI_CHEAT_SHEET}

## Gemmi headers are at:
  {GEMMI_INCLUDE}/

## Instructions

1. Use grep_codebase, lookup_type, read_file, or find_symbol to look up the
   real gemmi API. Do not guess — verify against the headers.
2. When you have enough information, stop calling tools and output ONLY this
   exact 4-line block (no prose, no extra lines):
3. DO NOT keep looking once you find a suitable answer, output it immediately. 
4. If there is no GEMMI equivalent of an MMDB call, think about how it could be achieved in an alternate route.

MMDB: <the mmdb method as given>
GEMMI: <gemmi equivalent expression, or NO_EQUIVALENT if none exists>
CONFIDENCE: HIGH or LOW
NOTE: <one short note, or blank>

Rules for the answer block:
- HIGH = you confirmed the mapping from the headers.
- LOW  = you could not find a clear equivalent after searching.
- If the method has no gemmi equivalent (e.g. MMDB-selection internals),
  write NO_EQUIVALENT and HIGH.
- Keep GEMMI to one expression or one short line.
- Do NOT output anything after the 4-line block.
"""


def _dispatch_tool(name: str, args: dict, conn: sqlite3.Connection) -> str:
    if name == "grep_codebase":
        return _tool_grep_codebase(
            args.get("pattern", ""),
            args.get("glob"),
            extra_roots=[_GEMMI_SEARCH_ROOT],
        )
    if name == "lookup_type":
        return _tool_lookup_type(conn, args.get("name", ""))
    if name == "read_file":
        return _tool_read_file(
            args.get("path", ""),
            args.get("offset", 0),
            args.get("limit", 300),
        )
    if name == "find_symbol":
        return _tool_find_symbol(args.get("symbol", ""))
    return f"Unknown tool: {name}"


def _parse_answer(text: str) -> dict:
    result = {"gemmi": "", "confidence": "LOW", "note": ""}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("GEMMI:"):
            result["gemmi"] = line[len("GEMMI:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            val = line[len("CONFIDENCE:"):].strip().upper()
            result["confidence"] = val if val in ("HIGH", "LOW") else "LOW"
        elif line.startswith("NOTE:"):
            result["note"] = line[len("NOTE:"):].strip()
    return result


def _ask_llm(
    qualified_name: str,
    display_name: str,
    model: str,
    conn: sqlite3.Connection,
    verbose: bool = False,
    max_turns: int = 50,
) -> dict:
    """Run a mini agentic loop: LLM researches with tools then outputs answer."""
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Find the gemmi equivalent for this MMDB method:\n\n"
                f"  Class:   {qualified_name.rsplit('::', 1)[0]}\n"
                f"  Method:  {display_name}\n"
                f"  Fully qualified: {qualified_name}\n\n"
                "Use tools to look up the gemmi API, then output the 4-line answer block."
            ),
        },
    ]

    # Maps (tool_name, args_json) → result already returned, so we can detect
    # repeated identical calls and nudge the model instead of re-executing.
    call_cache: dict[tuple[str, str], str] = {}

    for turn in range(max_turns):
        if verbose:
            print(f"  [turn {turn + 1}/{max_turns}]")
        else:
            print(f"  querying LLM (turn {turn + 1}/{max_turns})...", end="\r", flush=True)

        response = _stream_chat(messages, model=model, tools=CHEAT_SHEET_TOOLS, verbose=verbose)
        msg = response.get("message", {})
        content: str    = msg.get("content", "").strip()
        tool_calls: list = msg.get("tool_calls") or []

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        if not tool_calls:
            print(" " * 60, end="\r")
            return _parse_answer(content)

        # Print and execute tool calls, detecting repeats
        tool_results: list[dict] = []
        for tc in tool_calls:
            fn      = tc.get("function", {})
            name    = fn.get("name", "")
            raw     = fn.get("arguments", {})
            args    = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            key     = (name, json.dumps(args, sort_keys=True))

            if key in call_cache:
                result_text = (
                    f"REPEATED CALL: you already called {name}({_fmt_args(args)}) "
                    f"and received the result shown earlier in this conversation. "
                    f"Do not call it again. Either try a different tool/query, "
                    f"or output your final 4-line answer block now."
                )
                marker = "\033[31m[REPEAT]\033[0m" if verbose else "[REPEAT]"
                print(f"  {marker} {name}({_fmt_args(args)})" + " " * 10)
            else:
                if verbose:
                    print(f"  \033[33m→ {name}({_fmt_args(args)})\033[0m")
                else:
                    print(f"  → {name}({_fmt_args(args)})" + " " * 20, end="\r", flush=True)

                result_text = _dispatch_tool(name, args, conn)
                call_cache[key] = result_text

                if verbose:
                    lines = result_text.splitlines()
                    excerpt = "\n".join(f"    {l}" for l in lines[:8])
                    if len(lines) > 8:
                        excerpt += f"\n    ... ({len(lines) - 8} more lines)"
                    print(excerpt)

            tool_results.append({
                "role":    "tool",
                "name":    name,
                "content": result_text[:4000],
            })
        messages.extend(tool_results)

    print(" " * 60, end="\r")
    last = messages[-1].get("content", "") if messages else ""
    return _parse_answer(last)


def _fmt_args(args: dict) -> str:
    """Format tool args as a short one-liner for display."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.6",
                        help="Ollama model name (default: qwen3.6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print new entries but do not modify agent.py")
    parser.add_argument("--db", default=str(DB_PATH),
                        help="Path to code_graph.db")
    parser.add_argument("--cls", metavar="CLASS", action="append",
                        help="Restrict to this MMDB class (can repeat). "
                             "E.g. --cls mmdb::Atom")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show thinking tokens and full tool results")
    parser.add_argument("--cache-dir", default=str(_DEFAULT_CACHE_DIR),
                        help="Directory for caching LLM results "
                             f"(default: {_DEFAULT_CACHE_DIR})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the cache and re-query the LLM for everything")
    parser.add_argument("--retry-skipped", action="store_true",
                        help="Re-query only methods previously cached as skipped "
                             "(clears their cache entry so they are processed fresh)")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactive: auto-accept HIGH confidence, "
                             "auto-skip LOW confidence / no-equivalent. "
                             "Prints a report of all skipped methods at the end.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    candidates = _get_candidates(conn)
    if args.cls:
        allowed = set(args.cls)
        candidates = [
            (qn, dn) for qn, dn in candidates
            if qn.rsplit("::", 2)[0] in allowed or qn.rsplit("::", 1)[0] in allowed
        ]

    todo = [(qn, dn) for qn, dn in candidates if not _already_covered(qn)]
    skipped = len(candidates) - len(todo)

    if args.retry_skipped:
        previously_skipped = set()
        for qn, dn in todo:
            cached = _load_cache(cache_dir, qn)
            if cached is not None and cached.get("skipped"):
                previously_skipped.add(qn)
        todo = [(qn, dn) for qn, dn in todo if qn in previously_skipped]
        # Clear skipped cache entries so they are re-queried fresh
        for qn, _ in todo:
            _cache_path(cache_dir, qn).unlink(missing_ok=True)
        print(f"Candidates: {len(candidates)}  |  already covered: {skipped}  |  retrying skipped: {len(todo)}")
    else:
        print(f"Candidates: {len(candidates)}  |  already covered: {skipped}  |  to map: {len(todo)}")

    new_entries: list[str] = []
    missing: list[tuple[str, str, str]] = []  # (qualified_name, display_name, reason)

    for idx, (qn, dn) in enumerate(todo, 1):
        cls = qn.rsplit("::", 1)[0]
        print(f"\n[{idx}/{len(todo)}] {cls}  ::  {dn}")

        # --- cache hit ---
        if not args.no_cache:
            cached = _load_cache(cache_dir, qn)
            if cached is not None:
                if cached.get("skipped"):
                    print("  [cache] skipped previously.")
                    missing.append((qn, dn, "skipped (cached)"))
                    continue
                gemmi_expr = cached.get("gemmi", "")
                note       = cached.get("note", "")
                if gemmi_expr:
                    line = _format_entry(_mmdb_expr(qn, dn), gemmi_expr, note)
                    print(f"  [cache] {line}")
                    new_entries.append(line)
                    continue

        # --- LLM query ---
        try:
            result = _ask_llm(qn, dn, args.model, conn, verbose=args.verbose)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            print(f"  WARNING: LLM error ({exc}).")
            result = {"gemmi": "", "confidence": "LOW", "note": ""}

        gemmi_expr = result.get("gemmi", "")
        note       = result.get("note", "")
        confidence = result.get("confidence", "LOW")

        # --- NO_EQUIVALENT ---
        if gemmi_expr == "NO_EQUIVALENT":
            reason = f"no gemmi equivalent" + (f": {note}" if note else "")
            print(f"  LLM says: {reason}")
            _save_cache(cache_dir, qn, {"qualified_name": qn, "skipped": True, "note": note})
            missing.append((qn, dn, reason))
            if note:
                line = _format_entry(_mmdb_expr(qn, dn), "NO_EQUIVALENT", note)
                print(f"  Adding (no-equiv note): {line}")
                new_entries.append(line)
            continue

        # --- HIGH confidence: auto-accept ---
        if confidence == "HIGH" and gemmi_expr:
            display = gemmi_expr + (f"  // {note}" if note else "")
            print(f"  LLM (HIGH): {display}")

        # --- LOW confidence ---
        if not gemmi_expr or confidence == "LOW":
            if gemmi_expr:
                print(f"  LLM (LOW):  {gemmi_expr}" + (f"  // {note}" if note else ""))
            if args.auto:
                print("  --auto: skipping LOW confidence.")
                _save_cache(cache_dir, qn, {"qualified_name": qn, "skipped": True, "note": note})
                reason = f"LOW confidence (LLM suggestion: {gemmi_expr or 'none'})"
                if note:
                    reason += f" [note (LOW): {note}]"
                missing.append((qn, dn, reason))
                continue
            user_in = input("  Type gemmi equivalent (or Enter to skip): ").strip()
            if not user_in:
                print("  Skipping.")
                _save_cache(cache_dir, qn, {"qualified_name": qn, "skipped": True, "note": note})
                reason = "skipped by user"
                if note:
                    reason += f" [note (LOW): {note}]"
                missing.append((qn, dn, reason))
                continue
            if " // " in user_in:
                gemmi_expr, note = user_in.split(" // ", 1)
            else:
                gemmi_expr, note = user_in, ""

        if not gemmi_expr:
            continue

        # --- save and record ---
        _save_cache(cache_dir, qn, {
            "qualified_name": qn,
            "display_name":   dn,
            "gemmi":          gemmi_expr,
            "note":           note,
            "skipped":        False,
        })
        line = _format_entry(_mmdb_expr(qn, dn), gemmi_expr, note)
        print(f"  Adding: {line}")
        new_entries.append(line)

    conn.close()

    # --- end-of-run report ---
    if missing:
        print(f"\n{'='*60}")
        print(f"MISSING / SKIPPED ({len(missing)}):")
        for m_qn, m_dn, reason in missing:
            print(f"  {m_qn:<50}  [{reason}]")
        print("=" * 60)

    if not new_entries:
        print("\nNo new entries to add.")
        return

    print(f"\n{'='*60}")
    print(f"New entries ({len(new_entries)}):")
    for line in new_entries:
        print(f"  {line}")
    print("=" * 60)

    if args.dry_run:
        print("\n--dry-run: agent.py was NOT modified.")
        return

    _inject_entries(new_entries)
    print(f"\nInjected {len(new_entries)} new entries into {_AGENT_PY}.")


if __name__ == "__main__":
    main()

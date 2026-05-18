"""Agentic Google Test generation — model calls tools to resolve headers,
look up types, and iteratively compile its draft before finalising."""
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
    _TraceWriter,
    _chat,
    _log_llm_timing,
    _is_degenerate_thinking,
    NUDGE_EVERY_N_TURNS,
    NO_COMPILE_AFTER,
)

# Format-reminder nudge (injected every NUDGE_EVERY_N_TURNS turns).
_TEST_NUDGE = (
    "Reminder: when you stop calling tools, your final reply must be a "
    "single ```cpp fenced block containing the complete test.cc. "
    "If you have a working draft, call write_and_compile_test now to validate it."
)

_TEST_NO_COMPILE_NUDGE = (
    "WARNING: you have not attempted write_and_compile_test yet. "
    "Stop researching and DRAFT your best test.cc now, then call "
    "write_and_compile_test. The compiler's error messages are far more useful "
    "than further speculation. Failures are expected — you have multiple "
    "retries to fix them. Action over analysis."
)
from ..oracle.notes import load_notes, render_notes_for_prompt
from ..oracle.coverage import load_coverage, render_for_prompt as render_coverage_for_prompt
from ..oracle.runner.results import OracleResult
from .compile import MAX_COMPILE_ATTEMPTS, compile_test_cc, run_test_binary

TEST_SYSTEM_PROMPT = """\
You are converting a C++ oracle program into a Google Test suite (test.cc).

# The oracle's printed values are the ground truth
The numbers, strings, and sizes the oracle printed ARE the correct
answers. NEVER edit an expected literal to make a failing test pass — if
an assertion fails, the bug is in your setup or accessor, not the
expected value. Fix the test, not the oracle.

# Structure
1. Copy the oracle's setup (PDB/MTZ load, object construction, the call
   itself) verbatim. The only changes from oracle.cc are: remove the
   INPUT/OUTPUT cout lines, add assertions in their place.
2. Wrap ALL cases from the oracle in ONE block:
       TEST(OracleTest, <FunctionName>) { ... }
   Use inner `{ ... }` scopes (or a `// case: edge` comment) to label
   each case. Single TEST block keeps the binary small and makes
   diffing against the gemmi test trivial — do not split into multiple
   TESTs.
3. Do NOT write `main()` — gtest_main is linked. Adding `main()` causes
   a link error.

# Choosing assertions
- Integer / bool / enum  → EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE
- Float / double         → EXPECT_NEAR(actual, expected, 1e-4)
                           (use the oracle's printed precision; never
                           EXPECT_EQ on floats)
- Pointer null-ness      → EXPECT_NE(p, nullptr) / EXPECT_EQ(p, nullptr)
- Large strings (e.g. PDB dumps) → EXPECT_FALSE(s.empty()),
                           a size range, and
                           EXPECT_NE(s.find("HEADER"), std::string::npos).
                           Never hardcode byte counts.
- Void function, no observable mutation → EXPECT_NO_THROW around the call.


# Available tools (use ONLY these — do not invent tool names)
- read_file              — read a C++ source or header file
- lookup_function        — get source + docs for a function by qualified name
- lookup_type            — get class/struct definition and method list
- list_methods           — list all method signatures in a class
- get_callers            — find functions that call a given function
- find_header            — resolve a type/function name to its header path
- resolve_includes       — check which #includes are needed for a code draft
- search_functions       — find functions by partial name
- grep_codebase          — text search across the source tree
- get_base_classes       — list base classes of a type
- find_symbol            — locate a symbol in header files
- write_and_compile_test — write COMPLETE test.cc and compile+run it in one shot
- run_test               — re-run the last successfully compiled test binary
- get_compile_errors     — return full (untruncated) compiler output
- read_test              — read current test.cc from disk
- patch_test             — apply a targeted old→new replacement to test.cc and recompile
                           (prefer this over write_and_compile_test for small fixes after first write)

# Workflow — terminal condition
You are done when write_and_compile_test's response contains "All tests PASSED"
AND you have emitted the final program in one ```cpp fenced block.

- Run resolve_includes on your draft before the first write_and_compile_test.
- write_and_compile_test builds AND runs in one shot — read both compiler output
  and gtest output from its response.
- If the log ends with "... more lines truncated", call
  get_compile_errors before guessing at the fix.
- If a test FAILS: the literal is right (see top rule). Re-check your
  accessor, your inputs, or whether you're calling the same overload as
  the oracle.

# Insertion-code caveat (MMDB vs gemmi)
MMDB represents "no insertion code" as "" (empty string); gemmi stores
`seqid.icode` as ' ' (space char), matching the raw PDB column.
`std::string(1, residue.seqid.icode)` for a plain residue gives " ", NOT "".
When asserting insertion codes, use " " (one space), not "", in gemmi-based
tests. In MMDB-based (oracle-derived) tests use whatever the oracle printed.\
"""

# # Private/protected access
# test.cc is compiled with `-fno-access-control`. Write private/protected
# member access as if it were public. No friend declarations needed.


_RUN_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "run_test",
        "description": (
            "Run the last successfully compiled test binary and return the "
            "GoogleTest output. Use this after write_and_compile_test succeeds to check "
            "for failing assertions. Fix any EXPECT_EQ mismatches and recompile."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_COMPILE_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "write_and_compile_test",
        "description": (
            "Write the supplied C++ code as test.cc and attempt to compile it. "
            "Returns compiler output. Fix any errors shown and call again. "
            f"Maximum {MAX_COMPILE_ATTEMPTS} attempts — stop iterating if the "
            "limit is reached and output whatever compiles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete C++ test source to compile",
                },
            },
            "required": ["code"],
        },
    },
}

_GET_COMPILE_ERRORS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_compile_errors",
        "description": (
            "Return the full compiler output from the last write_and_compile_test call, "
            "without any line truncation. Use this when write_and_compile_test showed "
            "'... N more lines truncated'."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_READ_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "read_test",
        "description": (
            "Return the current contents of test.cc as written to disk. "
            "Use this before patch_test to confirm the exact text you want to replace."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_PATCH_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "patch_test",
        "description": (
            "Apply a targeted text replacement to the current test.cc and recompile. "
            "By default old_string must appear exactly once — extend it with surrounding "
            "context if it appears multiple times. Pass replace_all=true to replace every "
            "occurrence (useful for renames). Prefer this over write_and_compile_test for "
            "small fixes after the first successful write."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_string": {
                    "type": "string",
                    "description": "Exact substring to replace (must be unique in test.cc unless replace_all=true)",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace every occurrence of old_string. Defaults to false (single unique match).",
                },
            },
            "required": ["old_string", "new_string"],
        },
    },
}

TEST_TOOLS = TOOLS + [_COMPILE_TEST_TOOL, _RUN_TEST_TOOL, _GET_COMPILE_ERRORS_TOOL, _READ_TEST_TOOL, _PATCH_TEST_TOOL]


def _make_tool_handlers(test_subdir: Path) -> tuple[callable, callable, callable, callable]:
    """Return (compile_handler, run_handler, get_errors_handler, compiled_ok) sharing state about the last build."""
    attempts       = [0]
    last_binary    = [None]   # Path | None — set when a compile succeeds
    last_error_log = [None]   # Path | None

    def compile_handler(code: str) -> str:
        if attempts[0] >= MAX_COMPILE_ATTEMPTS:
            return (
                f"Compile limit reached ({MAX_COMPILE_ATTEMPTS} attempts). "
                "Output your best draft as the final ```cpp block."
            )

        # Pre-flight include check — free fix cycle on WRONG/MISSING paths.
        include_report = _tool_resolve_includes(code)
        if _has_unresolved_includes(include_report):
            return (
                "Include check FAILED (this does not count against your "
                f"{MAX_COMPILE_ATTEMPTS} compile attempts). Fix the paths "
                "below and call write_and_compile_test again:\n"
                + include_report
            )

        attempts[0] += 1

        test_subdir.mkdir(exist_ok=True)
        test_cc  = test_subdir / "test.cc"
        test_bin = test_subdir / "test_check"
        test_cc.write_text(code)

        success, output = compile_test_cc(test_cc, test_bin)

        # Always save full log so the agent can read it if needed.
        error_log = test_subdir / "compile_error.log"
        error_log.write_text(output)

        lines = output.splitlines()
        if len(lines) > 100:
            truncated = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines truncated)"
            truncated += f"\nFull log saved to: {error_log} — use get_compile_errors to see more."
            output = truncated

        if success:
            last_binary[0] = test_bin
        #     run_ok, run_out = run_test_binary(test_bin)
        #     run_lines = run_out.splitlines()
        #     if len(run_lines) > 100:
        #         run_out = "\n".join(run_lines[:100]) + f"\n... ({len(run_lines) - 100} more lines)"
            # status = "All tests PASSED." if run_ok else "Some tests FAILED — fix the assertions and recompile."
            return (
                f"Compilation succeeded (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}).\n"
            )
        last_binary[0] = None
        last_error_log[0] = error_log
        return f"Compilation FAILED (attempt {attempts[0]}/{MAX_COMPILE_ATTEMPTS}):\n{output}"

    def run_handler() -> str:
        if last_binary[0] is None:
            return "No compiled binary available — call write_and_compile_test first."
        success, output = run_test_binary(last_binary[0])
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        status = "All tests PASSED." if success else "Some tests FAILED."
        return f"{status}\n{output}"

    def get_errors_handler() -> str:
        if last_error_log[0] is None or not last_error_log[0].exists():
            return "No compile error log available."
        return last_error_log[0].read_text()

    def read_handler() -> str:
        test_cc = test_subdir / "test.cc"
        if not test_cc.exists():
            return "No test.cc written yet — call write_and_compile_test first."
        return test_cc.read_text()

    def patch_handler(old_string: str, new_string: str, replace_all: bool = False) -> str:
        test_cc = test_subdir / "test.cc"
        if not test_cc.exists():
            return "No test.cc written yet — call write_and_compile_test first."
        content = test_cc.read_text()
        count = content.count(old_string)
        if count == 0:
            return (
                "ERROR: old_string not found in test.cc. "
                "Call read_test to verify the current content, then supply "
                "an exact substring that appears in the file."
            )
        if count > 1 and not replace_all:
            return (
                f"ERROR: old_string appears {count} times — it must be unique. "
                "Extend the string to include more surrounding context so it "
                "identifies exactly one location, or pass replace_all=true to "
                "replace every occurrence."
            )
        if replace_all:
            return compile_handler(content.replace(old_string, new_string))
        return compile_handler(content.replace(old_string, new_string, 1))

    def compiled_ok() -> bool:
        return last_binary[0] is not None

    return compile_handler, run_handler, get_errors_handler, read_handler, patch_handler, compiled_ok




def generate_test_with_agent(
    conn: sqlite3.Connection,
    oracle_cc_text: str,
    oracle_result: OracleResult,
    test_subdir: Path,
    model: str,
    oracle_trace: str | None = None,
    verbose: bool = False,
) -> tuple[str | None, str]:
    """Run the agentic test generation loop.

    Returns (test_code, trace_text).
    test_code is None if the model produced nothing usable.
    """
    compile_handler, run_handler, get_errors_handler, read_handler, patch_handler, compiled_ok = _make_tool_handlers(test_subdir)

    def dispatch(name: str, args: dict) -> str:
        if name == "write_and_compile_test":
            return compile_handler(args["code"])
        if name == "run_test":
            if not compiled_ok() and last_draft[0]:
                compile_msg = compile_handler(last_draft[0])
                if not compiled_ok():
                    return (
                        "run_test: no compiled binary was available, so I "
                        "auto-compiled the most recent draft you provided. "
                        "Compilation failed — fix the errors below and call "
                        "write_and_compile_test with corrected code:\n" + compile_msg
                    )
                run_msg = run_handler()
                return (
                    "(auto-compiled latest draft before running — "
                    f"{compile_msg.splitlines()[0] if compile_msg else 'compile ok'})\n"
                    + run_msg
                )
            return run_handler()
        if name == "get_compile_errors":
            return get_errors_handler()
        if name == "read_test":
            return read_handler()
        if name == "patch_test":
            old_s = args.get("old_string")
            new_s = args.get("new_string")
            if old_s is None or new_s is None:
                return "Error: patch_test requires both 'old_string' and 'new_string'."
            return patch_handler(old_s, new_s, bool(args.get("replace_all", False)))
        return _dispatch(conn, name, args)

    parts: list[str] = []

    parts.append("## Oracle program")
    parts.append(f"```cpp\n{oracle_cc_text.rstrip()}\n```")

    parts.append("## Observed output when run")
    parts.append("_INPUT/OUTPUT lines are the ground truth; ignore any leading warnings._")
    parts.append(f"```\n{oracle_result.stdout.rstrip()}\n```")

    # Compact oracle trace summary (tool calls only)
    # if oracle_trace:
    #     tool_lines = [l.strip() for l in oracle_trace.splitlines() if l.strip().startswith("→")]
    #     if tool_lines:
    #         parts.append("## Oracle lookups already verified")
    #         parts.append("_Types and headers confirmed during oracle generation._")
    #         shown = tool_lines[:20]
    #         suffix = f"\n... ({len(tool_lines) - 20} more)" if len(tool_lines) > 20 else ""
    #         parts.append("```\n" + "\n".join(shown) + suffix + "\n```")

    # Oracle-derived notes, if the oracle stage produced any.
    notes = load_notes(test_subdir.parent / "oracle" / "notes.json")
    if notes:
        rendered = render_notes_for_prompt(notes, audience="test")
        if rendered:
            parts.append("## Validated facts from oracle stage")
            parts.append("_Reuse these verbatim rather than re-deriving._")
            parts.append(f"```\n{rendered.rstrip()}\n```")

    coverage = load_coverage(test_subdir.parent / "oracle" / "coverage.json")
    if coverage:
        rendered = render_coverage_for_prompt(coverage)
        if rendered:
            parts.append("## ⚠ Oracle coverage warning")
            parts.append(
                "_The oracle has weak coverage of the function. Translate "
                "the existing INPUT/OUTPUT lines into assertions, but if "
                "you can also add a complementary case that addresses the "
                "weakness, do so._"
            )
            parts.append(f"```\n{rendered.rstrip()}\n```")

    parts.append("## Task")
    parts.append(
        "Convert the oracle into a Google Test suite. Use the tools to verify "
        "headers, look up any types you are unsure about, then compile and run "
        "before finalising."
    )

    user_content = "\n\n".join(parts)

    messages: list[dict] = [
        {"role": "system", "content": TEST_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    test_subdir.mkdir(parents=True, exist_ok=True)
    (test_subdir / "prompt.txt").write_text(
        f"=== SYSTEM ===\n{TEST_SYSTEM_PROMPT}\n\n"
        f"=== USER ===\n{user_content}\n"
    )

    trace_lines = _TraceWriter(test_subdir / "agent_trace.txt")
    trace_lines.append("=== TEST AGENT TRACE ===\n")
    trace_lines.append(f"[user]\n{textwrap.indent(user_content, '  ')}\n")

    test_code: str | None = None
    last_draft: list[str | None] = [None]
    call_counts: dict[str, int] = {}
    tool_cache: dict[str, str] = {}
    REPEAT_LIMIT = 3
    NO_CACHE = {"write_and_compile_test", "run_test", "get_compile_errors", "patch_test", "leave_note"}
    no_compile_warned = [False]

    def _save_draft(code: str) -> None:
        if code and len(code) > 100 and "#include" in code:
            last_draft[0] = code

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
            if name == "write_and_compile_test" and isinstance(args.get("code"), str):
                _save_draft(args["code"])
            hash_args = {k: v for k, v in args.items() if k != "code"}
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
            if call_counts[key] > REPEAT_LIMIT and name not in ("write_and_compile_test", "run_test"):
                nudge = (
                    f"You have called {name} with these arguments {call_counts[key]} times. "
                    "Stop repeating — use the information you already have and proceed to "
                    "write_and_compile_test with your best draft."
                )
                trace_lines.append(f"  → {name}(repeated — nudged)")
                trace_lines.append_call(f"[repeat-intercept] {name}({json.dumps(hash_args)})")
                results.append({"role": "tool", "content": nudge})
                continue
            if verbose:
                display = {"code": "..."} if name == "write_and_compile_test" else args
                print(f"  tool: {name}({display})")
            result_text  = dispatch(name, args)
            result_lines = result_text.splitlines()
            if len(result_lines) > 150:
                result_text = (
                    "\n".join(result_lines[:150])
                    + f"\n... ({len(result_lines) - 150} more lines)"
                )
            short = json.dumps(args) if name != 'write_and_compile_test' else '{...}'
            trace_lines.append(f"  → {name}({short})")
            trace_lines.append(textwrap.indent(result_text, "      ") + "\n")
            trace_lines.append_call(f"{name}({short})")
            if name not in NO_CACHE:
                tool_cache[key] = result_text
            results.append({"role": "tool", "content": result_text})
        return results

    def _extract_code(content: str) -> str:
        m = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", content, re.DOTALL)
        code = m.group(1).strip() if m else content.strip()
        _save_draft(code)
        return code

    def _is_usable(code: str | None) -> bool:
        return bool(code and len(code) > 100 and "#include" in code)

    def _progress(label: str, tool_calls: list) -> None:
        if tool_calls:
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
            print(f"  [test] {label} → {names}", flush=True)
        else:
            print(f"  [test] {label} → done (final answer)", flush=True)

    for turn in range(20):
        print(f"  [test] turn {turn + 1}/20 ...", end="", flush=True)
        data = _chat(messages, model, TEST_TOOLS)
        _log_llm_timing(data, stage="test", turn=turn + 1, verbose=verbose, trace_lines=trace_lines)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking",  "") or ""
        assistant_content = msg.get("content",   "") or ""

        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        })

        if thinking:
            if verbose:
                print(f"\n[thinking]\n{textwrap.indent(thinking, '  ')}\n")
            trace_lines.append(
                f"[thinking — turn {turn + 1}]\n{textwrap.indent(thinking, '  ')}\n"
            )

        # Degenerate-thinking guard: pathological repetition has saturated
        # the response window — abort the loop and let rescue fire clean.
        degen, diag = _is_degenerate_thinking(thinking)
        if degen:
            print(f"\r  [test] turn {turn + 1}/20 → DEGENERATE — aborting", flush=True)
            trace_lines.append(f"[agent] {diag} — aborting loop, will issue rescue.\n")
            break

        _progress(f"turn {turn + 1}/20", tool_calls)

        if not tool_calls:
            trace_lines.append(
                f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n"
            )
            test_code = _extract_code(assistant_content)
            break

        trace_lines.append(
            f"[assistant — turn {turn + 1}, {len(tool_calls)} tool call(s)]"
        )
        messages.extend(_run_tool_calls(tool_calls))

        if (NO_COMPILE_AFTER and not no_compile_warned[0]
                and (turn + 1) >= NO_COMPILE_AFTER
                and not any(k.startswith("write_and_compile_test:") for k in call_counts)):
            messages.append({"role": "user", "content": _TEST_NO_COMPILE_NUDGE})
            trace_lines.append(f"[no-compile nudge — turn {turn + 1}]\n{textwrap.indent(_TEST_NO_COMPILE_NUDGE, '  ')}\n")
            no_compile_warned[0] = True

        if NUDGE_EVERY_N_TURNS and (turn + 1) % NUDGE_EVERY_N_TURNS == 0:
            messages.append({"role": "user", "content": _TEST_NUDGE})
            trace_lines.append(f"[nudge — turn {turn + 1}]\n{textwrap.indent(_TEST_NUDGE, '  ')}\n")

    else:
        # All 20 turns used — ask once if more time is needed.
        trace_lines.append("[agent] Turn limit reached — asking for extension.\n")
        messages.append({"role": "user",
                         "content": _EXTENSION_PROMPT.format(n=_EXTENSION_TURNS)})

        print(f"  [test] extension check ...", end="", flush=True)
        data = _chat(messages, model, TEST_TOOLS)
        _log_llm_timing(data, stage="test", turn="extension", verbose=verbose, trace_lines=trace_lines)
        msg  = data.get("message", {})
        tool_calls        = msg.get("tool_calls") or []
        thinking          = msg.get("thinking",  "") or ""
        assistant_content = msg.get("content",   "") or ""
        messages.append({"role": "assistant", "content": assistant_content,
                         "tool_calls": tool_calls})

        if thinking:
            trace_lines.append(f"[thinking — extension]\n{textwrap.indent(thinking, '  ')}\n")

        if not tool_calls:
            print(f"\r  [test] extension check → declined", flush=True)
            trace_lines.append(
                f"[assistant — final (declined extension)]\n{textwrap.indent(assistant_content, '  ')}\n"
            )
            test_code = _extract_code(assistant_content)
        else:
            print(f"\r  [test] extension check → granted", flush=True)
            trace_lines.append(f"[agent] Extension granted ({_EXTENSION_TURNS} more turns).\n")
            messages.extend(_run_tool_calls(tool_calls))

            for ext_turn in range(_EXTENSION_TURNS):
                print(f"  [test] ext turn {ext_turn + 1}/{_EXTENSION_TURNS} ...", end="", flush=True)
                data = _chat(messages, model, TEST_TOOLS)
                _log_llm_timing(data, stage="test", turn=f"ext {ext_turn + 1}", verbose=verbose, trace_lines=trace_lines)
                msg  = data.get("message", {})
                tool_calls        = msg.get("tool_calls") or []
                thinking          = msg.get("thinking",  "") or ""
                assistant_content = msg.get("content",   "") or ""
                messages.append({"role": "assistant", "content": assistant_content,
                                 "tool_calls": tool_calls})

                if thinking:
                    trace_lines.append(
                        f"[thinking — ext turn {ext_turn + 1}]\n{textwrap.indent(thinking, '  ')}\n"
                    )

                _progress(f"ext turn {ext_turn + 1}/{_EXTENSION_TURNS}", tool_calls)

                if not tool_calls:
                    trace_lines.append(
                        f"[assistant — final]\n{textwrap.indent(assistant_content, '  ')}\n"
                    )
                    test_code = _extract_code(assistant_content)
                    break

                trace_lines.append(
                    f"[assistant — ext turn {ext_turn + 1}, {len(tool_calls)} tool call(s)]"
                )
                messages.extend(_run_tool_calls(tool_calls))

                if (NO_COMPILE_AFTER and not no_compile_warned[0]
                        and not any(k.startswith("write_and_compile_test:") for k in call_counts)):
                    messages.append({"role": "user", "content": _TEST_NO_COMPILE_NUDGE})
                    trace_lines.append(f"[no-compile nudge — ext turn {ext_turn + 1}]\n{textwrap.indent(_TEST_NO_COMPILE_NUDGE, '  ')}\n")
                    no_compile_warned[0] = True

                if NUDGE_EVERY_N_TURNS and (ext_turn + 1) % NUDGE_EVERY_N_TURNS == 0:
                    messages.append({"role": "user", "content": _TEST_NUDGE})
                    trace_lines.append(f"[nudge — ext turn {ext_turn + 1}]\n{textwrap.indent(_TEST_NUDGE, '  ')}\n")
            else:
                trace_lines.append("[agent] Extension exhausted without final answer.\n")

    if not _is_usable(test_code):
        if _is_usable(last_draft[0]):
            trace_lines.append("[agent] Falling back to last saved draft.\n")
            test_code = last_draft[0]
        else:
            trace_lines.append("[agent] No usable output — issuing rescue prompt.\n")
            messages.append({"role": "user", "content": (
                "STOP. Do not call any tools. Output your best attempt at test.cc "
                "NOW inside a single ```cpp block. This is your last chance — any "
                "plausible draft is better than no output."
            )})
            try:
                data = _chat(messages, model, tools=[])
                _log_llm_timing(data, stage="test", turn="rescue", verbose=verbose, trace_lines=trace_lines)
                assistant_content = (data.get("message") or {}).get("content") or ""
                trace_lines.append(
                    f"[assistant — rescue]\n{textwrap.indent(assistant_content, '  ')}\n"
                )
                rescued = _extract_code(assistant_content)
                if _is_usable(rescued):
                    test_code = rescued
                elif _is_usable(last_draft[0]):
                    test_code = last_draft[0]
            except (urllib.error.URLError, json.JSONDecodeError) as e:
                trace_lines.append(f"[agent] Rescue call failed: {e}\n")
                if _is_usable(last_draft[0]):
                    test_code = last_draft[0]

    text = trace_lines.text()
    trace_lines.close()
    return test_code, text

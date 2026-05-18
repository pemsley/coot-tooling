#!/usr/bin/env python3
"""
Generate an oracle.cc for a given function.

Usage:
  python -m tooling.generate "coot::molecule_t::get_bonds_mesh"
  python -m tooling.generate "coot::molecule_t::get_bonds_mesh" --model gemma4:31b
  python -m tooling.generate "coot::molecule_t::get_bonds_mesh" --dry-run

Outputs to:
  oracle-data/<sanitized_name>/prompt.txt
  oracle-data/<sanitized_name>/oracle.cc
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from .runner import run_oracle
from ..db import connect, get_function
from .render import build_oracle_prompt
from .agent import generate_with_agent
from .compile import write_compile_script, compile_oracle
from .notes import extract_oracle_notes, save_notes
from .coverage import compute_coverage, save_coverage, render_summary
from .pdb_selector import select_pdb, catalog_note, structural_note, pdb_path as make_pdb_path
from ..ollama import generate_url

OLLAMA_URL    = "http://localhost:11434/api/generate"  # kept for reference
DEFAULT_MODEL = "qwen3.6"
OUT_ROOT      = Path(__file__).parent.parent.parent / "generated-tests"

CRITIQUE_INSTRUCTIONS = """\
You are reviewing a C++ oracle program that was generated to observe the inputs
and outputs of a specific function.

Critique the program below against the original context. Check for:
  - Incorrect or missing includes
  - Wrong construction of the receiver object
  - Methods or types used that are not shown in the context
  - Missing INPUT/OUTPUT print statements
  - Code that will not compile

If the program is correct and complete, respond with exactly: LGTM

If you can improve it, respond with the corrected program inside a ```cpp block,
with comments where you have changed it and why.\
"""


def sanitize_name(qname: str, sig_hash: str | None = None) -> str:
    """Convert a qualified name + optional overload hash to a safe directory name.

    Single-overload functions pass `sig_hash=None` and keep their legacy
    directory name (e.g. `coot__molecule_t__get_bonds_mesh/`). Overloaded
    functions pass a 6-char hex sig_hash so each overload gets its own
    directory (e.g. `coot__molecule_t__backrub_rotamer__461ac0/`).
    """
    base = re.sub(r"[^a-zA-Z0-9]", "_", qname).strip("_")
    if sig_hash:
        return f"{base}__{sig_hash}"
    return base


def find_function_dirs(qname: str, out_root: Path | None = None) -> list[Path]:
    """Return every per-overload output dir that exists for `qname`.

    Used by dep-resolution code (gemmi callee lookup, aggregation) that
    knows a callee qname but not which overload's port to load. Matches
    both the legacy single-overload form `<sanitized>/` and the per-overload
    `<sanitized>__<hash>/` form.
    """
    root = out_root if out_root is not None else OUT_ROOT
    base = sanitize_name(qname)
    out: list[Path] = []
    exact = root / base
    if exact.is_dir():
        out.append(exact)
    out.extend(sorted(root.glob(f"{base}__*")))
    return out


def call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "think":  False,
        # "options": {"temperature": 0.2, "num_predict": 2048},
    }).encode()

    req = urllib.request.Request(
        generate_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def extract_cpp(response: str) -> str:
    """Pull the C++ code block out of the LLM response if wrapped in markdown."""
    match = re.search(r"```(?:cpp|c\+\+)?\n(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def critique_oracle(oracle_code: str, original_prompt: str, model: str) -> str | None:
    """Run a critique pass on oracle_code.

    Returns improved C++ source if the LLM found issues, or None if LGTM.
    """
    critique_prompt = (
        f"{CRITIQUE_INSTRUCTIONS}\n\n"
        f"--- ORIGINAL CONTEXT ---\n{original_prompt}\n"
        f"--- GENERATED PROGRAM ---\n```cpp\n{oracle_code}\n```\n"
    )
    response = call_ollama(critique_prompt, model)
    if response.strip().upper().startswith("LGTM"):
        return None
    return extract_cpp(response)


def generate_one(
    conn,
    function_qname: str,
    sig_hash: str | None = None,
    model: str = DEFAULT_MODEL,
    second_pass: bool = False,
    verbose: bool = False,
    out_root: Path = OUT_ROOT,
) -> Path | None:
    """Generate oracle.cc for a single function (overload).

    `sig_hash` disambiguates among overloaded names. Pass None for
    non-overloaded functions to keep the legacy unsuffixed output dir.

    Returns the output directory on success, None if the function wasn't found.
    Raises urllib.error.URLError if Ollama is unreachable.
    """
    out_dir = out_root / sanitize_name(function_qname, sig_hash)
    oracle_out = out_dir / "oracle"
    oracle_out.mkdir(parents=True, exist_ok=True)
    oracle_cc_path = oracle_out / "oracle.cc"

    # Select the most appropriate example PDB for this function.
    fn_row = get_function(conn, function_qname, sig_hash)
    pdb_file, pdb_certain = select_pdb(
        function_qname,
        source_code=fn_row["source_code"] or "" if fn_row else "",
        doc_comment=fn_row["comment"] or "" if fn_row else "",
    )
    pdb_note = "" if pdb_certain else catalog_note(selected_file=pdb_file)
    snote = structural_note(pdb_file)
    if snote:
        pdb_note = f"{pdb_note}\n{snote}".strip()
    if pdb_certain:
        print(f"[pdb] auto-selected {pdb_file} for {function_qname}")
    else:
        print(f"[pdb] using default {pdb_file} for {function_qname} (LLM may choose another)")

    oracle_code, trace = generate_with_agent(
        conn, function_qname, model,
        oracle_out=oracle_out, verbose=verbose,
        pdb_file=pdb_file, pdb_note=pdb_note,
        sig_hash=sig_hash,
    )
    (oracle_out / "agent_trace.txt").write_text(trace)
    if oracle_code is None:
        return None
    oracle_cc_path.write_text(oracle_code)

    # if second_pass:
    #     context = (oracle_out / "prompt.txt").read_text() if (oracle_out / "prompt.txt").exists() else ""
    #     improved = critique_oracle(oracle_code, context, model)
    #     if improved:
    #         (oracle_out / "oracle_second_pass.cc").write_text(improved)

    write_compile_script(oracle_out)
    compile_oracle(oracle_out)

    result = run_oracle(oracle_out)
    print(result.summary())

    # Coverage signal — heuristic check that the oracle did something
    # interesting. Persist for the test/gemmi stages to read.
    if result.success:
        try:
            fn_src = get_function(conn, function_qname, sig_hash)
            cov = compute_coverage(
                fn_src["source_code"] if fn_src else "",
                result,
            )
            save_coverage(cov, oracle_out / "coverage.json")
            print(f"  {render_summary(cov)}")
            for s in cov.signals:
                print(f"  [coverage] {s}")
        except Exception as e:
            print(f"[coverage] skipped: {e}")

    # Extract structured notes from the working oracle for downstream stages.
    # Best-effort: a failure here should not fail oracle generation.
    if result.success:
        try:
            notes = extract_oracle_notes(oracle_code, function_qname, model)
            if notes:
                save_notes(notes, oracle_out / "notes.json")
        except Exception as e:
            print(f"[notes] extraction skipped: {e}")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an oracle.cc for a function")
    parser.add_argument("function", help="Fully-qualified function name")
    parser.add_argument("--sig",         metavar="HASH", default=None,
                        help="Overload sig_hash (6-char hex). Required when the "
                             "function is overloaded; defaults to the first overload "
                             "the DB returns otherwise.")
    parser.add_argument("--model",       default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--backend",     default="ollama", choices=["ollama", "openai"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable reasoning/thinking output (sets CT_THINK=0)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print the prompt without calling the LLM")
    parser.add_argument("--second-pass", action="store_true",
                        help="Critique and optionally improve the generated oracle")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print thinking and tool calls to the console")
    args = parser.parse_args()

    os.environ["CT_BACKEND"] = args.backend
    if args.no_thinking:
        os.environ["CT_THINK"] = "0"

    conn = connect()

    if args.dry_run:
        fn_row = get_function(conn, args.function, args.sig)
        pdb_file, pdb_certain = select_pdb(
            args.function,
            source_code=fn_row["source_code"] or "" if fn_row else "",
            doc_comment=fn_row["comment"] or "" if fn_row else "",
        )
        pdb_note = "" if pdb_certain else catalog_note(selected_file=pdb_file)
        snote = structural_note(pdb_file)
        if snote:
            pdb_note = f"{pdb_note}\n{snote}".strip()
        prompt = build_oracle_prompt(conn, args.function,
                                     pdb_file=pdb_file, pdb_note=pdb_note,
                                     sig_hash=args.sig)
        conn.close()
        if prompt is None:
            print(f"Function not found in DB: {args.function}", file=sys.stderr)
            sys.exit(1)
        print(prompt)
        return

    print(f"Calling {args.model}... (agent mode)")
    try:
        out_dir = generate_one(
            conn, args.function,
            sig_hash=args.sig,
            model=args.model,
            second_pass=args.second_pass,
            verbose=args.verbose,
        )
    except urllib.error.URLError as e:
        print(f"Ollama not reachable: {e}\nStart it with: ollama serve", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    if out_dir is None:
        print(f"Function not found in DB: {args.function}", file=sys.stderr)
        sys.exit(1)

    # for f in sorted(out_dir.iterdir()):
    #     print(f"Saved → {f}")


if __name__ == "__main__":
    main()

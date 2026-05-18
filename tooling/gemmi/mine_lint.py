#!/usr/bin/env python3
"""Auto-mine candidate lint patterns from `generated-tests/*/gemmi/compile.log`.

Reads every gemmi compile log, extracts `error:` lines, normalizes them into
clusterable templates, and emits a Markdown report ranked by uncovered
frequency. A cluster is treated as "covered" if the existing gemmi_lint
patterns produce at least one finding when run against the same port's
source files — so the report surfaces only error families the lint is
currently blind to.

Run:
  python -m tooling.gemmi.mine_lint
  python -m tooling.gemmi.mine_lint --top 50 --out report.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running as a script as well as a module.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tooling.gemmi.lint import gemmi_lint


_ERROR_RE = re.compile(r"^.+?:(\d+):(?:\d+:)?\s*error:\s*(.*)$")
_GENERATED = Path("generated-tests")
_REPORT = Path("tooling/gemmi/.lint_candidates.md")


def _normalize(message: str) -> str:
    """Reduce an error message to a clusterable template.

    Replaces literals and identifier-shaped tokens with placeholders so
    "no member named 'foo'" and "no member named 'bar'" collapse to the
    same template.
    """
    s = message
    s = re.sub(r"\s*\(aka\s+'[^']*'\)", "", s)
    s = re.sub(r"'[^']*'", "'X'", s)
    s = re.sub(r'"[^"]*"', '"S"', s)
    s = re.sub(r"\b\S+\.(?:cc|hh|cpp|c|h|hpp|hxx)\b", "P", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_tokens(message: str) -> list[str]:
    return re.findall(r"'([^']*)'", message)


def _harvest_errors(log: Path) -> list[dict]:
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        m = _ERROR_RE.match(line)
        if not m:
            continue
        msg = m.group(2).strip()
        out.append({
            "line": int(m.group(1)),
            "raw":  msg,
            "template": _normalize(msg),
            "tokens":   _extract_tokens(msg),
        })
    return out


def _existing_lint_covers(port_dir: Path) -> bool:
    """True if gemmi_lint would fire on at least one of this port's sources."""
    for fname in ("function.hh", "function.cc", "test.cc"):
        p = port_dir / fname
        if p.exists() and gemmi_lint(p.read_text()):
            return True
    return False


def _suggest_regex(tokens: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if not t or len(t) < 3 or t in seen:
            continue
        seen.add(t)
        if re.fullmatch(r"[\w:]+", t):
            out.append(rf"\b{re.escape(t)}\b")
        if len(out) >= 5:
            break
    return out


def mine(root: Path = _GENERATED) -> dict:
    clusters: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "samples": [],
        "tokens": Counter(),
        "covered_by_existing": 0,
    })
    total_logs = 0
    for log in sorted(root.glob("*/gemmi/compile.log")):
        total_logs += 1
        errs = _harvest_errors(log)
        if not errs:
            continue
        port_dir = log.parent
        is_covered = _existing_lint_covers(port_dir)
        for e in errs:
            c = clusters[e["template"]]
            c["count"] += 1
            if len(c["samples"]) < 5:
                c["samples"].append({
                    "log":  str(log),
                    "line": e["line"],
                    "raw":  e["raw"],
                    "port": port_dir.parent.name,
                })
            for t in e["tokens"]:
                c["tokens"][t] += 1
            if is_covered:
                c["covered_by_existing"] += 1
    return {"total_logs": total_logs, "clusters": clusters}


def render_markdown(report: dict, top_n: int = 30) -> str:
    clusters = report["clusters"]
    ranked = sorted(
        clusters.items(),
        key=lambda kv: (-(kv[1]["count"] - kv[1]["covered_by_existing"]),
                        -kv[1]["count"],
                        kv[0]),
    )
    lines: list[str] = []
    lines.append("# gemmi compile-error mining report")
    lines.append("")
    lines.append(
        f"Scanned **{report['total_logs']}** compile logs. "
        f"Found **{len(clusters)}** distinct error templates. "
        f"Ranked by uncovered count — errors whose port already trips an "
        f"existing `gemmi_lint` pattern are subtracted from the rank."
    )
    lines.append("")
    shown = 0
    for tmpl, c in ranked:
        uncovered = c["count"] - c["covered_by_existing"]
        if uncovered <= 0:
            continue
        shown += 1
        if shown > top_n:
            break
        lines.append(f"## {shown}. `{tmpl}`")
        lines.append(
            f"- **Total**: {c['count']} occurrence(s) "
            f"(uncovered: **{uncovered}**, "
            f"covered-by-existing-lint: {c['covered_by_existing']})"
        )
        top_tokens = c["tokens"].most_common(8)
        if top_tokens:
            lines.append("- **Most common identifiers in this cluster**:")
            for tok, n in top_tokens:
                lines.append(f"  - `{tok}` × {n}")
        if c["samples"]:
            lines.append("- **Sample errors**:")
            for s in c["samples"][:3]:
                snippet = s["raw"][:160]
                lines.append(f"  - `{s['port']}` — {snippet}")
        suggested = _suggest_regex([t for t, _ in top_tokens])
        if suggested:
            lines.append("- **Regex-pattern candidates** (paste into `_PATTERNS` in `gemmi/lint.py`):")
            for r in suggested:
                lines.append(f"  - `r\"{r}\"`")
        lines.append("")
    if shown == 0:
        lines.append("_(No uncovered clusters — every observed error is "
                     "already caught by an existing pattern.)_")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine candidate lint patterns from gemmi compile logs."
    )
    parser.add_argument("--root", type=Path, default=_GENERATED,
                        help="Root containing per-port directories.")
    parser.add_argument("--out", type=Path, default=_REPORT,
                        help="Markdown report path.")
    parser.add_argument("--top", type=int, default=30,
                        help="Max number of uncovered clusters to render.")
    args = parser.parse_args()

    report = mine(args.root)
    md = render_markdown(report, top_n=args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)

    uncovered_total = sum(
        max(c["count"] - c["covered_by_existing"], 0)
        for c in report["clusters"].values()
    )
    print(
        f"Scanned {report['total_logs']} logs; "
        f"{len(report['clusters'])} templates; "
        f"{uncovered_total} uncovered occurrences. "
        f"Wrote {args.out}"
    )


if __name__ == "__main__":
    main()

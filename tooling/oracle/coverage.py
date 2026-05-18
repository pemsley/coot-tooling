"""Oracle coverage signal — heuristic measure of whether the oracle exercises
the function meaningfully or only hits early-return guards.

Combines:
  * Static analysis of the function source (branches, returns, early guards)
  * Dynamic analysis of the oracle's runtime output (case count, output
    diversity, BEFORE/AFTER mutation pairs, trivial-only detection)

Produces a coverage_score 0.0–1.0 plus discrete weakness signals.
Persisted as oracle/coverage.json and rendered into downstream prompts so
weak oracles don't silently propagate into ports.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .runner.results import OracleResult


_LINE_COMMENT_RE  = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_RE        = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)
_CHAR_RE          = re.compile(r"'(?:[^'\\]|\\.)'")


def _strip_noncode(s: str) -> str:
    s = _BLOCK_COMMENT_RE.sub(" ", s)
    s = _LINE_COMMENT_RE.sub(" ", s)
    s = _STRING_RE.sub('""', s)
    s = _CHAR_RE.sub("''", s)
    return s


@dataclass
class StaticAnalysis:
    lines:        int = 0
    branches:     int = 0
    loops:        int = 0
    returns:      int = 0
    early_guards: int = 0


@dataclass
class DynamicAnalysis:
    n_cases:              int = 0
    n_inputs:             int = 0
    n_outputs:            int = 0
    n_distinct_outputs:   int = 0
    all_outputs_trivial:  bool = False
    n_before_after_pairs: int = 0
    n_identical_pairs:    int = 0


@dataclass
class CoverageReport:
    score:       float = 1.0
    static:      StaticAnalysis  = field(default_factory=StaticAnalysis)
    dynamic:     DynamicAnalysis = field(default_factory=DynamicAnalysis)
    signals:     list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


_TRIVIAL: frozenset[str] = frozenset({
    "0", "0.0", "0.00", "", "false", "FALSE", "False",
    "nullptr", "NULL", "(null)", "null", "none", "None",
})


def _is_trivial(value: str) -> bool:
    v = value.strip()
    if v in _TRIVIAL:
        return True
    if re.fullmatch(r"0*\.?0*", v) and v not in {"."}:
        return True
    return False


_BRANCH_RE  = re.compile(r"\bif\s*\(")
_CASE_RE    = re.compile(r"\bcase\b[^:]*:")
_LOOP_RE    = re.compile(r"\b(?:for|while)\s*\(")
_DO_RE      = re.compile(r"\bdo\s*\{")
_RETURN_RE  = re.compile(r"\breturn\b")
_TERNARY_RE = re.compile(r"\?[^?:]+:")


def analyze_static(source: str) -> StaticAnalysis:
    clean = _strip_noncode(source)
    non_blank = [ln for ln in clean.splitlines() if ln.strip()]
    n_lines = len(non_blank)
    branches = (len(_BRANCH_RE.findall(clean))
                + len(_CASE_RE.findall(clean))
                + len(_TERNARY_RE.findall(clean)))
    loops    = len(_LOOP_RE.findall(clean)) + len(_DO_RE.findall(clean))
    returns  = len(_RETURN_RE.findall(clean))
    if n_lines == 0:
        early = 0
    else:
        head_cutoff = max(1, n_lines // 4)
        head_src = "\n".join(non_blank[:head_cutoff])
        early = len(_RETURN_RE.findall(head_src))
    return StaticAnalysis(
        lines=n_lines, branches=branches, loops=loops,
        returns=returns, early_guards=early,
    )


def analyze_dynamic(result: OracleResult) -> DynamicAnalysis:
    if result.cases:
        n_cases = len(result.cases)
        n_inputs = sum(len(c["inputs"]) for c in result.cases)
        out_pairs: list[tuple[str, str]] = []
        for c in result.cases:
            out_pairs.extend(c["outputs"].items())
    else:
        n_cases = 1 if result.outputs else 0
        n_inputs = len(result.inputs)
        out_pairs = list(result.outputs.items())

    n_outputs = len(out_pairs)
    distinct = {v.strip() for _, v in out_pairs}
    all_triv = bool(out_pairs) and all(_is_trivial(v) for _, v in out_pairs)

    before_keys = [k for k, _ in out_pairs if "before" in k.lower()]
    after_keys  = {k.lower(): k for k, _ in out_pairs if "after" in k.lower()}
    out_map = dict(out_pairs)
    pairs: list[tuple[str, str]] = []
    for bk in before_keys:
        ak_lower = bk.lower().replace("before", "after")
        if ak_lower in after_keys:
            pairs.append((out_map[bk], out_map[after_keys[ak_lower]]))
    identical = sum(1 for b, a in pairs if b.strip() == a.strip())

    return DynamicAnalysis(
        n_cases=n_cases, n_inputs=n_inputs, n_outputs=n_outputs,
        n_distinct_outputs=len(distinct),
        all_outputs_trivial=all_triv,
        n_before_after_pairs=len(pairs),
        n_identical_pairs=identical,
    )


def _combine(static: StaticAnalysis, dyn: DynamicAnalysis) -> CoverageReport:
    score = 1.0
    signals: list[str] = []
    suggestions: list[str] = []

    if dyn.n_outputs == 0:
        score -= 0.5
        signals.append("Oracle produced no OUTPUT lines.")
        suggestions.append(
            "Print the function's return value, mutated state, or any "
            "computed result so the port has assertions to freeze."
        )

    if dyn.all_outputs_trivial:
        score -= 0.3
        signals.append(
            "Every OUTPUT value is trivial (0, empty, false, or nullptr). "
            "Inputs likely fail the function's guards — only the early-return "
            "path is being observed."
        )
        suggestions.append(
            "Choose inputs that pass every guard clause so the function's "
            "core logic runs and produces non-zero observable outputs."
        )

    if (dyn.n_before_after_pairs > 0
            and dyn.n_identical_pairs == dyn.n_before_after_pairs):
        score -= 0.3
        signals.append(
            f"BEFORE == AFTER for all {dyn.n_before_after_pairs} mutation "
            "observation(s). The function did nothing."
        )
        suggestions.append(
            "Construct inputs that actually cause the receiver to mutate."
        )

    if static.branches >= 3 and dyn.n_cases <= 1:
        score -= 0.2
        signals.append(
            f"Function source has {static.branches} branches but the oracle "
            f"exercised only {dyn.n_cases} case — multiple code paths are "
            "untested."
        )
        suggestions.append(
            "Add a second INPUT/OUTPUT case that takes a different branch "
            "through the function."
        )

    if dyn.n_outputs >= 4 and dyn.n_distinct_outputs <= 1:
        score -= 0.15
        signals.append(
            f"All {dyn.n_outputs} OUTPUTs reduced to {dyn.n_distinct_outputs} "
            "distinct value(s); the cases aren't distinguishing behaviour."
        )

    if static.lines >= 30 and dyn.n_outputs < 3:
        score -= 0.1
        signals.append(
            f"Long function ({static.lines} non-blank lines) but only "
            f"{dyn.n_outputs} OUTPUT(s). Outputs are likely under-observed."
        )
        suggestions.append(
            "Print intermediate values, container sizes, or shapes that the "
            "function computes."
        )

    return CoverageReport(
        score=max(0.0, round(score, 3)),
        static=static, dynamic=dyn,
        signals=signals, suggestions=suggestions,
    )


def compute_coverage(source_code: str, result: OracleResult) -> CoverageReport:
    return _combine(analyze_static(source_code or ""), analyze_dynamic(result))


def save_coverage(report: CoverageReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n")


def load_coverage(path: Path) -> CoverageReport | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return CoverageReport(
        score=float(data.get("score", 0.0)),
        static=StaticAnalysis(**data.get("static", {})),
        dynamic=DynamicAnalysis(**data.get("dynamic", {})),
        signals=list(data.get("signals", [])),
        suggestions=list(data.get("suggestions", [])),
    )


def render_for_prompt(report: CoverageReport) -> str:
    """One block of text describing weaknesses + how to compensate.
    Empty when the score is high and there are no signals."""
    if not report.signals and report.score >= 0.95:
        return ""
    lines = [f"Oracle coverage score: {report.score:.2f} / 1.00"]
    if report.signals:
        lines.append("Warnings about the oracle's coverage:")
        for s in report.signals:
            lines.append(f"  - {s}")
    if report.suggestions:
        lines.append("How to compensate when porting:")
        for s in report.suggestions:
            lines.append(f"  - {s}")
    return "\n".join(lines)


def render_summary(report: CoverageReport) -> str:
    return (
        f"coverage: score={report.score:.2f} "
        f"cases={report.dynamic.n_cases} "
        f"outputs={report.dynamic.n_outputs} "
        f"distinct={report.dynamic.n_distinct_outputs} "
        f"signals={len(report.signals)}"
    )

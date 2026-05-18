"""Profile test run successes and failures across generated-tests.

Scans every function directory under generated-tests/ and parses the
GoogleTest run.log in each stage subdirectory (oracle, test, gemmi).

Usage:
  python -m tooling.profile
  python -m tooling.profile --stage test
  python -m tooling.profile --failures
  python -m tooling.profile --csv results.csv
  python -m tooling.profile --graph
  python -m tooling.profile --graph --graph-out progress.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tooling import db as _db

OUT_ROOT = Path(__file__).parent.parent / "generated-tests"

STAGES = ("oracle", "test", "gemmi")

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) test")
_FAILED_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\d+) test")
_FAIL_NAME_RE = re.compile(r"\[\s*FAILED\s*\]\s+\S+\.\S+")


def _log_passed(log_path: Path) -> bool | None:
    """Return True=pass, False=fail, None=missing/incomplete from a GoogleTest run.log."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    m_f = _FAILED_RE.search(text)
    if m_f and int(m_f.group(1)) > 0:
        return False
    m_p = _PASSED_RE.search(text)
    if m_p and int(m_p.group(1)) > 0:
        return True
    return None


def _status_oracle(stage_dir: Path) -> str:
    """pass = result.json exists with at least one case; fail = exists but no cases; missing = absent."""
    result_json = stage_dir / "result.json"
    if not result_json.exists():
        return "missing"
    try:
        data = json.loads(result_json.read_text())
        return "pass" if data.get("cases") else "fail"
    except Exception:
        return "fail"


def _status_test(stage_dir: Path) -> str:
    """pass/fail from run.log PASSED/FAILED line; missing if log absent or incomplete."""
    result = _log_passed(stage_dir / "run.log")
    if result is True:
        return "pass"
    if result is False:
        return "fail"
    return "missing"


def _status_gemmi(stage_dir: Path) -> str:
    """pass = function.hh + test.cc exist AND run.log passes; fail = files exist but log fails; missing otherwise."""
    has_files = (stage_dir / "function.hh").exists() and (stage_dir / "test.cc").exists()
    if not has_files:
        return "missing"
    result = _log_passed(stage_dir / "run.log")
    if result is True:
        return "pass"
    if result is False:
        return "fail"
    return "missing"


_STATUS_FN = {
    "oracle": _status_oracle,
    "test": _status_test,
    "gemmi": _status_gemmi,
}


@dataclass
class FunctionResult:
    name: str
    stage_status: dict[str, str] = field(default_factory=dict)   # stage -> pass/fail/missing
    failure_details: dict[str, list[str]] = field(default_factory=dict)  # stage -> lines


def _collect_failure_lines(log_path: Path) -> list[str]:
    lines = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "Failure" in line or _FAIL_NAME_RE.match(line.strip()):
            lines.append(line.rstrip())
    return lines


def collect(stages: list[str]) -> list[FunctionResult]:
    results = []
    for fn_dir in sorted(OUT_ROOT.iterdir()):
        if not fn_dir.is_dir() or fn_dir.name.startswith("_"):
            continue
        r = FunctionResult(name=fn_dir.name)
        for stage in stages:
            stage_dir = fn_dir / stage
            status = _STATUS_FN[stage](stage_dir)
            r.stage_status[stage] = status
            if status == "fail":
                log = stage_dir / "run.log"
                if log.exists():
                    r.failure_details[stage] = _collect_failure_lines(log)
        results.append(r)
    return results


def _hr(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _stage_counts(results: list[FunctionResult], stage: str) -> tuple[int, int, int]:
    n_pass = sum(1 for r in results if r.stage_status.get(stage) == "pass")
    n_fail = sum(1 for r in results if r.stage_status.get(stage) == "fail")
    n_miss = sum(1 for r in results if r.stage_status.get(stage) == "missing")
    return n_pass, n_fail, n_miss


def print_report(results: list[FunctionResult], stages: list[str], failures_only: bool) -> None:
    total = len(results)

    if not failures_only:
        _hr(f"Per-function status ({total} functions)")
        col_w = max(len(r.name) for r in results)
        stage_w = 8
        hdr = f"  {'function':<{col_w}}" + "".join(f"  {s:^{stage_w}}" for s in stages)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in results:
            row = f"  {r.name:<{col_w}}"
            for stage in stages:
                st = r.stage_status.get(stage, "missing")
                sym = {"pass": "OK", "fail": "FAIL", "missing": "—"}.get(st, st)
                row += f"  {sym:^{stage_w}}"
            print(row)

    _hr("Failing tests")
    any_fail = False
    for r in results:
        for stage in stages:
            if r.stage_status.get(stage) == "fail":
                any_fail = True
                print(f"\n  [{stage}] {r.name}")
                for line in r.failure_details.get(stage, [])[:10]:
                    print(f"    {line}")
    if not any_fail:
        print("  (none)")

    _hr("Stage summary")
    header = f"  {'stage':<10}  {'pass':>6}  {'fail':>6}  {'missing':>8}  {'pass%':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stage in stages:
        n_pass, n_fail, n_miss = _stage_counts(results, stage)
        ran = n_pass + n_fail
        pct = f"{100*n_pass/ran:.1f}%" if ran else "—"
        print(f"  {stage:<10}  {n_pass:>6}  {n_fail:>6}  {n_miss:>8}  {pct:>6}")



def _sanitize(qname: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", qname).strip("_")


def db_coverage(results: list[FunctionResult]) -> dict:
    """Return counts of total MMDB functions in the DB vs attempted vs stage-complete."""
    conn = _db.connect()
    rows = conn.execute("""
        SELECT DISTINCT f.qualified_name
        FROM functions f
        JOIN uses_type u ON u.function_id = f.id
        WHERE f.kind IN ('CXX_METHOD', 'CONSTRUCTOR', 'DESTRUCTOR',
                         'FUNCTION_TEMPLATE', 'FUNCTION_DECL')
          AND u.type_qualified_name LIKE 'mmdb::%'
    """).fetchall()
    conn.close()

    total_mmdb = len(rows)
    result_map = {r.name: r for r in results}
    sanitized = [_sanitize(row[0]) for row in rows]
    attempted = sum(1 for s in sanitized if s in result_map)

    stage_complete: dict[str, int] = {}
    for stage in STAGES:
        stage_complete[stage] = sum(
            1 for s in sanitized
            if result_map.get(s) and result_map[s].stage_status.get(stage) == "pass"
        )

    return {
        "total_mmdb": total_mmdb,
        "attempted": attempted,
        "stage_complete": stage_complete,
    }


def plot_graph(results: list[FunctionResult], stages: list[str], out_path: str | None,
               presentation: bool = False) -> None:
    import datetime
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import numpy as np

    # ── Palette & style ────────────────────────────────────────────────────────
    if presentation:
        BG       = "none"
        PANEL_BG = "none"
        TEXT     = "#1A1A2E"
        SUBTEXT  = "#4B5563"
        GRID     = "#D1D5DB"
    else:
        BG       = "#F7F8FA"
        PANEL_BG = "#FFFFFF"
        TEXT     = "#1A1A2E"
        SUBTEXT  = "#6B7280"
        GRID     = "#E5E7EB"

    PASS     = "#3ECF8E"   # teal-green
    FAIL     = "#F06464"   # soft red
    MISS     = "#D8DCE6"   # cool grey
    ATT      = "#6EA8FE"   # periwinkle blue

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans"],
        "text.color":        TEXT,
        "axes.labelcolor":   SUBTEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "axes.edgecolor":    GRID,
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL_BG,
    })

    total = len(results)
    cov = db_coverage(results)
    total_mmdb = cov["total_mmdb"]
    stage_data = {s: _stage_counts(results, s) for s in stages}
    fig = plt.figure(figsize=(22, 6))
    fig.patch.set_facecolor(BG)

    # Title + subtitle
    fig.text(0.5, 0.98, "MMDB → Gemmi Refactor", ha="center", va="top",
             fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.93, f"{total} functions attempted  ·  {total_mmdb} total MMDB functions in DB",
             ha="center", va="top", fontsize=10, color=SUBTEXT)

    gs = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.84, bottom=0.12,
                          wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    def _style_ax(ax, title, xlabel=None, ylabel=None):
        ax.set_facecolor(PANEL_BG)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.set_title(title, fontsize=11, fontweight="semibold", color=TEXT,
                     pad=10, loc="left")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color=SUBTEXT, labelpad=6)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color=SUBTEXT, labelpad=6)
        ax.tick_params(labelsize=9)

    # ── Panel 1: pass-rate funnel (of attempted) ──────────────────────────────
    ax2 = axes[0]
    _style_ax(ax2, "Pass rate (of all attempted)", xlabel="% of attempted")
    labels = [s.capitalize() for s in stages]
    pass_counts = [stage_data[s][0] for s in stages]
    pcts = [100 * p / total if total else 0 for p in pass_counts]
    y = np.arange(len(stages))
    bar_h = 0.48

    # Background track
    ax2.barh(y, [100] * len(stages), height=bar_h, color=MISS, zorder=2, linewidth=0)
    # Value bars
    ax2.barh(y, pcts, height=bar_h, color=PASS, zorder=3, linewidth=0)

    for i, (pct, cnt) in enumerate(zip(pcts, pass_counts)):
        ax2.text(pct - 1.5, i, f"{pct:.1f}%",
                 va="center", ha="right", fontsize=9,
                 fontweight="bold", color="white", zorder=4)
        ax2.text(102, i, f"{cnt}/{total}",
                 va="center", ha="left", fontsize=8, color=SUBTEXT)

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlim(0, 118)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax2.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)

    # ── Panel 2: DB coverage ──────────────────────────────────────────────────
    ax3 = axes[2]
    _style_ax(ax3, "DB coverage", xlabel=f"% of {total_mmdb} MMDB functions")
    cov_labels = ["Attempted"] + [s.capitalize() for s in stages]
    cov_counts = [cov["attempted"]] + [cov["stage_complete"][s] for s in stages]
    cov_pcts   = [100 * c / total_mmdb if total_mmdb else 0 for c in cov_counts]
    cov_colors = [ATT] + [PASS] * len(stages)
    y3 = np.arange(len(cov_labels))

    ax3.barh(y3, [100] * len(cov_labels), height=bar_h, color=MISS, zorder=2, linewidth=0)
    for i, (pct, color) in enumerate(zip(cov_pcts, cov_colors)):
        ax3.barh(i, pct, height=bar_h, color=color, zorder=3, linewidth=0)

    for i, (pct, cnt) in enumerate(zip(cov_pcts, cov_counts)):
        label = f"{pct:.1f}%"
        if pct >= 12:
            ax3.text(pct - 1.5, i, label,
                     va="center", ha="right", fontsize=9,
                     fontweight="bold", color="white", zorder=4)
        else:
            ax3.text(pct + 1.5, i, label,
                     va="center", ha="left", fontsize=9,
                     fontweight="bold", color=SUBTEXT, zorder=4)
        ax3.text(102, i, f"{cnt}/{total_mmdb}",
                 va="center", ha="left", fontsize=8, color=SUBTEXT)

    ax3.set_yticks(y3)
    ax3.set_yticklabels(cov_labels, fontsize=10)
    ax3.set_xlim(0, 119)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax3.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax3.set_axisbelow(True)
    ax3.spines["left"].set_visible(False)
    ax3.tick_params(left=False)

    # ── Panel 3: conditional pass rate (vs previous stage) ───────────────────
    ax4 = axes[1]
    if all(s in stage_data for s in STAGES):
        _style_ax(ax4, "Conditional pass rate", xlabel="% of prior-stage passes")

        # Denominators: oracle uses total attempted; test uses oracle passes; gemmi uses test passes
        prior_pass = [total, stage_data["oracle"][0], stage_data["test"][0]]
        cond_pass_counts = [stage_data[s][0] for s in STAGES]
        cond_pcts = [
            100 * p / denom if denom else 0
            for p, denom in zip(cond_pass_counts, prior_pass)
        ]
        denom_labels = [f"{p}/{d}" for p, d in zip(cond_pass_counts, prior_pass)]
        y4 = np.arange(len(STAGES))

        ax4.barh(y4, [100] * len(STAGES), height=bar_h, color=MISS, zorder=2, linewidth=0)
        ax4.barh(y4, cond_pcts, height=bar_h, color=PASS, zorder=3, linewidth=0)

        for i, (pct, label) in enumerate(zip(cond_pcts, denom_labels)):
            if pct >= 12:
                ax4.text(pct - 1.5, i, f"{pct:.1f}%",
                         va="center", ha="right", fontsize=9,
                         fontweight="bold", color="white", zorder=4)
            else:
                ax4.text(pct + 1.5, i, f"{pct:.1f}%",
                         va="center", ha="left", fontsize=9,
                         fontweight="bold", color=SUBTEXT, zorder=4)
            ax4.text(102, i, label, va="center", ha="left", fontsize=8, color=SUBTEXT)

        ax4.set_yticks(y4)
        ax4.set_yticklabels([s.capitalize() for s in STAGES], fontsize=10)
        ax4.set_xlim(0, 119)
        ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
        ax4.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax4.set_axisbelow(True)
        ax4.spines["left"].set_visible(False)
        ax4.tick_params(left=False)
    else:
        ax4.set_visible(False)

    # ── Footer: legend for panel 3 (left) + timestamp (right) ────────────────
    legend_handles = [
        mpatches.Patch(color=ATT,  label="Attempted"),
        mpatches.Patch(color=PASS, label="Stage passed"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.01),
        ncol=2,
        fontsize=8,
        frameon=False,
        labelcolor=SUBTEXT,
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    fig.text(0.03, 0.01, f"Generated {now}", ha="left", va="bottom",
             fontsize=8, color=SUBTEXT)

    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight",
                    transparent=presentation)
        print(f"Graph saved to {out_path}")
    else:
        plt.show()


def write_csv(results: list[FunctionResult], stages: list[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function"] + stages)
        for r in results:
            w.writerow([r.name] + [r.stage_status.get(s, "missing") for s in stages])
    print(f"CSV written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="restrict to a single stage")
    parser.add_argument("--failures", action="store_true",
                        help="only show failure details, skip per-function table")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="also write results to a CSV file")
    parser.add_argument("--graph", action="store_true",
                        help="show a matplotlib progress chart")
    parser.add_argument("--graph-out", metavar="FILE", default=None,
                        help="save graph to FILE instead of displaying it (implies --graph)")
    parser.add_argument("--presentation", action="store_true",
                        help="transparent background and cleaner styling for slides")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else list(STAGES)
    results = collect(stages)

    print(f"Generated-tests root: {OUT_ROOT}")
    print(f"Functions found: {len(results)}  |  Stages: {', '.join(stages)}")

    print_report(results, stages, failures_only=args.failures)

    if args.csv:
        write_csv(results, stages, args.csv)

    if args.graph or args.graph_out:
        plot_graph(results, stages, args.graph_out, presentation=args.presentation)


if __name__ == "__main__":
    main()

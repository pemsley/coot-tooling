"""Aggregate per-function gemmi ports into a single compilable file pair.

Given a list of qnames that have already been processed by generate_gemmi,
this module collects their individual function.hh / function.cc outputs and
merges them into:

  OUT_ROOT/_aggregated/<stem>_gemmi.hh
  OUT_ROOT/_aggregated/<stem>_gemmi.cc   (only if any function.cc files exist)

where <stem> is derived from the original source file name.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..oracle.generate import OUT_ROOT, sanitize_name

_PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\s*$", re.MULTILINE)
_INCLUDE_RE     = re.compile(r"^\s*#\s*include\s+.*$", re.MULTILINE)


def _split_header(text: str) -> tuple[list[str], str]:
    """Return (include_lines, body_without_pragma_and_includes)."""
    includes = _INCLUDE_RE.findall(text)
    body = _PRAGMA_ONCE_RE.sub("", text)
    body = _INCLUDE_RE.sub("", body)
    # Collapse runs of blank lines left behind after stripping directives.
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return includes, body


def _dedup_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _stem_from_file(source_file: str) -> str:
    """'src/coot/geometry.cc' → 'geometry'"""
    return Path(source_file).stem


def aggregate_gemmi_files(
    entries: list[tuple[str, str | None]] | list[str],
    source_file: str,
    out_root: Path = OUT_ROOT,
) -> tuple[Path, Path | None]:
    """Merge all per-function gemmi outputs into combined header/source files.

    `entries` is either a list of (qname, sig_hash) tuples (new style,
    one entry per overload) or a list of bare qnames (legacy callers).

    Returns (hh_path, cc_path) where cc_path is None if no function.cc existed.
    Skips entries whose gemmi/function.hh is absent (e.g. the gemmi stage failed).
    """
    stem = _stem_from_file(source_file)
    agg_dir = out_root / "_aggregated"
    agg_dir.mkdir(exist_ok=True)

    hh_includes: list[str] = []
    hh_bodies:   list[str] = []
    cc_includes: list[str] = []
    cc_bodies:   list[str] = []
    has_cc = False
    missing: list[str] = []

    normalized: list[tuple[str, str | None]] = []
    for e in entries:
        if isinstance(e, tuple):
            normalized.append(e)
        else:
            normalized.append((e, None))

    for qname, sig_hash in normalized:
        gemmi_dir = out_root / sanitize_name(qname, sig_hash) / "gemmi"
        label = qname if sig_hash is None else f"{qname} [{sig_hash}]"

        hh_path = gemmi_dir / "function.hh"
        if not hh_path.exists():
            missing.append(label)
            continue

        incs, body = _split_header(hh_path.read_text())
        hh_includes.extend(incs)
        if body:
            hh_bodies.append(f"// --- {label} ---\n{body}")

        cc_path = gemmi_dir / "function.cc"
        if cc_path.exists():
            has_cc = True
            incs_cc, body_cc = _split_header(cc_path.read_text())
            cc_includes.extend(incs_cc)
            if body_cc:
                cc_bodies.append(f"// --- {label} ---\n{body_cc}")

    if missing:
        print(f"[aggregate] skipped {len(missing)} entries with no gemmi output: "
              + ", ".join(m.rsplit("::", 1)[-1] for m in missing))

    if not hh_bodies:
        print(f"[aggregate] nothing to aggregate for {source_file}")
        return agg_dir / f"{stem}_gemmi.hh", None

    # ── build .hh ─────────────────────────────────────────────────────────────
    hh_lines: list[str] = ["#pragma once", ""]
    for inc in _dedup_ordered(hh_includes):
        hh_lines.append(inc)
    hh_lines.append("")
    hh_lines.append("\n\n".join(hh_bodies))
    hh_lines.append("")

    hh_out = agg_dir / f"{stem}_gemmi.hh"
    hh_out.write_text("\n".join(hh_lines))

    # ── build .cc (only if at least one function had a .cc) ───────────────────
    cc_out: Path | None = None
    if has_cc:
        cc_lines: list[str] = [f'#include "{stem}_gemmi.hh"', ""]
        # Exclude includes already pulled into the header.
        hh_inc_set = {i.strip() for i in hh_includes}
        extra_cc_incs = [i for i in cc_includes if i.strip() not in hh_inc_set]
        for inc in _dedup_ordered(extra_cc_incs):
            cc_lines.append(inc)
        if extra_cc_incs:
            cc_lines.append("")
        cc_lines.append("\n\n".join(cc_bodies))
        cc_lines.append("")

        cc_out = agg_dir / f"{stem}_gemmi.cc"
        cc_out.write_text("\n".join(cc_lines))

    return hh_out, cc_out

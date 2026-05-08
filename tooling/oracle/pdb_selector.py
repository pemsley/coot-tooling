"""Select the most appropriate example PDB file for a given function.

Add new entries to PDB_CATALOG and _KEYWORD_MAP as new example PDB files are
created in test-data/.  The selector is intentionally simple: keyword matching
on the function qualified name, source code, and doc comment.  If no keyword
fires, the LLM receives the full catalog so it can choose the appropriate file
itself.
"""
from __future__ import annotations

import re
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test-data"

# Registry of available PDB files.  Key = filename, value = one-line description
# shown to the LLM when the choice is ambiguous.  Add new entries here.
PDB_CATALOG: dict[str, str] = {
    "example.pdb": (
        "standard protein (transferase 2VTQ, chains A+B, no hydrogens)"
    ),
    "example_alphafold.pdb": (
        "AlphaFold predicted model (pLDDT confidence scores as B-factors, no CRYST1 record)"
    ),
    "example-hydrogen.pdb": (
        "protein with explicit hydrogen atoms added"
    ),
}

# Priority-ordered list of (keywords, pdb_filename).  First match wins.
# Only files that exist in test-data/ are considered.
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (
        [
            "alphafold", "alpha_fold", "plddt", "confidence_score",
            "af2", "af3", "af_", "predicted_structure", "alphafold_score",
            "is_alphafold", "b_factor_type",
        ],
        "example_alphafold.pdb",
    ),
    (
        [
            "hydrogen", "deuterium", "proton", "h_bond", "hbond",
            "add_h", "delete_h", "protonate", "deprotonate", "polar_h",
            "riding_h", "named_h", "nh_", "_nh", "oh_", "_oh",
        ],
        "example-hydrogen.pdb",
    ),
]

_DEFAULT_PDB = "example.pdb"


def _tokens(text: str) -> str:
    """Lowercase and replace non-alphanumeric runs with spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def select_pdb(
    function_qname: str,
    source_code: str = "",
    doc_comment: str = "",
) -> tuple[str, bool]:
    """Return (pdb_filename, is_certain).

    is_certain=True  → keyword matched; use this file and tell the LLM why.
    is_certain=False → no clear match; use the default but give the LLM the
                       full catalog so it can choose the right path itself.
    """
    haystack = _tokens(f"{function_qname} {source_code} {doc_comment}")
    for keywords, pdb_file in _KEYWORD_MAP:
        if pdb_file not in PDB_CATALOG:
            continue
        if not (TEST_DATA_DIR / pdb_file).exists():
            continue
        if any(kw in haystack for kw in keywords):
            return pdb_file, True
    return _DEFAULT_PDB, False


def pdb_path(filename: str) -> Path:
    return TEST_DATA_DIR / filename


def catalog_note(selected_file: str | None = None) -> str:
    """One-line entries for every available PDB file.

    Returns an empty string when only one PDB file exists (nothing to choose
    from).  Used in the LLM prompt when the choice is ambiguous so the model
    can pick a more appropriate file itself.
    """
    entries: list[tuple[str, str]] = []
    for fname, desc in PDB_CATALOG.items():
        if (TEST_DATA_DIR / fname).exists():
            entries.append((fname, desc))

    if len(entries) <= 1:
        return ""

    lines: list[str] = []
    for fname, desc in entries:
        marker = " ← default" if fname == selected_file else ""
        lines.append(f"  {TEST_DATA_DIR / fname}  — {desc}{marker}")
    return "\n".join(lines)

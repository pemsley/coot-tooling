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
    "example-alphafold.pdb": (
        "AlphaFold predicted model (pLDDT confidence scores as B-factors, no CRYST1 record)"
    ),
    "example-hydrogen.pdb": (
        "protein with explicit hydrogen atoms added"
    ),
    "example-ligand.pdb": (
        "small-molecule ligand only (no protein), for restraint generation / rdkit / SMILES workflows"
    ),
    "example-protein-ligand.cif": (
        "protein-ligand complex (protein + bound small molecule), for pli / flev / contact / binding-site workflows"
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
        "example-alphafold.pdb",
    ),
    (
        [
            "hydrogen", "deuterium", "proton", "h_bond", "hbond",
            "add_h", "delete_h", "protonate", "deprotonate", "polar_h",
            "riding_h", "named_h", "nh_", "_nh", "oh_", "_oh",
        ],
        "example-hydrogen.pdb",
    ),
    (
        [
            "pli", "flev", "ligand_interaction", "protein_ligand",
            "ligand_environment", "binding_site", "ligand_contact",
            "ligand_water", "residues_near", "ligand_neighbour",
            "contact_dots", "ligand_to_protein",
        ],
        "example-protein-ligand.cif",
    ),
    (
        [
            "rdkit", "smiles", "mol_file", "monomer_restraint",
            "get_torsion", "ligand_only", "generate_restraint",
            "dictionary_entry", "cif_dictionary", "mogul",
            "acedrg", "ligand_builder",
        ],
        "example-ligand.pdb",
    ),
]

# Optional structural notes per file — shown to the LLM alongside the file path
# so it knows where key molecules are located.  Leave a key out (or set to "")
# to suppress the note for that file.
_PDB_NOTES: dict[str, str] = {
    "example-protein-ligand.cif": (
        "The ligand LZA (residue 1299, chain A) is the bound small molecule of interest."
    ),
}

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


def structural_note(filename: str) -> str:
    """Return a file-specific structural note (e.g. ligand location) for the LLM prompt.

    Returns an empty string when no note is registered for *filename*.
    """
    return _PDB_NOTES.get(filename, "")


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

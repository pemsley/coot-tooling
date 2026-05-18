// coot::molecule_t instances are owned and managed by molecules_container_t.
// Access them via the mc[] operator after loading with read_pdb().
//
// DO NOT call mc.geometry_init_standard() explicitly — the constructor already
// calls it. Calling it a second time re-initialises internal tables and causes
// duplicate INFO output but no crash; still, omit it.

// NOTE: `molecules_container_t` is in the GLOBAL namespace.
// Do NOT write `coot::molecules_container_t` — that does not exist and
// will fail to compile with "no type named 'molecules_container_t' in
// namespace 'coot'".
molecules_container_t mc;

int imol = mc.read_pdb("@PDB_PATH@");
// mc[imol] is the coot::molecule_t for that molecule

// ─── Accessing the underlying mmdb::Manager from a coot::molecule_t ───────
// coot::molecule_t has NO `.mol` field. Writing `mc[imol].mol` fails with
// `error: no member named 'mol' in 'coot::molecule_t'` (the #1 oracle
// hallucination in this codebase — 26+ failed oracles used this pattern).
//
// The MMDB Manager lives at TWO equivalent locations:
//   ✅ mmdb::Manager *mol = mc[imol].get_mol();        // public accessor (preferred)
//   ✅ mmdb::Manager *mol = mc[imol].atom_sel.mol;     // direct field access
//                                                     // (works because oracle.cc
//                                                     //  is compiled with
//                                                     //  -fno-access-control)
//   ❌ mmdb::Manager *mol = mc[imol].mol;              // does not compile
//
// `atom_sel` is a `coot::atom_selection_container_t` (header:
// "coot-utils/atom-selection-container.hh"). Its `.mol` field is the
// `mmdb::Manager *`, alongside `n_selected_atoms`, `atom_selection`
// (PPAtom), `SelectionHandle`, `UDDAtomIndexHandle`, etc.
//
// Example traversal from a loaded coot::molecule_t:
//   mmdb::Manager *mol  = mc[imol].get_mol();
//   mmdb::Model   *m    = mol->GetModel(1);
//   mmdb::Chain   *c    = m->GetChain(0);
//   mmdb::Residue *r    = c->GetResidue(0);
//   mmdb::Atom    *a    = r->GetAtom(0);

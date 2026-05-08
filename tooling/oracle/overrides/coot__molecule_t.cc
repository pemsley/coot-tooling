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

// *** REQUIRED CONSTRUCTION PATTERN — DO NOT DEVIATE ***
// coot::protein_geometry must ALWAYS be initialised with init_standard().
// The constructor alone leaves all dictionary tables empty; any call to
// get_monomer_restraints / get_monomer_restraints_at_least_minimal / etc.
// will silently return empty/false results without it.
//
// Use EXACTLY this two-line pattern — no other init method is correct here:
//
//   coot::protein_geometry geom;
//   geom.init_standard();
//
// Do NOT call build_residue_restraints, init_ccp4srs, init_refmac_mon_lib,
// or any other init variant unless the function under test explicitly requires
// a non-standard dictionary. For standard amino acids and ligands,
// init_standard() is the only correct choice.
//
// init_standard() auto-detects $COOT_DATA_DIR or the compiled-in prefix.
// It is idempotent but slow — call it once per oracle program.

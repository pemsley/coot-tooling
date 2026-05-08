// Construct and initialise a molecules_container_t, then load a PDB and map.
// geometry_init_standard() must be called before reading any molecules.
molecules_container_t mc;
mc.geometry_init_standard();

int imol     = mc.read_pdb("@PDB_PATH@");
int imol_map = mc.read_mtz("@TEST_DATA_DIR@/example.mtz", "FWT", "PHWT", "W", false, false);

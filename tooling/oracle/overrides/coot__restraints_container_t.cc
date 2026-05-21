// coot::restraints_container_t requires a populated mmdb::Manager and a
// vector of residues. It is constructed once per refinement region.
//
// The residues-vector constructor internally calls init_shared_post(), which:
//   - registers UDD "atom_array_index" on the mol (sets udd_atom_index_handle)
//   - fills fixed_atom_indices
// Do NOT set udd_atom_index_handle or other private fields manually —
// they are populated as a side-effect of construction.

// THIS DEFINE MUST BE AT THE TOP OF THE FILE OR IT WILL NOT COMPILE.
#define HAVE_BOOST_BASED_THREAD_POOL_LIBRARY


```cpp
molecules_container_t mc;
int imol     = mc.read_pdb("@PDB_PATH@");

mmdb::Manager *mol = mc.get_mol(imol);

coot::protein_geometry pg;
pg.init_standard();

clipper::Xmap<float> dummy_xmap;

std::vector<std::pair<bool,mmdb::Residue *> > residues;
residues.push_back(std::pair<bool,mmdb::Residue *>(false, residue_p));

coot::restraints_container_t restraints(residues, {}, pg, mol, {},  &dummy_xmap);

// restraint_usage_Flags flags = coot::BONDS_ANGLES_PLANES_NON_BONDED_AND_CHIRALS;
coot::restraint_usage_Flags flags = coot::BONDS_ANGLES_TORSIONS_PLANES_NON_BONDED_AND_CHIRALS;
coot::pseudo_restraint_bond_type pseudos = coot::NO_PSEUDO_BONDS;
bool do_internal_torsions = true;
bool do_trans_peptide_restraints = true;

int n_threads = 1;

ctpl::thread_pool thread_pool(n_threads);
restraints.thread_pool(&thread_pool, n_threads);

restraints.make_restraints(imol, pg, flags, do_internal_torsions,
                           do_trans_peptide_restraints, 0, true, false, false, true, pseudos);
```
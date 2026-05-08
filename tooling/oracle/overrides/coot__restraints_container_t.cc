// coot::restraints_container_t requires a populated mmdb::Manager and a
// vector of residues. It is constructed once per refinement region.
//
// The residues-vector constructor internally calls init_shared_post(), which:
//   - registers UDD "atom_array_index" on the mol (sets udd_atom_index_handle)
//   - fills fixed_atom_indices
// Do NOT set udd_atom_index_handle or other private fields manually —
// they are populated as a side-effect of construction.
mmdb::Manager *mol = new mmdb::Manager();
mol->ReadCoorFile("@PDB_PATH@");

std::vector<std::pair<bool, mmdb::Residue *>> residues_vec;
int nModels = mol->GetNumberOfModels();
for (int imod = 1; imod <= nModels; imod++) {
   mmdb::Model *model_p = mol->GetModel(imod);
   if (!model_p) continue;
   int nChains = model_p->GetNumberOfChains();
   for (int ich = 0; ich < nChains; ich++) {
      mmdb::Chain *chain_p = model_p->GetChain(ich);
      if (!chain_p) continue;
      int nRes = chain_p->GetNumberOfResidues();
      for (int ires = 0; ires < nRes; ires++) {
         mmdb::Residue *residue_p = chain_p->GetResidue(ires);
         if (residue_p)
            residues_vec.push_back({false, residue_p});
      }
   }
}

std::vector<mmdb::Link> links;
std::vector<coot::atom_spec_t> fixed_atom_specs;
clipper::Xmap<float> xmap;

coot::protein_geometry geom;
geom.init_standard();

coot::restraints_container_t restraints(
   residues_vec, links, geom, mol, fixed_atom_specs, &xmap);

// thread_pool must be set before make_restraints — without it the call
// returns 0 restraints and restraints_vec stays empty.
int n_threads = 4;
ctpl::thread_pool tp(n_threads);
restraints.thread_pool(&tp, n_threads);

// Populate restraints_vec so atoms have bonds/angles/planes applied.
// imol=0 is a placeholder (only used for rotamer lookups).
int imol = 0;
restraints.make_restraints(imol, geom,
   coot::TYPICAL_RESTRAINTS,
   /*do_residue_internal_torsions=*/false,
   /*do_trans_peptide_restraints=*/true,
   /*rama_plot_target_weight=*/1.0,
   /*do_rama_plot_restraints=*/false,
   /*do_auto_helix_restraints=*/false,
   /*do_auto_strand_restraints=*/false,
   /*do_auto_h_bond_restraints=*/false,
   coot::NO_PSEUDO_BONDS);

// restraints is now fully set up — call the target function on it directly.

// 

molecules_container_t mc;

int success = mc.import_cif_dictionary("@PDB_PATH@");
if (!success) {
    std::cerr << "Failed to load CIF dictionary\n";
    return 1;
}

int imol_enc = mc.get_imol_enc_any();
int imol = mc.get_monomer_from_dictionary("LZA", imol_enc, true);

mmdb::Manager *mol = mc.get_mol(imol);

mmdb::Residue* res = mol->GetModel(1)->GetChain(0)->GetResidue(0);

coot::protein_geometry pg;
pg.import_cif_dictionary_ligand("@PDB_PATH@");

RDKIT::RWMol mol = rdkit_mol(res, imol_enc, pg);

// If you need to test this mol.
std::string smiles = RDKIT::MolToSmiles(mol);
std::cout << "SMILES: " << smiles << "\n";
std::cout << "Number of atoms: " << mol.getNumAtoms() << "\n";
std::cout << "Number of bonds: " << mol.getNumBonds() << "\n";
std::cout << "Number of rings: " << mol.getRingInfo()->numRings() << "\n";
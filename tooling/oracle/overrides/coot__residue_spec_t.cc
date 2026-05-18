// coot::residue_spec_t — plain value type, declared in
// "geometry/residue-and-atom-specs.hh". Always construct as a value
// (never `new coot::residue_spec_t(...)`); pass by const reference.
//
// Fields (all public):
//   int         model_number;     // mmdb::MinInt4 if unset
//   std::string chain_id;
//   int         res_no;           // mmdb::MinInt4 if unset
//   std::string ins_code;         // "" (empty) when there is no insertion code
//   int         int_user_data;
//   float       float_user_data;
//   std::string string_user_data;
//
// Construction — pick the constructor that matches what the caller has:
//
//   coot::residue_spec_t spec("A", 42);                    // chain + resno
//   coot::residue_spec_t spec("A", 42, "");                // + empty ins_code
//   coot::residue_spec_t spec(1, "A", 42, "");             // + model number
//   coot::residue_spec_t spec(42);                         // resno only (rare)
//   coot::residue_spec_t spec;                             // default — sets
//                                                          // res_no/model_number
//                                                          // to mmdb::MinInt4
//
//   // From an mmdb::Residue pointer — handles null safely:
//   mmdb::Residue *r = mol->GetModel(1)->GetChain(0)->GetResidue(0);
//   coot::residue_spec_t spec(r);
//
//   // From an atom_spec_t — copies chain_id, res_no, ins_code, model_number:
//   coot::residue_spec_t spec(some_atom_spec);
//
// Common idioms:
//   spec.unset_p()       // true if res_no == mmdb::MinInt4
//   spec.empty()         // alias for unset_p()
//   spec.next()          // residue_spec_t with res_no + 1
//   spec.previous()      // residue_spec_t with res_no - 1
//   spec == other        // compares chain_id + res_no + ins_code
//   spec <  other        // total order for std::map / std::set keys
//
// PRINTING — there is an `operator<<` for residue_spec_t, but it is declared
// in residue-and-atom-specs.hh. If your oracle does:
//   std::cout << "OUTPUT spec: " << spec << std::endl;
// and gets "invalid operands to binary expression", switch to:
//   std::cout << "OUTPUT spec: "
//             << spec.chain_id << "/" << spec.res_no
//             << (spec.ins_code.empty() ? "" : spec.ins_code) << std::endl;
//
// INSERTION CODE caveat: MMDB returns "" (empty std::string) for residues
// with no insertion code. When constructing manually, pass "" — NOT " " (space).
// The space/empty distinction matters when porting to gemmi later (gemmi uses
// ' ' for the same concept), but for the oracle stage stick with MMDB's "".

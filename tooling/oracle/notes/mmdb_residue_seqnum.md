# mmdb_residue_seqnum

## Question
When the oracle has an `mmdb::Residue *` and needs to print or assert its residue number, which getter should it call?

## Answer
Use `GetSeqNum()` — it returns the PDB sequence number (the same number used in CID strings like `//A/50`).

Do NOT use `GetResidueNo()`. That returns the 0-based index of the residue within its chain, which will silently disagree with the CID. For example, looking up `//A/50` and then printing `GetResidueNo()` may produce `40`, which looks like a CID-resolution bug but is actually the wrong getter.

If insertion codes matter for the function under test, also print `GetInsCode()` alongside `GetSeqNum()`. Don't use them unless it is clear from the context they are used, they cause issues otherwise.

Equivalent pattern:
```cpp
mmdb::Residue *res = mc[imol].get_residue("//A/50");
if (res) {
    const char *chain_id = res->GetChain()->GetChainID();
    int seqnum          = res->GetSeqNum();   // <-- not GetResidueNo()
    const char *ins     = res->GetInsCode();  // may be ""
    const char *resname = res->GetResName();
    std::cout << "OUTPUT residue: " << chain_id << " " << seqnum << ins << " " << resname << std::endl;
}
```

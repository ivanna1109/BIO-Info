from rdkit.Chem import Draw 
from rdkit.Chem import AllChem
from rdkit import Chem
import os

def visualize_rdkit_molecule(features, label, size=(450, 300), show_atom_idx=False, output_filepath='data/output_molecules/output_molecule.png'):
    preferred_name = features['preferredName'].numpy().decode('utf-8')
    num_nodes = features['num_nodes'].numpy()
    atom_features = features['node_features'].numpy()
    adjacency_indices = features['adjacency_indices'].numpy()
    bond_features = features['edge_features'].numpy()

    if output_filepath is None:
        print("Greška: 'output_filepath' mora biti definisan za spremanje slike.")
        return

    print(f"\n--- RDKit Vizualizacija molekula: '{preferred_name}' (Labela: {label.numpy():.2f}) ---")
    print(f"Broj atoma: {num_nodes}, Broj veza (u sparse formatu): {features['num_edges'].numpy()}")

    if num_nodes == 0:
        print("Molekul nema atoma za RDKit vizualizaciju.")
        return

    mol = Chem.Mol()
    editable_mol = Chem.EditableMol(mol)

    for i in range(num_nodes):
        atomic_num = int(atom_features[i][0])
        atom = Chem.Atom(atomic_num)
        editable_mol.AddAtom(atom)
    
    bond_type_map = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC
    }

    added_bonds = set()
    for i in range(0, adjacency_indices.shape[0], 2):
        u, v = adjacency_indices[i]
        
        if (u, v) in added_bonds or (v, u) in added_bonds:
            continue
        
        bond_feat = bond_features[i]
        bond_type_double = bond_feat[0]

        rdkit_bond_type = bond_type_map.get(bond_type_double, Chem.BondType.UNSPECIFIED)
        
        editable_mol.AddBond(int(u), int(v), rdkit_bond_type)
        added_bonds.add((u, v))
        added_bonds.add((v, u))

    mol = editable_mol.GetMol()

    AllChem.Compute2DCoords(mol)

    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        
        img = Draw.MolToImage(mol, size=size, fitImage=True,
                                   highlightAtoms= [], highlightBonds=[],
                                   addAtomIndices=show_atom_idx) # MolToImage ima svoj addAtomIndices
        img.save(output_filepath) # Sprema sliku

        print(f"RDKit 2D prikaz sačuvan kao: {output_filepath}")
    except Exception as e:
        print(f"Greška pri RDKit 2D vizualizaciji molekula '{preferred_name}': {e}")
        print("Pokušajte osigurati da je RDKit ispravno instaliran sa Pillow bibliotekom za crtanje.")


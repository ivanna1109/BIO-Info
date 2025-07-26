import os
import tensorflow as tf
import numpy as np
from rdkit import Chem
#from visualize_tfds_example import visualize_rdkit_molecule

def get_atom_features(atom):
    # Dummies, samo da bi RDKit mogao da vrati broj feature-a
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.IsInRingSize(3), atom.IsInRingSize(4), atom.IsInRingSize(5),
        atom.IsInRingSize(6), atom.IsInRingSize(7), atom.IsInRingSize(8),
    ]
    return np.array(features, dtype=np.float32)

def get_bond_features(bond):
    # Dummies, samo da bi RDKit mogao da vrati broj feature-a
    features = [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    return np.array(features, dtype=np.float32)

def count_labels(dataset, dataset_name):
    count_label_0 = 0
    count_label_1 = 0
    
    print(f"\n--- Brojanje labela za {dataset_name} dataset ---")
    for _, label in dataset: # Iteriramo kroz dataset; ignoriramo features, zanima nas samo labela
        if label.numpy() == 0:
            count_label_0 += 1
        elif label.numpy() == 1:
            count_label_1 += 1
        else:
            print(f"Upozorenje: Pronađena neočekivana labela: {label.numpy()}")
            
    print(f"Ukupno instanci u {dataset_name}: {count_label_0 + count_label_1}")
    print(f"  Labela 0: {count_label_0} instanci")
    print(f"  Labela 1: {count_label_1} instanci")
    
    return {'0':count_label_0, '1':count_label_1}


def load_tf_datasets(
    output_directory: str = 'tf_datasets',
    common_global_features_list: list[str] = None # Lista globalnih feature-a za dimenziju
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_path = os.path.join(output_directory, 'train_dataset')
    val_path = os.path.join(output_directory, 'val_dataset')
    test_path = os.path.join(output_directory, 'test_dataset')

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print(f"Greška: Neki od putanja datasetova ne postoje u '{output_directory}'.")
        print("Uverite se da ste prethodno sačuvali datasete.")
        return tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({})

    print(f"\n--- Počinjem učitavanje TF Datasetova iz '{output_directory}' ---")

    try:
        dummy_atom_dim = get_atom_features(Chem.Atom(0)).shape[0]
    except Exception:
        print("Upozorenje: Nije moguće dobiti dimenziju atomskih feature-a iz Chem.Atom(0). Pretpostavljam 12.")
        dummy_atom_dim = 12

    try:
        dummy_mol_for_bond_dim = Chem.MolFromSmiles('CC')
        if dummy_mol_for_bond_dim and dummy_mol_for_bond_dim.GetNumBonds() > 0:
            dummy_bond_dim = get_bond_features(dummy_mol_for_bond_dim.GetBonds()[0]).shape[0]
        else:
            raise ValueError("Could not get bond feature dimension from dummy molecule.")
    except Exception:
        print("Upozorenje: Nije moguće dobiti dimenziju veza feature-a iz Chem.Bond() ili dummy molekula. Pretpostavljam 3.")
        dummy_bond_dim = 3

    # Dimenzija globalnih karakteristika mora biti konzistentna
    if common_global_features_list is None or not common_global_features_list:
        print("Greška: common_global_features_list mora biti prosleđen da bi se odredila dimenzija globalnih karakteristika.")
        return tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({})
    
    global_features_dim = len(common_global_features_list)
    print(f"Učitavam datasete sa očekivanom dimenzijom globalnih karakteristika: {global_features_dim}")


    reloaded_output_signature = (
        {
            'node_features': tf.TensorSpec(shape=(None, dummy_atom_dim), dtype=tf.float32),
            'edge_features': tf.TensorSpec(shape=(None, dummy_bond_dim), dtype=tf.float32),
            'adjacency_indices': tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
            'adjacency_values': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'global_features': tf.TensorSpec(shape=(global_features_dim,), dtype=tf.float32),
            'num_nodes': tf.TensorSpec(shape=(), dtype=tf.int32),
            'num_edges': tf.TensorSpec(shape=(), dtype=tf.int32),
            'preferredName': tf.TensorSpec(shape=(), dtype=tf.string),
            'esr': tf.TensorSpec(shape=(), dtype=tf.string),
            'source_assay_name': tf.TensorSpec(shape=(), dtype=tf.string)
        },
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    loaded_train_ds = None
    loaded_val_ds = None
    loaded_test_ds = None

    try:
        loaded_train_ds = tf.data.Dataset.load(train_path, element_spec=reloaded_output_signature)
        print(f"Trening dataset uspešno učitan iz: {train_path}")
    except Exception as e:
        print(f"Greška pri učitavanju trening dataseta iz {train_path}: {e}")

    try:
        loaded_val_ds = tf.data.Dataset.load(val_path, element_spec=reloaded_output_signature)
        print(f"Validacioni dataset uspešno učitan iz: {val_path}")
    except Exception as e:
        print(f"Greška pri učitavanju validacionog dataseta iz {val_path}: {e}")

    try:
        loaded_test_ds = tf.data.Dataset.load(test_path, element_spec=reloaded_output_signature)
        print(f"Test dataset uspešno učitan iz: {test_path}")
    except Exception as e:
        print(f"Greška pri učitavanju test dataseta iz {test_path}: {e}")

    print("\n--- Učitavanje TF Datasetova završeno ---")
    return loaded_train_ds, loaded_val_ds, loaded_test_ds

def element_details(element, output_dir):
    try:
        for features, label in element:
            preferred_name = features['preferredName'].numpy().decode('utf-8')
            
            
            clean_preferred_name = "".join(c for c in preferred_name if c.isalnum() or c in (' ', '.', '-')).replace(' ', '_')
            
        
            image_filename = os.path.join(output_dir, f"{clean_preferred_name}_mol.png")
            text_filename = os.path.join(output_dir, f"{clean_preferred_name}_details.txt")

            text_output_path_full = text_filename # Ovo je putanja za text fajl
            text_output_dir_full = os.path.dirname(text_output_path_full)
            if text_output_dir_full and not os.path.exists(text_output_dir_full):
                os.makedirs(text_output_dir_full, exist_ok=True)
            
            with open(text_output_path_full, 'w') as f: # Otvaramo fajl za pisanje ('w')
                f.write(f"--- Detalji molekula: '{preferred_name}' ---\n")
                f.write(f"Naziv molekula (preferredName): {preferred_name}\n")
                f.write(f"Labela (hitc): {label.numpy()}\n")
                f.write(f"ESR: {features['esr'].numpy().decode('utf-8')}\n")
                f.write(f"Broj atoma : {features['num_nodes'].numpy()}\n")
                f.write(f"Broj veza: {features['num_edges'].numpy()}\n")
                f.write(f"Globalne karakteristike: {features['global_features'].numpy()}\n")
                f.write(f"Shape node_features: {features['node_features'].shape}\n")
                f.write(f"Shape edge_features: {features['edge_features'].shape}\n")
                f.write("\nMatrica susedsta (Sparse reprezentacija):\n")
                f.write(f"Adjacency_indices:\n{features['adjacency_indices'].numpy()}\n")
                f.write(f"Adjacency_values:\n{features['adjacency_values'].numpy()}\n")
                
                num_nodes = features['num_nodes'].numpy()
                adj_matrix_dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                indices = features['adjacency_indices'].numpy()
                values = features['adjacency_values'].numpy()
                for i, (u, v) in enumerate(indices):
                    adj_matrix_dense[u, v] = values[i]
                f.write("\nMatrica susedstva (Gusta reprezentacija - prvih 5x5 ako je veća):\n")
                if num_nodes <= 5:
                    f.write(f"{np.array_str(adj_matrix_dense)}\n")
                else:
                    f.write(f"{np.array_str(adj_matrix_dense[:5, :5])}\n")
                    f.write(f"    ... (Prikazano samo prvih 5x5 od {num_nodes}x{num_nodes} matrice)\n")
            
            print(f"Detalji molekula sačuvani kao: {text_output_path_full}")
            """
            visualize_rdkit_molecule(features, label, 
                                    show_atom_idx=True, 
                                    output_filepath=image_filename)
                                    """
            
    except Exception as e:
        print(f"Greška pri obradi elementa: {e}")


        """
if __name__ == "__main__":
    COMMON_GLOBAL_FEATURES_LIST = [
        'monoisotopicMass', 'ac50',]

    dataset_dir = 'data/tfrecords'

    train_ds, val_ds, test_ds = load_tf_datasets(
        output_directory=dataset_dir,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST
    )

    if train_ds is not None and val_ds is not None and test_ds is not None:
        print("Ucitani svi skupovi podataka.")

        ### Veličine Skupova Podataka
        
        print("\n--- Veličine TF Datasetova ---")
        train_size = train_ds.cardinality().numpy()
        print(f"Veličina trening skupa: {train_size}")
        count_labels(train_ds, "Trening")

        val_size = val_ds.cardinality().numpy()
        print(f"Veličina validacionog skupa: {val_size}")
        count_labels(val_ds, "Validacija")

        test_size = test_ds.cardinality().numpy()
        print(f"Veličina test skupa: {test_size}")
        count_labels(test_ds, "Test")
        
        ### Detalji Prvog Elementa Trening Skupa
        
        print("\n--- Provera prvog elementa iz UČITANOG TRENING SKUPA ---")
        output_dir = 'data/output_molecules/train_example'
        #element_details(train_ds.take(1), output_dir)
    
        ### Detalji Prvog Elementa Test Skupa
        output_dir = 'data/output_molecules/test_example'
        print("\n--- Provera prvog elementa iz UČITANOG TEST SKUPA ---")
        #element_details(test_ds.take(1), output_dir)
    else:
        print("\nNije moguće učitati sve datasete. Proverite poruke o greškama iznad.")

"""
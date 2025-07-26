import tensorflow as tf
import deepchem as dc
from rdkit import Chem
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# --- Inicijalizacija DeepChem Featurizera ---
featurizer = dc.feat.MolGraphConvFeaturizer()

# --- Definisanje dimenzija (standardne vrednosti za MolGraphConvFeaturizer) ---
NODE_FEATURE_DIM = 75
EDGE_FEATURE_DIM = 14

# --- Definisanje putanja i kolona (VAŽNO: PRILAGODITE OVO!) ---
single_csv_file_path = 'data/binders/NVS_NR_hER-ESR1.csv' # Vaš glavni CSV fajl sa podacima za SVE molekule
all_mol_files_directory = 'data/mols_from_original_csv/NVS_NR_hER-ESR1'   # <-- OVO JE NOVO/VAŽNO: Roditeljski folder koji sadrži podfoldere poput 'csv_fajl_1/', 'csv_fajl_2/', itd.

# Kolona u CSV-u koja se podudara sa imenom MOL fajla (bez .mol ekstenzije)
molecule_name_csv_column = 'preferredName' 

# VAŠA CILJANA VARIJABLA (LABELA) JE 'hitc'
your_label_column = 'hitc' 

# KOLONA ZA STRATIFIKACIJU (takođe 'hitc' jer je vaša ciljana varijabla binarna)
stratification_column = 'hitc' 

additional_properties_to_include = [
    'ac50',             
    'averageMass',      
    'monoisotopicMass',
    'multicomponent',
    'stereo'
]
GLOBAL_PROPERTIES_DIM = len(additional_properties_to_include)

# Definisanje output_signature za from_generator
output_signature = (
    (
        tf.TensorSpec(shape=(None, NODE_FEATURE_DIM), dtype=tf.float32),  # node_features
        tf.TensorSpec(shape=(None, EDGE_FEATURE_DIM), dtype=tf.float32),  # edge_features
        tf.TensorSpec(shape=(2, None), dtype=tf.int32)                    # edge_index
    ),
    tf.TensorSpec(shape=(GLOBAL_PROPERTIES_DIM,), dtype=tf.float32), # Oblik globalnih atributa (fiksna dimenzija)
    tf.TensorSpec(shape=(), dtype=tf.float32)                         # Oblik labele (skalar, npr. 0.0 ili 1.0)
)


# --- MODIFIKOVANA FUNKCIJA ZA GENERATOR PODATAKA (prihvata mapu mol imena na putanje) ---
def load_molecular_data_for_split(
    single_csv_df,            # Prosleđujemo već učitan DataFrame
    mol_name_to_path_map,     # Mapa mol imena na njihove pune putanje (kreirana ranije pomoću os.walk)
    label_column_name,
    additional_properties_columns,
    molecules_for_this_split_set # set imena molekula za ovaj split
):
    """
    Generator koji učitava podatke o molekulima iz DataFrame-a i koristi mapu putanja MOL fajlova,
    ali obrađuje samo molekule čija su imena navedena u 'molecules_for_this_split_set'.
    """
    if additional_properties_columns is None:
        additional_properties_columns = []

    print(f" Procesiram {len(molecules_for_this_split_set)} molekula za ovaj split...")
    processed_count = 0
    skipped_count = 0

    # Iteriramo samo kroz imena molekula koja su predviđena za ovaj split
    for mol_name_from_split_set in molecules_for_this_split_set:
        mol_file_path = mol_name_to_path_map.get(mol_name_from_split_set)

        if mol_file_path is None:
            # Ova situacija ne bi trebala da se desi ako je mapa ispravno formirana iz validnih fajlova
            # (tj. mol_name_from_split_set bi trebalo da je došao iz mol_name_to_path_map)
            print(f"  Upozorenje: Putanja za molekul '{mol_name_from_split_set}' nije pronađena u mapi. Preskačem.")
            skipped_count += 1
            continue

        # Proveravamo da li molekul postoji u DataFrame-u
        if mol_name_from_split_set in single_csv_df.index: 
            row_data = single_csv_df.loc[mol_name_from_split_set]
            
            label = row_data[label_column_name]

            global_properties = []
            for col_name in additional_properties_columns:
                val = row_data[col_name]
                # Rukovanje NaN vrednostima: popunjavamo sa 0.0
                global_properties.append(float(val) if pd.notna(val) else 0.0)
            
            global_features_tensor = tf.constant(global_properties, dtype=tf.float32)

            try:
                mol = Chem.MolFromMolFile(mol_file_path, sanitize=True, removeHs=False)
                if mol is None:
                    skipped_count += 1
                    continue
                
                graph_data_list = featurizer.featurize([mol])
                if not graph_data_list:
                    skipped_count += 1
                    continue
                graph_data = graph_data_list[0]

                node_features = tf.constant(graph_data.node_features, dtype=tf.float32)
                edge_features = tf.constant(graph_data.edge_features, dtype=tf.float32)
                edge_index = tf.constant(graph_data.edge_index, dtype=tf.int32)

                # Labela se pretvara u float32 kako bi odgovarala output_signature
                yield ((node_features, edge_features, edge_index), global_features_tensor), tf.constant(float(label), dtype=tf.float32)
                processed_count += 1
                
            except Exception as e:
                print(f"  Greška pri obradi Mol fajla '{mol_file_path}': {e}. Preskačem.")
                skipped_count += 1
        else:
            print(f"  Upozorenje: Molekul '{mol_name_from_split_set}' nije pronađen u CSV-u. Preskačem.")
            skipped_count += 1
    
    print(f"  Generisanje podataka završeno za ovaj split. Uspešno obrađeno: {processed_count}, Preskočeno: {skipped_count}.")

# --- FUNKCIJA ZA SERIJALIZACIJU JEDNOG PRIMERA U TF.TRAIN.EXAMPLE ---
def serialize_example(graph_feats, global_feats, label):
    node_features_np = graph_feats[0].numpy()
    edge_features_np = graph_feats[1].numpy()
    edge_index_np = graph_feats[2].numpy()
    global_features_np = global_feats.numpy()
    label_np = label.numpy()

    feature = {
        'node_features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(node_features_np).numpy()])),
        'edge_features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(edge_features_np).numpy()])),
        'edge_index': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(edge_index_np).numpy()])),
        'global_features': tf.train.Feature(float_list=tf.train.FloatList(value=global_features_np.flatten())),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label_np.item()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

# --- FUNKCIJA ZA PARSIRANJE JEDNOG PRIMERA IZ TFRECORD-A ---
# (Ova funkcija je neophodna ako ćete kasnije učitavati TFRecord fajlove, iako je izbačena iz "krajnjeg" dela koda)
def parse_example(example_proto):
    feature_description = {
        'node_features': tf.io.FixedLenFeature([], tf.string),
        'edge_features': tf.io.FixedLenFeature([], tf.string),
        'edge_index': tf.io.FixedLenFeature([], tf.string),
        'global_features': tf.io.FixedLenFeature([GLOBAL_PROPERTIES_DIM], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    node_features = tf.io.parse_tensor(parsed_example['node_features'], out_type=tf.float32)
    edge_features = tf.io.parse_tensor(parsed_example['edge_features'], out_type=tf.float32)
    edge_index = tf.io.parse_tensor(parsed_example['edge_index'], out_type=tf.int32)

    node_features.set_shape([None, NODE_FEATURE_DIM])
    edge_features.set_shape([None, EDGE_FEATURE_DIM])
    edge_index.set_shape([2, None])

    global_features = parsed_example['global_features']
    label = parsed_example['label']

    return ((node_features, edge_features, edge_index), global_features), label


# --- KORAK 1: Učitavanje svih imena molekula, labela i mapiranje putanja MOL fajlova ---
print("Učitavam glavni CSV i mapiram putanje MOL fajlova iz hijerarhijske strukture...")
csv_files_lists = [
    'data/binders/NVS_NR_hER-ESR1.csv',
    'data/binders/NVS_NR_bER-ESR1.csv',
    'data/binders/NVS_NR_mERa-ESR1.csv',
    'data/binders/OT_ER_ERaERa_0480-ESR1.csv',
    'data/binders/OT_ER_ERaERa_1440-ESR1.csv',
    'data/binders/OT_ER_ERaERb_0480-ESR1-ESR2.csv',
    'data/binders/OT_ER_ERaERb_1440-ESR1-ESR2.csv',
    'data/binders/OT_ER_ERbERb_0480-ESR2.csv',
    'data/binders/OT_ER_ERbERb_1440-ESR2.csv',
    'data/binders/OT_ERa_GFPERaERE_0120-ESR1.csv',
    'data/binders/OT_ERa_GFPERaERE_0480-ESR1.csv',
    #'data/aa_csv/TOX21_ERa_BLA_Agonist_ratio-ESR1.csv',
    #'data/aa_csv/TOX21_ERa_BLA_Antagonist_ratio-ESR1.csv',
]
important_columns = ['hitc', 'ac50','preferredName']
importan_columns_tox = ['PREFERRED NAME', 'HIT CALL', 'AC50']
csv_files_map = {}
try:
    for csv_file in csv_files_lists:
        csv_pd = pd.read_csv(csv_file)
        if not all(col in csv_pd.columns for col in important_columns):
            raise ValueError(f"CSV fajl mora sadržati kolone iz liste important columns.")
        csv_pd.set_index('preferredName', inplace=True)
        csv_name = csv_file.split('/')[-1].split('.')[0]
        csv_files_map[csv_name] = csv_pd
    print(f'Shape mape CSV-a {len(csv_files_map)}')
    #print(f"Mapa CSV-ova: {csv_files_map}")
    
    # Kreiranje mape mol imena na pune putanje fajlova pomoću os.walk
    mol_name_to_path_map = {}
    total_mol_files_found = 0
    
    # Prolazimo kroz sve podfoldere i fajlove unutar all_mol_files_directory
    for root, dirs, files in os.walk(all_mol_files_directory):
        for file in files:
            if file.endswith(".mol"):
                mol_name = os.path.splitext(file)[0] # Ime fajla bez .mol ekstenzije
                full_mol_path = os.path.join(root, file) # Puna putanja do .mol fajla
                mol_name_to_path_map[mol_name] = full_mol_path
                total_mol_files_found += 1
    for key, value in mol_name_to_path_map.items():
        print(f"{key}:{value}")
        break           
    print(f"Pronađeno {total_mol_files_found} MOL fajlova u strukturi podfoldera.")
except Exception as e:
    print(f"greska u prvom delu: {e}")
""""
    # Sada filtriramo molekule koji imaju i CSV unos i MOL fajl na disku
    final_mol_names_for_split = []
    final_stratification_labels = []

    for mol_name_from_csv in main_df.index:
        if mol_name_from_csv in mol_name_to_path_map:
            # Proveriti da li labela za stratifikaciju nije NaN (ovo je već urađeno sa dropna)
            final_mol_names_for_split.append(mol_name_from_csv)
            final_stratification_labels.append(main_df.loc[mol_name_from_csv, stratification_column])
    
    if not final_mol_names_for_split:
        raise ValueError("Nema validnih molekula (CSV podaci + MOL fajl) za podelu nakon provere.")

    print(f"Pronađeno {len(final_mol_names_for_split)} molekula sa validnim podacima za podelu nakon usklađivanja CSV-a i MOL fajlova.")

except Exception as e:
    print(f"FATALNA GREŠKA pri učitavanju podataka za podelu: {e}")
    exit() # Prekid programa ako se ne mogu učitati podaci

# --- KORAK 2: Stratifikovana podela na train, validation i test ---
RANDOM_SEED = 42 # Fiksni seed za reproduktivnost
TEST_SPLIT_RATIO = 0.15 # 15% za test
VAL_SPLIT_RATIO_FROM_TRAIN_VAL = 0.15 / (1 - TEST_SPLIT_RATIO) # 15% validacije od preostalog (trening+validacija) dela

print(f"\nObavljam stratifikovanu podelu (Train:{1 - TEST_SPLIT_RATIO - (1-TEST_SPLIT_RATIO)*VAL_SPLIT_RATIO_FROM_TRAIN_VAL:.2f}, Val:{ (1-TEST_SPLIT_RATIO)*VAL_SPLIT_RATIO_FROM_TRAIN_VAL:.2f}, Test:{TEST_SPLIT_RATIO:.2f}) po koloni '{stratification_column}'...")

# Prva podela: train_val vs test
train_val_names, test_names, train_val_labels, test_labels = train_test_split(
    final_mol_names_for_split, final_stratification_labels,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_SEED,
    stratify=final_stratification_labels # STRATIFIKACIJA!
)

# Druga podela: train vs val
train_names, val_names, _, _ = train_test_split(
    train_val_names, train_val_labels, # Koristimo labele train_val za stratifikaciju
    test_size=VAL_SPLIT_RATIO_FROM_TRAIN_VAL,
    random_state=RANDOM_SEED,
    stratify=train_val_labels # STRATIFIKACIJA!
)

print(f"Broj uzoraka: Trening={len(train_names)}, Validacija={len(val_names)}, Test={len(test_names)}")


# --- KORAK 3 & 4: Kreiranje i čuvanje svakog TFDataSeta u poseban TFRecord fajl ---
output_tfrecord_dir = 'data/tfrecords_splits'
os.makedirs(output_tfrecord_dir, exist_ok=True)

splits_to_process = {
    'train': train_names,
    'val': val_names,
    'test': test_names
}

for split_name, molecules_list_for_split in splits_to_process.items():
    output_tfrecord_path = os.path.join(output_tfrecord_dir, f'molecular_graph_{split_name}.tfrecord')
    print(f"\nKreiram i zapisujem {split_name} dataset u {output_tfrecord_path}...")

    # Kreiramo raw dataset za trenutni split
    current_raw_dataset = tf.data.Dataset.from_generator(
        lambda: load_molecular_data_for_split(
            single_csv_df=main_df, 
            mol_name_to_path_map=mol_name_to_path_map, # Prosleđujemo mapu putanja!
            label_column_name=your_label_column, 
            additional_properties_columns=additional_properties_to_include,
            molecules_for_this_split_set=set(molecules_list_for_split) 
        ),
        output_signature=output_signature
    )

    # Zapisujemo dataset u TFRecord fajl
    try:
        with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
            for ((node_feats, edge_feats, edge_idx), global_feats), label in current_raw_dataset:
                serialized_example = serialize_example((node_feats, edge_feats, edge_idx), global_feats, label)
                writer.write(serialized_example)
        print(f"{split_name.capitalize()} dataset uspešno zapisan u TFRecord.")
    except Exception as e:
        print(f"Greška prilikom zapisivanja {split_name} TFRecord-a: {e}")

print("\nSvi TFRecord fajlovi za trening, validaciju i test su kreirani.")
"""
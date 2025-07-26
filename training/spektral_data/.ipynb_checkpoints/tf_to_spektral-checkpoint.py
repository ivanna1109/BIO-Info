import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from spektral.data import Graph

def convert_tf_dataset_to_spektral(tf_dataset):
    graphs = []
    labels = []
    
    print(f"Pretvaranje TF Dataset-a u Spektral Graph objekte i One-Hot enkodiranje labela...")
    
    for features, label in tf_dataset:
        # Pretvori labele u NumPy array i one-hot enkodiranje
        # Pretpostavljamo da je 'label' skalarna vrednost (0 ili 1)
        label_numpy = label.numpy()
        
        # One-hot enkodiranje za binarne klase (0 ili 1)
        # 0 -> [1., 0.]
        # 1 -> [0., 1.]
        one_hot_label = np.zeros(2, dtype=np.float32)
        if label_numpy == 0:
            one_hot_label[0] = 1.0
        elif label_numpy == 1:
            one_hot_label[1] = 1.0
        else:
            print(f"Upozorenje: Neočekivana labela: {label_numpy}. One-hot enkodiranje možda neće biti ispravno.")
        
        labels.append(one_hot_label)

        # Izvuci podatke iz TF features za Spektral Graph
        x = features['node_features'].numpy()
        a_indices = features['adjacency_indices'].numpy()
        a_values = features['adjacency_values'].numpy()
        
        # Spektral zahteva sparse matricu susedstva za 'a'
        # Kreiramo SciPy sparse matricu iz indeksa i vrednosti
        num_nodes = features['num_nodes'].numpy()
        # Proveravamo da li su indeksi prazni
        if a_indices.size == 0:
            # Ako nema veza, kreiramo praznu sparse matricu
            a = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32).tocsr()
        else:
            # U suprotnom, kreiramo sparse matricu iz adjacency_indices i adjacency_values
            # Važno: 'coo_matrix' je dobar format za kreiranje iz liste indeksa i vrednosti
            # Prvo, kreiramo koordinate i podatke za CO_matrix
            rows = a_indices[:, 0]
            cols = a_indices[:, 1]
            data = a_values
            a = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()

        # Globalne karakteristike (u Spektralu se obično zovu 'u')
        u = features['global_features'].numpy()
        
        # Ostale informacije za Spektral Graph (ako su potrebne)
        # Npr., 'molecule_name'
        molecule_name = features['preferredName'].numpy().decode('utf-8')

        # Kreiraj Spektral Graph objekat
        graph = Graph(x=x, a=a, u=u, y=one_hot_label, molecule_name=molecule_name)
        graphs.append(graph)
        
    return graphs, np.array(labels, dtype=np.float32) # Vraćamo NumPy niz za labele
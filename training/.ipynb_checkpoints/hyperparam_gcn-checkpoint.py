import sys
import os
sys.path.append('/home/ivanam/Master-thesis')

#import data_proces.data_processing as dp
import new_data_preprocess.read_data as dp
import metrics.calculate_metrics as metric
from models.gcnModel import GCN
from models.gSageModel import GraphSageModel
import optuna
import tensorflow as tf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spektral.data import BatchLoader
from spektral.data import Dataset, Graph
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical


graphs, labels = dp.read_molecule_data()
print(f"Podaci su pribavljeni! Velicina grafova: {len(graphs)} velicina labela: {len(labels)}")
classes_num = 3

# trening i test skupovi (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.2, random_state=42, stratify=labels)
print("Podaci su podeljeni na trening i test.")

# Dalja podela na trening i validacioni skup (80% trening, 20% validacija)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify = y_train)
print("Podaci su podeljeni na trening i val.")

y_train = to_categorical(y_train, num_classes=classes_num)
y_val = to_categorical(y_val, num_classes=classes_num)
y_test = to_categorical(y_test, num_classes=classes_num)

num_features = 117  # Broj feature-a po čvoru (36 DW i 54 za atom label)
num_labels = 3     # Broj klasa
#dense_units = [128, 64]  # Fully connected slojevi sa 128 i 64 jedinice - 1. i 2. trening
#dense_units = [256, 128]
class MyDataset(Dataset):
    def __init__(self, graphs, labels, **kwargs):
        self.graphs = graphs
        self.labels = labels
        self.molecule_names = []
        print(f"Loaded {len(self.graphs)} graphs")
        super().__init__(**kwargs)

    def read(self):
        output = []
        print(f"Adjacency matrix shape: {graphs[0].a.shape}")
        for i in range(len(self.graphs)):
            #e_reshaped = np.tile(self.graphs[i].e, (2, 1))
            output.append(
            Graph(x=self.graphs[i].x, a=self.graphs[i].a,
                   y=self.labels[i], 
                   molecule_name=self.graphs[i].get('molecule_name')))
            self.molecule_names.append(self.graphs[i].get('molecule_name'))
        return output

print("Kreiranje dataset instanci na osnovu podeljenih podataka..")
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)

print(f"Number of elements in train dataset: {len(train_dataset.labels)}")
print(f"Number of elements in train dataset 1: {len(train_dataset.graphs)}")

print("Prelazimo na batch loadere...")
train_loader = BatchLoader(train_dataset, batch_size=16)
val_loader = BatchLoader(val_dataset, batch_size=16)
test_loader = BatchLoader(test_dataset, batch_size=16)
i = 0

def objective(trial):
    global i
    hidden_units = trial.suggest_int('hidden_units', 32, 256)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 16, 256) for i in range(2)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    
    # Kreiraj GCN model sa Optuna parametrima
    model = GCN(num_features, num_labels, hidden_units, dense_units, dropout_rate)

    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))   
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[metric.f1_score])  # Koristi F1-score kao metricu
    
    
    # Train the model
    history = model.fit(train_loader.load(),
                    validation_data=val_loader.load(),
                    steps_per_epoch=train_loader.steps_per_epoch,
                    validation_steps=val_loader.steps_per_epoch,
                    epochs=60,
                    batch_size=batch_size,
                    #class_weight=class_weights_dict,
                    verbose=0)
    
    print("Pisemo history treninga..")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'optuna_gcn/history_results_{i}.csv', index=False)
    i+=1
    # Evaluate the model on the testing set
    f1 = model.evaluate(test_loader.load(),
    steps=test_loader.steps_per_epoch,
    verbose=0)[1]  # F1-score je na indeksu 1
    
    return f1  # Maksimizujemo F1-score za optimizaciju

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("  Value: {:.5f}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
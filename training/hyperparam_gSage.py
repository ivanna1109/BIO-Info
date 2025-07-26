import sys
import os
sys.path.append('/home/ivanam/Master-thesis')

#import data_proces.data_processing as dp
import new_data_preprocess.read_data as dp
import metrics.calculate_metrics as metric
from models.gSage import GraphSageModel
import optuna
import tensorflow as tf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spektral.data import Dataset, Graph
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.utils import to_categorical
import math
from spektral.data import Loader, DisjointLoader



graphs, labels = dp.read_molecule_data()
print(f"Podaci su pribavljeni! Velicina grafova: {len(graphs)} velicina labela: {len(labels)}")
classes_num = 3

# trening i test skupovi (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.2, random_state=42, stratify=labels)
print("Podaci su podeljeni na trening i test.")

# Dalja podela na trening i validacioni skup (80% trening, 20% validacija)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify = y_train)
print("Podaci su podeljeni na trening i val.")

# Konvertovanje oznaka u one-hot format
y_train = to_categorical(y_train, num_classes=classes_num)
y_val = to_categorical(y_val, num_classes=classes_num)
y_test = to_categorical(y_test, num_classes=classes_num)


num_features = 117  
num_labels = 3     # Broj klasa
#dense_units = [128, 64] 
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
        print(f"Labels 0 izgled: {labels[0]}")
        for i in range(len(self.graphs)):
            dense_matrix = self.graphs[i].a
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

batch_size = 16
print("Prelazimo na batch loadere...")
train_loader = DisjointLoader(train_dataset, batch_size=batch_size)
val_loader = DisjointLoader(val_dataset, batch_size=batch_size)
test_loader = DisjointLoader(test_dataset, batch_size=batch_size)

batch = train_loader.__next__()
inputs, target = batch
x, a, i = inputs
print(x.shape)
print(a.shape)
print(target.shape)
print(target)
print("Target batch shape:", target.shape)
print("Target batch example:", target[0])
print(f"Broj grafova u batchu: {len(np.unique(i))}")  # Koliko različitih grafova ima u batch-u?
print(f"x shape: {x.shape}, a shape: {a.shape}, target shape: {target.shape}")

print("Provera izlaza modela...:")
batch = train_loader.__next__()
inputs, target = batch
x, a, _ = inputs

model = GraphSageModel(num_node_features=x.shape[1], 
                       num_classes=target.shape[1], 
                       hidden_units=64, 
                       dense_units=[64, 32], 
                       dropout_rate=0.5)

output = model(inputs)
print("Output shape modela iz batcha:", output.shape)  # Treba da bude (batch_size, num_classes)

i = 0
print("Idemo u objective funkciju....")

def objective(trial):
    global i
    hidden_units = trial.suggest_int('hidden_units', 32, 256)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 16, 256) for i in range(2)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    
    model = GraphSageModel(x.shape[1], target.shape[1], hidden_units, dense_units, dropout_rate)

    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))   
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[metric.f1_score]) 

    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    validation_steps = math.ceil(len(val_dataset) / batch_size)
    test_steps = math.ceil(len(test_dataset) / batch_size)

    print("Trening gSage-a u optune...")
    history = model.fit(train_loader.load(),
                    validation_data=val_loader.load(),
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=60,
                    batch_size=batch_size,
                    class_weight=class_weights_dict,
                    verbose=0)
    
    print("Pisemo history treninga..")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'new_training/optuna_gSage/history_results_{i}.csv', index=False)
    i+=1
    
    print("Racunamo f1.......")
    f1 = model.evaluate(test_loader.load(),
    steps=test_steps,
    verbose=0)[1]  # F1-score je na indeksu 1 
    
    return f1  


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


print("Best trial:")
trial = study.best_trial
print("  Value: {:.5f}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
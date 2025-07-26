import sys
import os

# Dodajte putanju do glavnog direktorijuma
sys.path.append('/home/ivana.milutinovic.pmfns/bio_info')

import data_preprocessing.tfds_load as tfds
import training.metrics.calculate_metrics as metric
import spektral_data.tf_to_spektral as tf_to_s
from spektral_data.spektral_dataset import MyDataset
from new_training.models.gin import GINModel
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from sklearn.metrics import average_precision_score
from spektral.data import DisjointLoader
from spektral.data import Dataset, Graph
from spektral.models import GeneralGNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import math
import numpy as np
from sklearn.metrics import classification_report
import metrics.oversampling_graphs as oversampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
#import evaluate_model as eval



COMMON_GLOBAL_FEATURES_LIST = [
        'monoisotopicMass', 'ac50',]

dataset_dir = '/home/ivana.milutinovic.pmfns/bio_info/data_preprocessing/data/tfrecords'

train_ds, val_ds, test_ds = tfds.load_tf_datasets(output_directory=dataset_dir, common_global_features_list=COMMON_GLOBAL_FEATURES_LIST)
train_size = train_ds.cardinality().numpy()
print(f"Veličina trening skupa: {train_size}")
#tfds.count_labels(train_ds, 'Training')
print("-"*56)
val_size = val_ds.cardinality().numpy()
print(f"Veličina validacionog skupa: {val_size}")
#tfds.count_labels(val_ds, 'Val')
print("-"*56)
test_size = test_ds.cardinality().numpy()
print(f"Veličina test skupa: {test_size}")
#tfds.count_labels(test_ds, 'Test')
print("-"*56)
        
print("\n--- Konvertovanje trening skupa ---")
X_train_graphs, y_train_one_hot = tf_to_s.convert_tf_dataset_to_spektral(train_ds)
        
print("\n--- Konvertovanje validacionog skupa ---")
X_val_graphs, y_val_one_hot = tf_to_s.convert_tf_dataset_to_spektral(val_ds)
        
print("\n--- Konvertovanje test skupa ---")
X_test_graphs, y_test_one_hot = tf_to_s.convert_tf_dataset_to_spektral(test_ds)

print("\nKreiranje dataset instanci na osnovu konvertovanih podataka za Spektral..")
train_dataset = MyDataset(X_train_graphs, y_train_one_hot)
val_dataset = MyDataset(X_val_graphs, y_val_one_hot)
test_dataset = MyDataset(X_test_graphs, y_test_one_hot)

print(f"\nDimenzije finalnih datasetova:")
print(f"Train dataset: {len(train_dataset)} grafova, labeli oblika: {train_dataset.labels_data.shape}")
print(f"Validation dataset: {len(val_dataset)} grafova, labeli oblika: {val_dataset.labels_data.shape}")
print(f"Test dataset: {len(test_dataset)} grafova, labeli oblika: {test_dataset.labels_data.shape}")

print(f"Broj labela u train dataset: {len(train_dataset.labels_data)}")
print(f"Broj grafova u train datasetu: {len(train_dataset.graphs_data)}")

batch_size = 50 #po optuna proracunu
print("Prelazimo na batch loadere...")
train_loader = DisjointLoader(train_dataset, batch_size=batch_size)
val_loader = DisjointLoader(val_dataset, batch_size=batch_size)
test_loader = DisjointLoader(test_dataset, batch_size=batch_size)


print("Tri skupa prilagodjena za trening, val i test kreirani da bi mogli pustiti u GIN.")
num_features = 117  # Broj feature-a po čvoru
num_labels = 2     # Broj klasa
hidden_units = 203  # Skriveni slojevi gSage
dense_units = [79, 27]
dropout_rate = 0.28

model = GINModel(num_features, hidden_units, dense_units, dropout_rate, classes_num)
print("Kreiran GIN model.")

# Dodajemo LearningRateScheduler i EarlyStopping sheduler za monitoring treninga
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#custom call back za pracene f1 val score-a
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_f1 = logs.get('val_f1_score')
        if val_f1 is not None:
            print(f'\nEpoch {epoch + 1}: val_f1_score = {val_f1:.4f}')

f1_score_callback = F1ScoreCallback()
print("Dodati LR i ES i custom f1 score")

model.compile(tf.keras.optimizers.Adam(learning_rate=4.027378994399108e-05), #lr po optuna proracunu
              loss='categorical_crossentropy', 
              metrics=['accuracy', 
                       metric.f1_score,
                       AUC(name='roc_auc', multi_label=True),
                    tf.keras.metrics.AUC(curve='PR', name='average_precision')])
                    #metric.weighted_f1_score,
                    #metric.balanced_accuracy

print("Model kompajliran.")

y_train_labels = np.argmax(y_train_one_hot, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights_dict = dict(enumerate(class_weights))

train_steps_per_epoch = math.ceil(len(X_train_graphs) / batch_size)
val_steps_per_epoch = math.ceil(len(X_val_graphs) / batch_size)

print("Start treninga...........................")
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs = 50,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    class_weight = class_weights_dict,
    callbacks=[lr_scheduler, early_stopping, f1_score_callback]
)

print("Zavrsen trening....")
print("Pisemo sumarry modela u fajl......")

with open('new_training/train_results/gin_new/model_summary-gin_moreActiveABCDE.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Zavrseno pisanje u fajl.")

print("Pisemo history treninga..")
history_df = pd.DataFrame(history.history)
history_df.to_csv('new_training/train_results/gin_new/gin_history_moreActiveABCDE.csv', index=False)

print("Cuvanje tezina modela..")
model.save_weights('new_training/train_results/gin_new/gin_model_moreActiveABCDE.h5')

print("Evalucija modela....")
steps_for_test = math.ceil(len(X_test) / batch_size)

test_loss, test_acc, test_f1_score, test_roc_auc, test_avg_precision = model.evaluate(test_loader.load(), batch_size= batch_size, steps =steps_for_test, verbose=1)


print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1Score: {test_f1_score:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")
print(f"Test Average Precision: {test_avg_precision:.4f}")


print("Idemo na classification report....")

y_true_list = []
y_pred_list = []

for step, batch in enumerate(test_loader):
    if step >= steps_for_test:
        break
    inputs, target = batch
    x, a, i = inputs
    y_true_batch = target # Pretvaranje u numpy array ako već nije
    y_pred_batch = model.predict_on_batch((x, a, i))
    y_pred_batch = np.argmax(y_pred_batch, axis=-1)
    
    # Ako je y_true_batch jedan-hot enkodiran, koristi np.argmax za pretvaranje
    if len(y_true_batch.shape) > 1 and y_true_batch.shape[1] > 1:
        y_true_batch = np.argmax(y_true_batch, axis=-1)
    
    y_true_list.extend(y_true_batch)
    y_pred_list.extend(y_pred_batch)


print(len(y_true_list))
print(y_true_list[0])
print(len(y_pred_list))
print(y_pred_list[0])
y_true = y_true_list
y_pred = y_pred_list


print(classification_report(y_true, y_pred))


print("Sve uspesno zavrseno.")



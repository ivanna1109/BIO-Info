import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from spektral.layers import GraphSageConv, GlobalSumPool, GlobalAvgPool, GlobalMaxPool

class GraphSageModel(Model):
    def __init__(self, num_node_features, num_classes, hidden_units, dense_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphSageConv(hidden_units, agregate='mean', activation='relu')
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)

        self.conv2 = GraphSageConv(hidden_units, agregate='mean', activation='relu')
        self.bn2 = BatchNormalization()
        self.conv3 = GraphSageConv(hidden_units, agregate='mean', activation='relu')
        self.bn3 = BatchNormalization()

        self.pool = GlobalAvgPool()
        self.flatten = tf.keras.layers.Flatten() 
        self.dropout2= Dropout(dropout_rate)
        self.dense1 = Dense(dense_units[0], activation='relu')
        self.dense2 = Dense(dense_units[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        #print(len(inputs))
        x, sparse_a, i = inputs
        x = self.conv1([x, sparse_a])
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2([x, sparse_a])
        x = self.bn2(x, training=training)
        x = self.conv3([x, sparse_a]) 
        x = self.bn3(x, training=training)
        x = self.pool([x, i])
        x = self.flatten(x)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output
    

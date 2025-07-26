import tensorflow as tf
from spektral.layers import GATConv, GlobalSumPool
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class GATModel(Model):
    def __init__(self, units, n_heads, dropout_rate, num_classes):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(units, n_heads, dropout_rate=dropout_rate, activation='relu')
        self.gat2 = GATConv(units, n_heads, dropout_rate=dropout_rate, activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.dropout = Dropout(dropout_rate)
        self.pool = GlobalSumPool()
        self.dense1 = Dense(units * 2, activation='relu')
        self.dense2 = Dense(units, activation='relu')
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        print(len(inputs))
        x, a = inputs
        #mask = tf.ones_like(a, dtype=tf.float32)
        x = self.gat1([x, a])
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = self.gat2([x, a])
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        x = self.pool(x)
        x = self.output_layer(x)
        return x
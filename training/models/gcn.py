import tensorflow as tf
from spektral.layers import GCNConv, GlobalMaxPool
from tensorflow.keras.layers import Dense, Dropout

class GCN(tf.keras.Model):
    def __init__(self, num_features, num_labels, hidden_units, dense_units, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(hidden_units, activation='relu')
        self.conv2 = GCNConv(hidden_units, activation='relu')
        self.conv3 = GCNConv(hidden_units, activation='relu')
        self.global_pool = GlobalMaxPool()

        self.dropout2 = Dropout(dropout_rate)
        self.dense1 = Dense(dense_units[0], activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.dense2 = Dense(dense_units[1], activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.output_layer = Dense(num_labels, activation='softmax')

    def call(self, inputs, training=False):
        x, a = inputs 
        x = self.conv1([x, a])
        x = self.dropout2(x, training=training)

        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = tf.reduce_mean(x, axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)

        return output
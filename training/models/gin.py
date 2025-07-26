import tensorflow as tf
from spektral.layers import GINConv, GlobalMaxPool
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten

class GINModel(tf.keras.Model):
    def __init__(self, num_features, hidden_units, dense_units, dropout_rate, num_classes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(hidden_units, activation='relu',
            aggregators='sum'
        )
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)

        self.conv2 = GINConv(hidden_units, activation='relu',
            aggregators='sum'
        )
        self.batch_norm2 = BatchNormalization()
        self.pool = GlobalMaxPool()
        self.flatten = Flatten() 
        self.dropout2 = Dropout(dropout_rate)
        self.dense1 = Dense(dense_units[0], activation='relu')
        self.dense2 = Dense(dense_units[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.output_layer = Dense(num_classes, activation='softmax')


    def call(self, inputs, training=False):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.batch_norm1(x, training=training)  
        x = tf.keras.activations.relu(x) 
        x = self.dropout1(x, training=training)
        x = self.conv2([x, a])
        x = self.batch_norm2(x, training=training) 
        x = tf.keras.activations.relu(x)  
        x = self.pool([x, i])
        x = self.flatten(x)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output
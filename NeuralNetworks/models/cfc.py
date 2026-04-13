import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Layer, RNN, GlobalAveragePooling1D, Reshape
)


class CfCCell(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # main transformation
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="glorot_uniform")
        self.U = self.add_weight(shape=(self.units, self.units),
                                 initializer="orthogonal")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")

        # gating (continuous-time inspired)
        self.W_tau = self.add_weight(shape=(input_dim, self.units),
                                    initializer="glorot_uniform")
        self.b_tau = self.add_weight(shape=(self.units,),
                                    initializer="zeros")

    def call(self, inputs, states):
        prev_h = states[0]

        # compute "time constant"
        tau = tf.sigmoid(tf.matmul(inputs, self.W_tau) + self.b_tau)

        # candidate state
        h_tilde = tf.tanh(
            tf.matmul(inputs, self.W) +
            tf.matmul(prev_h, self.U) +
            self.b
        )

        # closed-form inspired update
        new_h = (1 - tau) * prev_h + tau * h_tilde

        return new_h, [new_h]


def CfC(nb_classes, Chans, Samples, units=64):

    # reshape: (channels, time, 1) → (time, channels)
    input1 = Input(shape=(Chans, Samples, 1))
    x = Reshape((Samples, Chans))(input1)

    # recurrent processing
    x = RNN(CfCCell(units), return_sequences=True)(x)

    # aggregate over time
    x = GlobalAveragePooling1D()(x)

    output = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=input1, outputs=output)
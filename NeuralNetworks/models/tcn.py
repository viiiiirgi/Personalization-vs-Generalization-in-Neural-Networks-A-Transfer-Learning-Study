from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    Dropout, GlobalAveragePooling1D, Dense, Reshape
)


def TCN(nb_classes, Chans, Samples, filters=32, kernel_size=5, dilations=[1,2,4,8], dropout=0.3):

    # reshape (channels, time, 1) → (time, channels)
    input1 = Input(shape=(Chans, Samples, 1))
    x = Reshape((Samples, Chans))(input1)

    for d in dilations:
        x_res = x

        x = Conv1D(filters, kernel_size,
                   padding='causal',
                   dilation_rate=d)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)

        x = Conv1D(filters, kernel_size,
                   padding='causal',
                   dilation_rate=d)(x)
        x = BatchNormalization()(x)

        # residual connection
        if x_res.shape[-1] != filters:
            x_res = Conv1D(filters, 1, padding='same')(x_res)

        x = Activation('relu')(x + x_res)

    x = GlobalAveragePooling1D()(x)

    output = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=input1, outputs=output)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation,
    AveragePooling2D, Dropout, Flatten, Dense
)
from tensorflow.keras.constraints import max_norm


def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=16, F1=8, D=2, F2=None):

    kernLength = max(kernLength, Samples // 8)

    if F2 is None:
        F2 = F1*D
    input1 = Input(shape=(Chans, Samples, 1))

    # Block 1 Temporal and spatial filtering
    block1 = Conv2D(F1, (1, kernLength),
                    padding='same',
                    use_bias=False)(input1) # (1st layer) learns temporal patterns (finds frequencies that are relevant)

    block1 = BatchNormalization()(block1) #normalizes the data

    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.)
    )(block1) # (2nd layer) learns spatial patterns across electrodes. Each temporal filter gets its own spatial filter. (which parts of the brain are communicating)

    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1) # non linear activation function that allows the network to learn complex patterns
    block1 = AveragePooling2D((1, 4))(block1) # downsamples the data, keeping the important features while reducing noise and computational load
    block1 = Dropout(dropoutRate)(block1) # prevents overfitting: randomly disables neurons

    # Block 2 Feature refinement
    block2 = SeparableConv2D(
        F2,
        (1, 16),
        use_bias=False,
        padding='same'
    )(block1) # (3rd layer) convolution that combine (summarize) spatial and temporal features learned in block 1 

    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    #4th layer
    flatten = Flatten()(block2) # flattens the 2d feature maps into a 1d vector
    dense = Dense(nb_classes,
                  kernel_constraint=max_norm(0.25))(flatten) # fully connected layer that maps the extracted features to the number of classes (3 memory conditions)
    softmax = Activation('softmax', dtype='float32')(dense) # converts the final raw scores into probabilities

    return Model(inputs=input1, outputs=softmax)
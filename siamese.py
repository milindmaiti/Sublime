import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def weighted_bce(y_true, y_pred):
    epsilon = 1e-5
    loss = -(weight_pos * y_true * tf.math.log(y_pred + epsilon) +
            weight_neg * (1 - y_true) * tf.math.log(1 - y_pred + epsilon))
    return tf.reduce_mean(loss)

def create_siamese_network(input_shape):
    def base_network(input_shape):
        input = Input(shape=input_shape)
        x = Flatten()(input) # collapses x and y
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_net = base_network(input_shape)
    processed_a = base_net(input_a)
    processed_b = base_net(input_b)
    concatenated = Concatenate()([processed_a, processed_b])
    concatenated = Dense(128, activation="relu")(concatenated)
    output = Dense(1, activation="sigmoid")(concatenated)

    siamese_net = Model(inputs=[input_a, input_b], outputs=output)
    siamese_net.compile(loss=weighted_bce, optimizer="adam", metrics=["accuracy"])

    return siamese_net
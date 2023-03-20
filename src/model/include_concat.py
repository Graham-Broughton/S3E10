import sys

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def NN(x_train, activation=tf.nn.swish):
    inputs = layers.Input(shape=(x_train.shape[-1],))
    x = tf.keras.layers.Dense(20, activation=activation)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    layer_0 = tf.keras.layers.Dense(20,activation=activation)(x)
    layer_0_0 = tf.keras.layers.Dense(19,activation=activation)(layer_0)
    layer_0_0_0 = tf.keras.layers.Dense(18,activation=activation)(layer_0_0)
    layer_0_0_0_0 = tf.keras.layers.Dense(17,activation=activation)(layer_0_0_0)
    layer_0_0_0_0_0 = tf.keras.layers.Dense(16,activation=activation)(layer_0_0_0_0)
    layer_0_0_0_0_0 = tf.keras.layers.Dense(15, activation=activation, name="layer_0_0_0_0_0")(layer_0_0_0_0_0)
    layer_0_0_0_0_0 = layers.BatchNormalization()(layer_0_0_0_0_0)
    layer_0_0_0_0_1 = tf.keras.layers.Dense(15,activation=activation)(layer_0_0_0_0)
    layer_0_0_0_0_1 = tf.keras.layers.Dense(14, activation=activation, name="layer_0_0_0_0_1")(layer_0_0_0_0_1)
    layer_0_0_0_0_1 = layers.BatchNormalization()(layer_0_0_0_0_1)

    layer_0_0_0_1 = tf.keras.layers.Dense(16,activation=activation)(layer_0_0_0)
    layer_0_0_0_1_0 = tf.keras.layers.Dense(15,activation=activation)(layer_0_0_0_1)
    layer_0_0_0_1_0 = tf.keras.layers.Dense(14, activation=activation, name="layer_0_0_0_1_0")(layer_0_0_0_1_0)
    layer_0_0_0_1_0 = layers.BatchNormalization()(layer_0_0_0_1_0)

    layer_0_0_0_1_1 = tf.keras.layers.Dense(14,activation=activation)(layer_0_0_0_1)
    layer_0_0_0_1_1 = tf.keras.layers.Dense(13, activation=activation, name="layer_0_0_0_1_1")(layer_0_0_0_1_1)
    layer_0_0_0_1_1 = layers.BatchNormalization()(layer_0_0_0_1_1)

    layer_0_0_1 = tf.keras.layers.Dense(17,activation=activation)(layer_0_0)
    layer_0_0_1_0 = tf.keras.layers.Dense(16,activation=activation)(layer_0_0_1)
    layer_0_0_1_0_0 = tf.keras.layers.Dense(15,activation=activation)(layer_0_0_1_0)
    layer_0_0_1_0_0 = tf.keras.layers.Dense(14, activation=activation, name="layer_0_0_1_0_0")(layer_0_0_1_0_0)
    layer_0_0_1_0_0 = layers.BatchNormalization()(layer_0_0_1_0_0)

    layer_0_0_1_0_1 = tf.keras.layers.Dense(14,activation=activation)(layer_0_0_1_0)
    layer_0_0_1_0_1 = tf.keras.layers.Dense(13, activation=activation, name="layer_0_0_1_0_1")(layer_0_0_1_0_1)
    layer_0_0_1_0_1 = layers.BatchNormalization()(layer_0_0_1_0_1)

    layer_0_0_1_1 = tf.keras.layers.Dense(15,activation=activation)(layer_0_0_1)
    layer_0_0_1_1_0 = tf.keras.layers.Dense(14,activation=activation)(layer_0_0_1_1)
    layer_0_0_1_1_0 = tf.keras.layers.Dense(13, activation=activation, name="layer_0_0_1_1_0")(layer_0_0_1_1_0)
    layer_0_0_1_1_0 = layers.BatchNormalization()(layer_0_0_1_1_0)

    layer_0_0_1_1_1 = tf.keras.layers.Dense(13,activation=activation)(layer_0_0_1_1)
    layer_0_0_1_1_1 = tf.keras.layers.Dense(12, activation=activation, name="layer_0_0_1_1_1")(layer_0_0_1_1_1)
    layer_0_0_1_1_1 = layers.BatchNormalization()(layer_0_0_1_1_1)

    layer_0_1 = tf.keras.layers.Dense(18,activation=activation)(layer_0)
    layer_0_1_0 = tf.keras.layers.Dense(17,activation=activation)(layer_0_1)
    layer_0_1_0_0 = tf.keras.layers.Dense(16,activation=activation)(layer_0_1_0)
    layer_0_1_0_0_0 = tf.keras.layers.Dense(15,activation=activation)(layer_0_1_0_0)
    layer_0_1_0_0_0 = tf.keras.layers.Dense(14, activation=activation, name="layer_0_1_0_0_0")(layer_0_1_0_0_0)
    layer_0_1_0_0_0 = layers.BatchNormalization()(layer_0_1_0_0_0)

    layer_0_1_0_0_1 = tf.keras.layers.Dense(14,activation=activation)(layer_0_1_0_0)
    layer_0_1_0_0_1 = tf.keras.layers.Dense(13, activation=activation, name="layer_0_1_0_0_1")(layer_0_1_0_0_1)
    layer_0_1_0_0_1 = layers.BatchNormalization()(layer_0_1_0_0_1)

    layer_0_1_0_1 = tf.keras.layers.Dense(15,activation=activation)(layer_0_1_0)
    layer_0_1_0_1_0 = tf.keras.layers.Dense(14,activation=activation)(layer_0_1_0_1)
    layer_0_1_0_1_0 = tf.keras.layers.Dense(13, activation=activation, name="layer_0_1_0_1_0")(layer_0_1_0_1_0)
    layer_0_1_0_1_0 = layers.BatchNormalization()(layer_0_1_0_1_0)

    layer_0_1_0_1_1 = tf.keras.layers.Dense(13,activation=activation)(layer_0_1_0_1)
    layer_0_1_0_1_1 = tf.keras.layers.Dense(12, activation=activation, name="layer_0_1_0_1_1")(layer_0_1_0_1_1)
    layer_0_1_0_1_1 = layers.BatchNormalization()(layer_0_1_0_1_1)

    layer_0_1_1 = tf.keras.layers.Dense(16,activation=activation)(layer_0_1)
    layer_0_1_1_0 = tf.keras.layers.Dense(15,activation=activation)(layer_0_1_1)
    layer_0_1_1_0_0 = tf.keras.layers.Dense(14,activation=activation)(layer_0_1_1_0)
    layer_0_1_1_0_0 = tf.keras.layers.Dense(13, activation=activation, name="layer_0_1_1_0_0")(layer_0_1_1_0_0)
    layer_0_1_1_0_0 = layers.BatchNormalization()(layer_0_1_1_0_0)

    layer_0_1_1_0_1 = tf.keras.layers.Dense(13,activation=activation)(layer_0_1_1_0)
    layer_0_1_1_0_1 = tf.keras.layers.Dense(12, activation=activation, name="layer_0_1_1_0_1")(layer_0_1_1_0_1)
    layer_0_1_1_0_1 = layers.BatchNormalization()(layer_0_1_1_0_1)

    layer_0_1_1_1 = tf.keras.layers.Dense(14,activation=activation)(layer_0_1_1)
    layer_0_1_1_1_0 = tf.keras.layers.Dense(13,activation=activation)(layer_0_1_1_1)
    layer_0_1_1_1_0 = tf.keras.layers.Dense(12, activation=activation, name="layer_0_1_1_1_0")(layer_0_1_1_1_0)
    layer_0_1_1_1_0 = layers.BatchNormalization()(layer_0_1_1_1_0)

    layer_0_1_1_1_1 = tf.keras.layers.Dense(12,activation=activation)(layer_0_1_1_1)
    layer_0_1_1_1_1 = tf.keras.layers.Dense(11, activation=activation, name="layer_0_1_1_1_1")(layer_0_1_1_1_1)
    layer_0_1_1_1_1 = layers.BatchNormalization()(layer_0_1_1_1_1)

    concat = tf.keras.layers.concatenate([
        layer_0_0_0_0_0, layer_0_0_0_0_1, layer_0_0_0_1_0, layer_0_0_0_1_1, layer_0_0_1_0_0,
        layer_0_0_1_0_1, layer_0_0_1_1_0, layer_0_0_1_1_1, layer_0_1_0_0_0, layer_0_1_0_0_1, layer_0_1_0_1_0,
        layer_0_1_0_1_1, layer_0_1_1_0_0, layer_0_1_1_0_1, layer_0_1_1_1_0, layer_0_1_1_1_1]
    )
    output0 = tf.keras.layers.Dense(20, activation=activation)(concat)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output0)
    ensemble = tf.keras.models.Model(inputs, output)
    plot_model(ensemble, to_file="model_concat.png", show_shapes=True)
    return ensemble

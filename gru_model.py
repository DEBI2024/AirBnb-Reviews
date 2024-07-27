"""This module constructs and runs the model which is based on GRU architecture
The model uses GLOVE pre-trained embeddings

Author: AbdElRhman ElMoghazy
Date 27-07-2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, GRU


def get_embedding(embedding_matrix, num_tokens):
    """build and initialize the embedding layer with the pretrained weights"""
    embeddins = keras.layers.Embedding(
        num_tokens,
        100,
        trainable=False,
    )

    embeddins.build((1,))
    embeddins.set_weights([embedding_matrix])

    return embeddins


def build_model(embedding_matrix, num_tokens):
    """define the model using keras functional API
    Args:
      embedding_matrix: matrix with glove pre-trained embeddings
      num_tokens: number of tokens for the embedding layer

    Returns:
      model: the functional API GRU based model
    """
    embedding_layer = get_embedding(embedding_matrix, num_tokens)
    input_text = keras.Input(shape=(None,), dtype="int32")
    embedded = embedding_layer(input_text)

    rnn = GRU(200, return_sequences=True)(embedded)
    dense_rnn = Dense(600)(rnn)

    polarity_layer = Dense(
        1,
        activation='sigmoid',
        name='polarity_class')(dense_rnn)

    model = tf.keras.Model(
        inputs=input_text,
        outputs=polarity_layer,
    )

    return model


def main(x_train, y_train, embedding_matrix, num_tokens):
    """run the model pipeline"""

    model = build_model(embedding_matrix, num_tokens)

    model.compile(
        optimizer=Adam(),
        loss=['binary_crossentropy'],
        metrics=['accuracy'],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=2000,
        batch_size=128,
        validation_split=0.2,
        # using early stopping to stop the model if validation loss didn't
        # improve for 5 epochs
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='min')
        ],
        verbose=1
    )

    return history, model

"""This module constructs and runs the model which is based on GRU architecture
The model uses GLOVE pre-trained embeddings

Author: AbdElRhman ElMoghazy
Date 27-07-2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Input,
    TimeDistributed,
    Activation,
    Bidirectional,
    SimpleRNN,
    GRU,
    LSTM)
import tensorflow.data as tf_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

    rnn = GRU(200, return_sequences=False)(embedded)
    dense_rnn = Dense(600)(rnn)

    polarity_layer = Dense(1, activation='sigmoid')(dense_rnn)

    model = tf.keras.Model(
        inputs=input_text,
        outputs=polarity_layer,
    )

    return model


def show_learning_curves(history):

    f = plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 2, 1)

    # plot accuracy and validation accuracy
    plt.plot(history.history.get("accuracy"), label=f"train_accuracy")
    plt.plot(history.history.get("val_accuracy"), label=f"validation_accuracy")

    plt.title('accuracy curves plot'), plt.ylabel(
        'accuracy'), plt.xlabel('epoch')
    plt.legend()

    ax2 = plt.subplot(1, 2, 2, sharey=ax1)

    # plot loss and validation loss
    plt.plot(history.history.get("loss"), label=f"train_loss")
    plt.plot(history.history.get("val_loss"), label=f"validation_loss")

    plt.title('loss curves plot'), plt.ylabel('loss'), plt.xlabel('epoch')
    plt.legend()


def plot_confusion(pred, y):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


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

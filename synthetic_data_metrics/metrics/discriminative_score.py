"""Implementation of Discriminative Score for Multivariate Time-series data

A post-hoc binary LSTM classifier is trained to distinguish between real and
synthetic data examples. Here, real data instances are labelled ‘real’
and generated data instances are labelled ‘fake’, and the classifier is
trained in a fully supervised way.

If the comapred datasets include a target(label) column, it must be numeriacl.

The score is (0.5 - classification_accuracy). If the synthetic data is similar
to real data, the score would be close to zero. The closer the score to 0.5
means the synthetic data is dissimilar from real data.

Returns:
    disc_scores: Score between 0 and 0.5 (lower means good quality synth data)
"""
import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
tf.get_logger().setLevel('ERROR')


def calculate_ds(X_train, y_train, X_test, y_test, epochs,
                 verbose):

    seq_length, input_dim = X_train.shape[1], X_train.shape[2]
    model = Sequential()
    model.add(LSTM(32, input_shape=(seq_length, input_dim),
                   activation='tanh', return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(32))
    # model.add(Dropout(0.2))
    # model.add(Dense(16,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.fit(x=X_train,
              y=y_train,
              epochs=epochs,
              verbose=verbose,
              batch_size=64)
    evaluated = model.evaluate(X_test, y_test, verbose=0)
    score = evaluated[1]
    return np.abs(0.5 - score)

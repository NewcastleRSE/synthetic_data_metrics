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
from scipy import stats
import pandas as pd
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
tf.get_logger().setLevel('ERROR')


# prepare train/test in windows of fixed size.
def prep_data_updated(X, y):
    data = []
    labels = []
    for i in range(0, X.shape[0] - 10, 1):
        _data = X.values[i: i + 10]
        _y = stats.mode(y[i: i + 10])[0][0]
        data.append(_data)
        labels.append(_y)
    return data, labels


def calculate_ds(real, synth, window_size, step, epochs,
                 verbose, plot_loss):
    # real = real.copy()
    # synth = synth.copy()
    if len(real) > len(synth):
        real = real[:len(synth)]
    else:
        synth = synth[:len(real)]
    real['label'] = 1
    synth['label'] = 0
    data = pd.concat([real, synth], axis=0)
    X, y = prep_data_updated(data.drop('label', axis=1), data.label)
    # shuffle the two lists
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    X = np.asarray(X, dtype=np.float32).reshape(-1, 10, 13)
    y = np.asarray(y)
    # split into training/testing
    limit = int(0.8*len(X))
    X_train, y_train = X[:limit], y[:limit]
    X_test, y_test = X[limit:], y[limit:]
    model = Sequential()
    model.add(LSTM(32, input_shape=(10, 13),
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
              epochs=20,
              verbose=False,
              batch_size=64)
    evaluated = model.evaluate(X_test, y_test, verbose=0)
    score = evaluated[1]
    return np.abs(0.5 - score)


def is_categorical(col):
    return col.dtype.name == 'object'


def dis_score(real, synth, target: str = None, window_size=10,
              step=1, epochs=50, verbose=False, plot_loss=False):
    """The main function

    Args:
        real (DataFrame): The real dataset.
        synth (DataFrame): The synthetic dataset.
        target (str, optional): The name of the target column.
                                Defaults to None.
        window_size (int, optional): Used to divide training/testing
                                    data into windows. Defaults to 10.
        step (int, optional): How much overlap between windows.
                              Defaults to 1.
        epochs (int, optional): Number of model training iterations.
                                Defaults to 50.
        verbose (bool, optional): To display model training progress.
                                Defaults to False.
        plot_loss (bool, optional): To plot loss/accuracy of model.
                                    Defaults to False.

    Returns:
        scores: A list containing the discriminative scores.
    """
    print("Calculating the discrimiative score of real and synthetic data")
    real = real.copy()
    synth = synth.copy()
    # convert categorical columns to numerical
    for col in real.columns:
        if is_categorical(real[col]):
            real[col] = pd.factorize(real[col])[0]
        if is_categorical(synth[col]):
            synth[col] = pd.factorize(synth[col])[0]
    # remove the time channel
    for col in ['time', 'date']:
        if col in real.columns:
            real.drop(col, axis=1, inplace=True)
        if col in synth.columns:
            synth.drop(col, axis=1, inplace=True)
    disc_scores = []
    # check if the datasets include a target column,
    # divide the datasets by label and calculate a score for each label
    if target is not None:
        # retrive all the unique labels
        labels = real[target].unique()
        # slice the dataset into subsets by label
        # and calculate a score for each subset separetely
        for label in labels:
            chosen = [label]
            real_temp = real.loc[real[target].isin(chosen)].copy()
            synth_temp = synth.loc[synth[target].isin(chosen)].copy()
            real_temp.drop(target, axis=1, inplace=True)
            synth_temp.drop(target, axis=1, inplace=True)
            ds_temp = calculate_ds(real_temp, synth_temp, window_size,
                                   step, epochs, verbose, plot_loss)
            disc_scores.append(ds_temp)
    # else if the dataset has no target column pass it as a whole
    else:
        result = calculate_ds(real, synth, window_size, step, epochs,
                              verbose, plot_loss)
        disc_scores.append(result)
    return sum(disc_scores)/len(disc_scores)

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
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None
tf.get_logger().setLevel('ERROR')


# prepare train/test in windows of fixed size.
def prep_data(X, y, window_size, step):
    data = []
    labels = []
    for i in range(0, X.shape[0] - window_size, step):
        _data = X.values[i: i + window_size]
        _y = stats.mode(y[i: i + window_size])[0][0]
        data.append(_data)
        labels.append(_y)
    return data, labels


def calculate_ds(real, synth, window_size, step, epochs,
                 verbose, plot_loss):
    num_features = real.shape[1]
    if len(real) > len(synth):
        real = real[:len(synth)]
    else:
        synth = synth[:len(real)]
    # add label column to indicate the source of the data point.
    real['label'] = 1
    synth['label'] = 0
    # merge real and synth data together
    data = pd.concat([real, synth], axis=0)
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    data_X, data_y = data.drop('label', axis=1), data.label
    # split into train/test 0.6/0.4
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.4,  shuffle=False, random_state=42)
    # prepare the training/testing data into batches of window_size
    X_train, y_train = prep_data(X_train, y_train, window_size, step)
    X_train = np.asarray(X_train, dtype=np.float32).reshape(
        -1, window_size, num_features)
    y_train = np.asarray(y_train)
    X_test, y_test = prep_data(X_test, y_test, window_size, step)
    X_test = np.asarray(X_test, dtype=np.float32).reshape(
        -1, window_size, num_features)
    y_test = np.asarray(y_test)
    # binary LSTM classifier architecture
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, num_features),
                   activation='tanh', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(X_test, y_test),
                        batch_size=128)
    if plot_loss:
        val_acc = history.history['val_accuracy']
        acc = history.history['accuracy']
        val_loss = history.history['val_loss']
        loss = history.history['loss']
        # fig = plt.figure(figsize=(10, 10))
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.plot(val_acc, label='val_acc')
        plt.plot(val_loss, label='val_loss')
        plt.plot(acc, label='train_acc')
        plt.plot(loss, label='train_loss')
        plt.legend()
        plt.show()
    evaluated = model.evaluate(X_test, y_test, verbose=verbose)
    # print('Model metrics: ', model.metrics_names)
    # print('Model evaluation: ', evaluated)
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
    return disc_scores

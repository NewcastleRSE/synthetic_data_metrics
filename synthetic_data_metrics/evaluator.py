from synthetic_data_metrics.utils import (get_inception_features,
                                          is_categorical, prep_data_updated)
from synthetic_data_metrics.metrics.image_metrics import inception_score
from synthetic_data_metrics.metrics.discriminative_score import calculate_ds
import pandas as pd
import random
import numpy as np


class Image_Evaluator:
    """
    The central class Image_Evaluator, used to evaluate synthetic image data.

        Parameters:
            synth (np.ndarray): 4D array containing data with `uint8` type.
            real (np.ndarray, optional): 4D array containing
                data with `uint8` type.

        Methods:
            metrics():
                Provides all image evaluation metrics implemented.
            inception_score(n_splits):
                Runs the inception score evaluation metric.
    """

    def __init__(self, synth, real=None):
        """
        Constructs all the necessary attributes to create an Image_Evaluator.

            Parameters:
                synth (np.ndarray): 4D array containing data with `uint8` type.
                real (np.ndarray, optional): 4D array containing
                    data with `uint8` type.

            Returns:
                None
        """
        self.synth_data = synth
        self.real_data = real
        self._metrics = ['inception_score']
        self.inception_softmax_scores = None

    def metrics(self):
        """
        Provides all image evaluation metrics implemented.

            Paremeters:
                None

            Returns:
                self._metrics (list): All image evaluation metrics
                    provided by Image_Evaluator.
        """
        return self._metrics

    def inception_score(self, n_splits):
        """
        Runs the inception score evaluation metric.
        Code implementation based on:
        https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/

            Paremeters:
                n_splits (int): The number of splits to divide
                    the image data into when processing.

            Returns:
                mean (float): The mean of the inception score
                    for each image split.
                std (float): The standard deviation of the inception score
                    for each image split.
        """
        if self.inception_softmax_scores is None:
            self.inception_softmax_scores = get_inception_features(
                                                self.synth_data, n_splits)

            # then run metric.
            mean, std = inception_score(self.inception_softmax_scores)
            return mean, std

        else:
            # should run some checks here to ensure data looks correct.
            mean, std = inception_score(self.inception_softmax_scores)
            return mean, std


class TS_Evaluator:
    """
    The central class TS_Evaluator,
    used to evaluate synthetic time-series data.

        Parameters:
            real (np.ndarray): 4D array containing data with `uint8` type.
            synth (np.ndarray): 4D array containing data with `uint8` type.
            target (String, optional): The name of the data's target column.
            window_size (int, optional): Determines the size
                of the moving window.
            step (int, optional): The sliding window overlap.
            epochs (int, optional): The number of epochs to train the model.
            verbose (String/int, optional): The verbositiy of the
                model's training process.

        Methods:
            metrics():
                Provides all image evaluation metrics implemented.
            discriminative_score():
                Runs the discriminative score evaluation metric.
    """

    def __init__(self, real, synth, target=None,
                 window_size=10,
                 step=1, epochs=20, verbose=False):
        """
        Constructs all the necessary attributes to create a TS_Evaluator.

            Parameters:
                real (np.ndarray): 4D array containing data with `uint8` type.
                synth (np.ndarray): 4D array containing data with `uint8` type.
                target (String, optional): The name of the
                    data's target column.
                window_size (int, optional): Determines the size
                    of the moving window.
                step (int, optional): The sliding window overlap.
                epochs (int, optional): The number of epochs
                    to train the model.
                verbose (String/int, optional): The verbositiy
                    of the model's training process.

            Returns:
                None
        """
        self.synth_data = synth
        self.real_data = real
        self.target = target
        self.window_size = window_size
        self.step = step
        self.epoch = epochs
        self.verbose = verbose
        self._metrics = ['discriminator_score', 't-SNE']

    def metrics(self):
        """
        Provides all time-series evaluation metrics implemented.

            Paremeters:
                None

            Returns:
                self._metrics (list): All time-series evaluation
                    metrics provided by TS_Evaluator.
        """
        return self._metrics

    def discriminative_score(self):
        """
        Runs the discriminative score evaluation metric.
        Code implementation based on:
        https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/discriminative_metrics.py

            Paremeters:
                None

            Returns:
                score (float): The discriminative score
                    for the real and synthetic data.
        """
        print("Calculating the discrimiative score of real and synthetic data")

        # convert categorical columns to numerical
        for col in self.real_data.columns:
            if is_categorical(self.real_data[col]):
                self.real_data[col] = pd.factorize(self.real_data[col])[0]
            if is_categorical(self.synth_data[col]):
                self.synth_data[col] = pd.factorize(self.synth_data[col])[0]
        # Naively remove the time channel if it exists
        for col in ['time', 'Time', 'Date', 'date']:
            if col in self.real_data.columns:
                self.real.drop(col, axis=1, inplace=True)
            if col in self.synth.columns:
                self.synth.drop(col, axis=1, inplace=True)
        disc_scores = []
        # check if the datasets include a target column,
        # divide the datasets by label and calculate a score for each label
        if self.target is not None:
            # retrive all the unique labels
            labels = self.real[self.target].unique()
            # slice the dataset into subsets by label
            # and calculate a score for each subset separetely
            for label in labels:
                chosen = [label]
                real_temp = self.real.loc[self.real[self.target].isin(chosen)].copy() # noqa
                synth_temp = self.synth.loc[self.synth[self.target].isin(chosen)].copy() # noqa
                real_temp.drop(self.target, axis=1, inplace=True)
                synth_temp.drop(self.target, axis=1, inplace=True)
                if len(real_temp) > len(synth_temp):
                    real_temp = real_temp[:len(synth_temp)]
                else:
                    synth_temp = synth_temp[:len(real_temp)]
                real_temp['label'] = 1
                synth_temp['label'] = 0
                data = pd.concat([real_temp, synth_temp], axis=0)
                X, y = prep_data_updated(data.drop('label', axis=1),
                                         data.label,
                                         window_size=self.window_size,
                                         step=self.step)
                # shuffle the two lists
                c = list(zip(X, y))
                random.shuffle(c)
                X, y = zip(*c)
                X = np.asarray(X, dtype=np.float32)
                y = np.asarray(y)
                # split into training/testing
                limit = int(0.8*len(X))
                X_train, y_train = X[:limit], y[:limit]
                X_test, y_test = X[limit:], y[limit:]
                ds_temp = calculate_ds(X_train, y_train, X_test, y_test,
                                       self.epochs, self.verbose)
                disc_scores.append(ds_temp)
        # else if the dataset has no target column pass it as a whole
        else:
            result = calculate_ds(self.real, self.synth, self.epochs,
                                  self.verbose)
            disc_scores.append(result)
        score = sum(disc_scores)/len(disc_scores)
        return score

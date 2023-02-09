from synthetic_data_metrics.utils import (get_inception_features,
                                          is_categorical, prep_data_updated)
from synthetic_data_metrics.metrics.image_metrics import inception_score
from synthetic_data_metrics.metrics.discriminative_score import calculate_ds
import pandas as pd
import random
import numpy as np


class Image_Evaluator:
    """The central class image evaluator, used to hold and evaluate data
    via metrics and visualisation.
    Parameters
    ----------
    synth : a synthetic dataset of images.
        numpy array: shape [N_samples, H, w, C]
    real : an optional real dataset of images to compare against.
        numpy array: shape [N_samples, H, w, C]
    Returns
    -------
    Evaluator
        An `Evaluator` object ready for a metric to be called.
    """

    def __init__(self, synth, real=None):

        # Metrics is private to apply some validation
        self.synth_data = synth
        self.real_data = real

        self._metrics = ['inception_score']

        # initialise the feature sets as None
        self.inception_softmax_scores = None

    def metrics(self):
        """funtion to return all image metrics implemented"""

        return self._metrics

    def inception_score(self, n_splits):

        # call the inception score metric function - need to add documentation

        if self.inception_softmax_scores is None:
            self.inception_softmax_scores = get_inception_features(
                                                self.synth_data, n_splits)

            # then run metric.
            return inception_score(self.inception_softmax_scores)

        else:
            # should run some checks here to ensure data looks correct.
            return inception_score(self.inception_softmax_scores)


class TS_Evaluator:
    '''The central time series evaluator, used to hold and evaluate data
    via metrics and visualisation.
    Parameters
    ----------
    synth : a dataframe of synthetic time series data.
    real : a dataframe of real time series data to compare against.
    Returns
    -------
    Evaluator
        An `Evaluator` object ready for a metric to be called.
    '''

    def __init__(self, real, synth, target=None,
                 window_size=10,
                 step=1, epochs=20, verbose=False):

        # Metrics is private to apply some validation
        self.synth_data = synth
        self.real_data = real
        self.target = target
        self.window_size = window_size
        self.step = step
        self.epoch = epochs
        self.verbose = verbose
        self._metrics = ['discriminator_score', 't-SNE']

    def metrics(self):
        """funtion to return all image metrics implemented"""

        return self._metrics

    def discriminative_score(self):
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
                self.real_data.drop(col, axis=1, inplace=True)
            if col in self.synth_data.columns:
                self.synth_data.drop(col, axis=1, inplace=True)
        disc_scores = []
        # check if the datasets include a target column,
        # divide the datasets by label and calculate a score for each label
        if self.target is not None:
            # retrive all the unique labels
            labels = self.real_data[self.target].unique()
            # slice the dataset into subsets by label
            # and calculate a score for each subset separetely
            for label in labels:
                chosen = [label]
                real_temp = self.real_data.loc[self.real_data[self.target].isin(chosen)].copy() # noqa
                synth_temp = self.synth_data.loc[self.synth_data[self.target].isin(chosen)].copy() # noqa
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
                                       self.epoch, self.verbose)
                disc_scores.append(ds_temp)
        # else if the dataset has no target column pass it as a whole
        else:
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
            result = calculate_ds(X_train, y_train, X_test, y_test, self.epoch,
                                  self.verbose)
            disc_scores.append(result)
        return sum(disc_scores)/len(disc_scores)

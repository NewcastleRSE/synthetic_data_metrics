from typing import List
from skimage.transform import resize
from numpy import asarray
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from math import floor
from sklearn.manifold import TSNE
from numpy.random import shuffle
import random
from scipy import stats
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def scale_images(images, new_shape) -> np.ndarray:
    """
    Returns an array of images scaled to a new shape.

        Parameters:
                images (np.ndarray): A 4d ndarray of `uint8`.
                new_shape (Tuple[int, int, int]): New size specifications of
                    images.
        Returns:
            images_array (np.ndarray): Ndarray of uint8 resizes to match the
                size of new_shape.
    """
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
        images_array = asarray(images_list)
    return images_array


def get_inception_softmax_score(images, n_splits=10) -> List[np.ndarray]:
    """
        Returns a list of np.ndarrays containing the class predictions
        for each image in the passed image subset.

        Parameters:
                images (np.ndarray): A 4d ndarray of `uint8`.
                n_splits (int): Number of partitions the data is split into.
        Returns:
                softmax_scores (List[np.ndarray]): List of ndarrays containing
                    model predictions.
    """
    shuffle(images)
    model = InceptionV3()
    softmax_scores = list()
    n_part = floor(images.shape[0] / n_splits)
    for i in range(n_splits):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype("float32")
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        softmax_scores.append(p_yx)
    return softmax_scores


def window_time_series(x, y, window_size, step) -> (List[pd.array], List[int]):
    """
        Returns two lists, one of a time series broken into windows, and
            another of the labels for each of those windows.

        Parameters:
                x (pd.Dataframe): Dataframe of predictor variables.
                y (pd.Series): Series of response variables.
                window_size (int): Size of each window.
                step (int): Number of windows x is broken into.
        Returns:
                data (List[pd.array]): x converted into a step length List of
                    pd.array of size window_size.
                labels (List[int]): List of int labels for each array in data.
    """
    data = []
    labels = []
    for i in range(0, x.shape[0] - window_size, step):
        _data = x.values[i: i + window_size]
        _y = stats.mode(y[i: i + window_size])[0][0]
        data.append(_data)
        labels.append(_y)
    return data, labels


def is_categorical(col) -> bool:
    """
        Checks if the input column is categorical.

        Parameters:
                col (pd.Series): Series of input variables.
        Returns:
                True if column is categorical, False otherwise.
    """
    return col.dtype.name == 'object'


def clean_time_series(real, synth,
                      target=None) -> (pd.DataFrame, pd.DataFrame):
    """
        Returns two Dataframes after dropping unncessary columns
        and balancing the two datasets.

        Parameters:
                real (pd.DataFrame): Dataframe of real time series.
                synth (pd.DataFrame): Dataframe of synthetic time series.
                target (string, optional): Name of target column.
        Returns:
                real (pd.DataFrame): Dataframe of real time series.
                synth (pd.DataFrame): Dataframe of synthetic time series.
    """
    # convert categorical columns to numerical
    for col in real.columns:
        if is_categorical(real[col]):
            real[col] = pd.factorize(real[col])[0]
        if is_categorical(synth[col]):
            synth[col] = pd.factorize(synth[col])[0]
    # Naively remove the time channel if it exists
    for col in ['time', 'Time', 'Date', 'date']:
        if col in real.columns:
            real.drop(col, axis=1, inplace=True)
        if col in synth.columns:
            synth.drop(col, axis=1, inplace=True)
    # If target is specified, balance the data
    if target:
        balanced_real = pd.DataFrame()
        balanced_synth = pd.DataFrame()
        unique_labels = real[target].unique()
        for label in unique_labels:
            real_temp = real[real[target] == label]
            synth_temp = synth[synth[target] == label]
            if len(real_temp) > len(synth_temp):
                real_temp = real_temp[:len(synth_temp)]
            else:
                synth_temp = synth_temp[:len(real_temp)]
            balanced_real = pd.concat([balanced_real, real_temp],
                                      ignore_index=True)
            balanced_synth = pd.concat([balanced_synth, synth_temp],
                                       ignore_index=True)
        real, synth = balanced_real, balanced_synth
    else:
        if len(real) > len(synth):
            real = real[:len(synth)]
        else:
            synth = synth[:len(real)]
    return real, synth


def get_train_test_split(real, synth, target, window_size, step, label=None):
    """
        Return train and test splits for a time series.

        Args:
            real (pd.DataFrame): Dataframe of real time series.
            synth (pd.DataFrame): Dataframe of synthetic time series.
            target (string): Name of target column.
            window_size (int): Determines the size
                        of the moving window.
            step (int): The sliding window overlap.
            label (int, optional): The label used to return a specific
                                subset. Defaults to None.

        Returns:
            Four pd.arrays: Corrseponding to the train and test splits
    """
    if label:
        chosen = [label]
        real = real.loc[real[target].isin(chosen)].copy() # noqa
        synth = synth.loc[synth[target].isin(chosen)].copy() # noqa
    real = real.drop(target, axis=1)
    synth = synth.drop(target, axis=1)
    real['label'] = 1
    synth['label'] = 0
    data = pd.concat([real, synth], axis=0)
    X, y = window_time_series(data.drop('label', axis=1),
                              data.label,
                              window_size=window_size,
                              step=step)
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
    return X_train, y_train, X_test, y_test


def calculate_tsne(data, perplexity=30) -> pd.DataFrame:
    """
        Returns the results of running the t-SNE algorithm on input data.

        Parameters:
                data (List[pd.Dataframe]): List of two Dataframes containing
                    one real time series and one synthetic time series.
                perplexity (int): Perplexity of manifold learning algorithm,
                    number of nearest neighbors used.
        Returns:
                tsne_results (pd.Dataframe): Dataframe containing the results
                    of the t-SNE algorithm.
    """
    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=500, random_state=123)
    all_data = np.concatenate(data)
    tsne_results = pd.DataFrame(tsne.fit_transform(all_data))
    return tsne_results


def calculate_pca():
    return None

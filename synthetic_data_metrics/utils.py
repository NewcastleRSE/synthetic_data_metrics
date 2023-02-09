from typing import List
from skimage.transform import resize
from numpy import asarray
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from math import floor
from sklearn.manifold import TSNE
from numpy.random import shuffle
from scipy import stats
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def scale_images(images, new_shape) -> np.ndarray:
    """
    Returns an array of images scaled to a new shape.

        Parameters:
                images (np.ndarray): A 4d ndarray of uint8
                new_shape (Tuple[int, int, int]): New size specifications of
                    images
        Returns:
            images_array (np.ndarray): Ndarray of uint8 resizes to match the
                size of new_shape
    """
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
        images_array = asarray(images_list)
    return images_array


def get_inception_softmax_score(images, n_splits=10) -> List[np.ndarray]:
    """
        Returns a list of ndarrays containing the predictions of a model
            trained on a shuffled image set.

        Parameters:
                images (np.ndarray): A 4d ndarray of uint8
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


def prep_data_updated(x, y, window_size, step) -> (List[pd.array], List[int]):
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
                    pd.array ofsize window_size.
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

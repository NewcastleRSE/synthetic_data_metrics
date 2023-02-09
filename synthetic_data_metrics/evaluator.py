from synthetic_data_metrics.utils import (get_inception_features,
                                          is_categorical, prep_data_updated,
                                          calculate_tsne)
from synthetic_data_metrics.metrics.image_metrics import inception_score
from synthetic_data_metrics.metrics.discriminative_score import calculate_ds
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import matplotlib.colors as mcolors
import math
import uuid


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
        self._metrics = ['discriminator_score', 't_SNE']

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
        score = sum(disc_scores)/len(disc_scores)
        return score

    def t_SNE(self, sample_size=500, perplexity=40,
              save_plot: bool = False, tag=''):

        """A function to plot tSNE 2d embeddings of multiple
        generated datasets along with the original dataset.

        Args:
            real (_type_): The real dataset.
            synth: The synthetic datasets.
            target (str): name of the target (label) column. Defaults to None.
            sample_size (int, optional): take only s subset of this length from
                                        each label. Defaults to 500.
            perplexity (int, optional): internal t-SNE hyperparameter.
                                        Defaults to 500.
            save_plot (bool, optional): whether to save the plot.
                                        Defaults to False.
            tag (str, optional): to be added to the name of
                                the saved plot. Defaults to ''.
        """

        print('Plotting t-SNE of the real and synthetic data')
        # creat a folder to save plots
        real = self.real_data.copy()
        synth = self.synth_data.copy()
        if save_plot:
            Path("plots").mkdir(parents=True, exist_ok=True)
        # Naively remove the time channel if it exists
        for col in ['time', 'Time', 'Date', 'date']:
            if col in real.columns:
                real.drop(col, axis=1, inplace=True)
            if col in synth.columns:
                synth.drop(col, axis=1, inplace=True)
        for col in real.columns:
            if is_categorical(real[col]):
                real[col] = pd.factorize(real[col])[0]
                synth[col] = pd.factorize(synth[col])[0]
        # get number of unique labels
        num_labels = len(real[self.target].unique())
        if num_labels > 1:
            print(f'Splitting datasets by label column ({self.target})...')
            labels = real[self.target].unique()
            colors = list(mcolors.BASE_COLORS)
            ncols = 2
            nrows = math.ceil(len(labels)/ncols)
            fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15),
                                    constrained_layout=True)
            for label, ax in enumerate(fig.axes):
                if label < len(labels):
                    # print(f'Plotting label {label}')
                    chosen = [label]
                    all_data = []
                    real_temp = real.loc[real[self.target].isin(chosen)].copy()
                    assert sample_size <= len(real_temp), "sample size"
                    "should be smaller"
                    "than the number of samples for each label"
                    real_temp = real_temp[:sample_size]
                    all_data.append(real_temp)
                    synth_temp = synth.loc[synth[self.target].isin(chosen)].copy() # noqa
                    synth_temp = synth_temp[:sample_size]
                    synth_temp.dropna(inplace=True)
                    all_data.append(synth_temp)
                    for dataset in all_data:
                        dataset.drop([self.target], axis=1, inplace=True)
                    tsne_results = calculate_tsne(all_data,
                                                  perplexity=perplexity)
                    ax.set_title(f'label {label}', fontsize=10,
                                 color='black', pad=10)
                    labels_for_legend = []
                    for n, i in enumerate(np.arange(start=0,
                                                    stop=len(tsne_results),
                                                    step=sample_size)):
                        if n == 0:
                            ax_label = 'REAL'
                        else:
                            ax_label = 'SYNTH'
                        labels_for_legend.append(ax_label)
                        ax.scatter(tsne_results.iloc[i:i+sample_size, 0].values, # noqa
                                   tsne_results.iloc[i:i+sample_size, 1].values, # noqa
                                   c=colors[n], alpha=0.2, label=ax_label)
                    fig_dim = 45
                    plt.xlim(-fig_dim, fig_dim)
                    plt.ylim(-fig_dim, fig_dim)
                    fig.suptitle(f'sample_size={sample_size}'
                                 f' perplexity={perplexity}',
                                 fontsize=15, color='black')
                    # ax.axis('scaled')
                    # ax.legend()
                    leg = ax.legend()
                    for lh in leg.legendHandles:
                        lh.set_alpha(1)
            if len(labels) % 2 != 0:
                axs.flat[-1].set_visible(False)  # to remove last plot
            # plt.legend()
            if save_plot:
                plt.savefig(f'plots/{tag} 2D_tSNE_{uuid.uuid4()}')
            # plt.show()
            return plt
        else:
            real.drop('ACTIVITY', axis=1, inplace=True)
            synth.drop('ACTIVITY', axis=1, inplace=True)
            assert sample_size <= len(real), "sample size should be smaller than the size of the dataset" # noqa
            print(f'Plotting {sample_size} samples from each dataset...')
            colors = list(mcolors.BASE_COLORS)
            fig = plt.figure(constrained_layout=True, figsize=(10, 7))
            all_data = []
            real_temp = real[:sample_size]
            all_data.append(real_temp)
            synth_temp = synth[:sample_size]
            all_data.append(synth_temp)
            tsne_results = calculate_tsne(all_data,  perplexity=perplexity)
            for n, i in enumerate(np.arange(start=0, stop=len(tsne_results),
                                            step=sample_size)):
                label = 'real' if n == 0 else 'synth'
                plt.scatter(tsne_results.iloc[i:i+sample_size, 0].values,
                            tsne_results.iloc[i:i+sample_size, 1].values,
                            c=colors[n], alpha=0.2, label=label)
            plt.title(f'sample_size={sample_size}, perplexity={perplexity}',
                      fontsize=15, color='black')
            # plt.legend()
            # legend = plt.legend(loc="upper right", edgecolor="black")
            # legend.get_frame().set_alpha(None)
            # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
            leg = plt.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            if save_plot:
                plt.savefig(f'plots/{tag} 2D_tSNE_{uuid.uuid4()}')
            # plt.show()
            return plt

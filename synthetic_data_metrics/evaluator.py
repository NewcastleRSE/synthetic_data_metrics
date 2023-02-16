from synthetic_data_metrics.utils import (get_inception_softmax_score,
                                          calculate_2pca, clean_time_series,
                                          get_train_test_split)
from synthetic_data_metrics.plot import plot_tsne, plot_2pca
from synthetic_data_metrics.metrics.image_metrics import (
    calc_inception_score, calc_dim_reduced_iou, calc_dim_reduced_dice,
)
from synthetic_data_metrics.metrics.ts_metrics import calculate_ds
from pathlib import Path


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
        self._metrics = ['inception_score', 'dim_reduced_iou_score',
                         'dim_reduced_dice_score', 'plot_2PC_compare']
        self.inception_softmax_scores = None
        self.two_pca_values_synth = None
        self.two_pca_values_real = None

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
            self.inception_softmax_scores = get_inception_softmax_score(
                                                self.synth_data, n_splits)

        mean, std = calc_inception_score(self.inception_softmax_scores)
        return mean, std

    def dim_reduced_iou_score(self):
        """
        Reduce the dimensions of both datasets to 2 principal componenets and
        calculate IoU score between these two 2PC arrays.
        Paremeters:
                n_components (int): set by default to 2 components.
            Returns:
                IoU (float): The IoU score between the two 2PC arrays
                of given two datasets.
        """
        if self.two_pca_values_synth is None:
            self.two_pca_values_synth = calculate_2pca(
                                                self.synth_data)
        if self.two_pca_values_real is None:
            self.two_pca_values_real = calculate_2pca(
                                                self.real_data)
        dim_reduced_iou = calc_dim_reduced_iou(self.two_pca_values_synth,
                                               self.two_pca_values_real)
        return dim_reduced_iou

    def dim_reduced_dice_score(self):
        """
        Reduce the dimensions of both datasets to 2 principal componenets
        and calculate Dice score between these two 2PC arrays.
        Paremeters:
                n_components (int): set by default to 2 components.
        Returns:
                Dice (float): The dice score between the two 2PC arrays of
                given two datasets.
        """
        if self.two_pca_values_synth is None:
            self.two_pca_values_synth = calculate_2pca(
                                                self.synth_data)
        if self.two_pca_values_real is None:
            self.two_pca_values_real = calculate_2pca(
                                                self.real_data)
        dim_reduced_dice = calc_dim_reduced_dice(self.two_pca_values_synth,
                                                 self.two_pca_values_real)
        return dim_reduced_dice

    def plot_2PC_compare(self, save_plot):
        """A function to plot overlap of the first two principal
        components of the two datasets.

        Args:
            dataset1: The first dataset.
            dataset2: The second datasets.
            save_plot (bool, optional): whether to save the plot.
                                        Defaults to False.
        """
        if self.two_pca_values_synth is None:
            self.two_pca_values_synth = calculate_2pca(
                                                self.synth_data)
        if self.two_pca_values_real is None:
            self.two_pca_values_real = calculate_2pca(
                                                self.real_data)
        return plot_2pca(self.two_pca_values_synth, self.two_pca_values_real,
                         save_plot)


class TS_Evaluator:
    """
    The central class TS_Evaluator,
    used to evaluate synthetic time-series data.

        Parameters:
            real (pd.DataFrame): Any Pandas dataframe.
            synth (pd.DataFrame): Any Pandas dataframe.
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

    def __init__(self, real, synth, target=None):
        """
        Constructs all the necessary attributes to create a TS_Evaluator.

            Parameters:
                real (np.ndarray): 4D array containing data with `uint8` type.
                synth (np.ndarray): 4D array containing data with `uint8` type.
                target (String, optional): The name of the
                    data's target column.

            Returns:
                None
        """
        self.synth_data = synth
        self.real_data = real
        self.target = target
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

    def discriminative_score(self, window_size=10, step=1,
                             epochs=20, verbose=False):
        """
        Runs the discriminative score evaluation metric.
        Code implementation based on:
        https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/discriminative_metrics.py

            Paremeters:
                window_size (int, optional): Determines the size
                    of the moving window.
                step (int, optional): The sliding window overlap.
                epochs (int, optional): The number of epochs
                    to train the model.
                verbose (String/int, optional): The verbositiy
                    of the model's training process.

            Returns:
                score (float): The discriminative score
                    for the real and synthetic data.
        """
        real, synth = clean_time_series(self.real_data, self.synth_data,
                                        self.target)
        disc_scores = []
        if self.target is not None:
            # retrive all the unique labels
            labels = self.real_data[self.target].unique()
            for label in labels:
                X_train, y_train, X_test, y_test = get_train_test_split(
                    real, synth, self.target,
                    label=label,
                    window_size=window_size,
                    step=step)
                ds_temp = calculate_ds(X_train, y_train, X_test, y_test,
                                       epochs, verbose)
                disc_scores.append(ds_temp)
        else:
            X_train, y_train, X_test, y_test = get_train_test_split(
                real, synth, self.target,
                window_size=window_size,
                step=step)
            result = calculate_ds(X_train, y_train, X_test, y_test, epochs,
                                  verbose)
            disc_scores.append(result)
        score = sum(disc_scores)/len(disc_scores)
        return round(score, 2)

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

        real = self.real_data.copy()
        synth = self.synth_data.copy()
        if save_plot:
            Path("plots").mkdir(parents=True, exist_ok=True)
        real, synth = clean_time_series(real, synth, self.target)
        return plot_tsne(real=real, synth=synth, target=self.target,
                         sample_size=sample_size, perplexity=perplexity,
                         save_plot=save_plot, tag=tag)

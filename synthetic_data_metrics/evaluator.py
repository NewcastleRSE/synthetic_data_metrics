from synthetic_data_metrics.utils import get_inception_features
from synthetic_data_metrics.metrics.image_metrics import inception_score


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

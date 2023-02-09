"""Implementation of Inception Score.

Data is passed through Google's pre-trained and open source classifier,
Inception-v3. For each image passed through the classifier, the output is a
label distribution - a value between 0 and 1 for each class in the dataset,
which together sum to 1 and provide the probability that the image belongs
to the class. The sum over all label distributions is the marginal
distribution.
Inception Score then works on the premise that, in a goodsynthetic dataset,
every image should distinctly look like something (narrow label distributions)
and the dataset as a whole should have variety (wide marginal distribution).
Since these shapes are the opposite of each other, a good synthetic image
dataset should produce a high KL divergence between the label distribution
and the marginal distribution for every image.

Inception Score is the average of this divergence measure across samples. The
metric produces a value between 1.0 and the number of classes in the dataset.
The higher the value the better the dataset.

Requirements:
- Synthetic image data (generated using ImageNet, a subset of ImageNet, or
  ImageNet-like data). Paper recommends 50k+ images.

This implementation was sourced from
https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/

"""

from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp


def inception_score(softmax_list,  eps=1e-16) -> float:
    """Calculates the Inception Score.

    This function calculates the score (KL divergence between the label
    distribution and marginal distribution) for each image in images.

    Args:
        - features (array) : list of Array of features to evaluate.
        list of length splits containing array size [n_samples/n_splits, 1000]
        - eps (#TODO) : #TODO.

    Returns:
        The mean and standard deviation of the scores.

    Raises:
        Checks images is a nested array of values in the range [0,255].
    """

    # TODO: add assertions here

    scores = []

    for split_data in softmax_list:
        # average across all pieces of data
        p_y = expand_dims(split_data.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = split_data * (log(split_data + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        score = exp(avg_kl_d)
        print(score)
        # store

        scores.append(score)

    # average across images
    is_avg, is_std = mean(scores), std(scores)

    return is_avg, is_std

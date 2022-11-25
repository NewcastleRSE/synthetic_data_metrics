"""Implementation of Inception Score.

Data is passed through Google's pre-trained and open source classifier, 
Inception-v3. For each image passed through the classifier, the output is a 
label distribution - a value between 0 and 1 for each class in the dataset, 
which together sum to 1 and provide the probability that the image belongs 
to the class. The sum over all label distributions is the marginal distribution. 
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

from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from synthetic_data_metrics.utils import scale_images

def calculate_inception_score(images, n_split=10, eps=1E-16) -> float:
    """ Calculates the Inception Score.
    
    This function calculates the score (KL divergence between the label
    distribution and marginal distribution) for each image in images.
    
    Args:
        images (array) : Array of images to evaluate.
        n_split (int) : #TODO.
        eps (#TODO) : #TODO.
        
    Returns:
        The mean and standard deviation of the scores.
        
    Raises:
        Checks images is a nested array of values in the range [0,255]. 
    """

    #TODO: add assertions here

    # load inception v3 model
    model = InceptionV3()

    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)

    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)

    # average across images
    is_avg, is_std = mean(scores), std(scores)
    
    return is_avg, is_std

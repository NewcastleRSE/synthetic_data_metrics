from synthetic_data_metrics.metrics.inception_score import (
    calculate_inception_score)
from synthetic_data_metrics.datasets import load_cifar10
from numpy.random import shuffle

# load dataset
images, _, _, _ = load_cifar10()

# shuffle dataset
shuffle(images)

# reduce dataset size as quick fix to memory allocation error
images = images[:10000]

# print dataset size
print('loaded', images.shape)

# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)

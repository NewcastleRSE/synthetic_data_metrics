from synthetic_data_metrics import evaluator
from synthetic_data_metrics.datasets import load_cifar10
from numpy.random import shuffle


# load dataset
images, _, _, _ = load_cifar10()

# shuffle dataset
shuffle(images)

# reduce dataset size as quick fix to memory allocation error
images = images

# print dataset size
print('loaded', images.shape)

# calculate inception score
inception_ev = evaluator.Inception_Evaluator(images)
inception_ev.add_metric('calculate_inception_score')
print(inception_ev.evaluate())
# is_avg, is_std = calculate_inception_score(images)
# print('score', is_avg, is_std)

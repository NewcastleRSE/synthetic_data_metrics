from synthetic_data_metrics import evaluator
from synthetic_data_metrics.datasets import load_cifar10


# load dataset
images, _, _, _ = load_cifar10()

# reduce dataset size as quick fix to memory allocation error
images = images[:500]

# print dataset size
print('loaded', images.shape)

# calculate inception score
evaluator = evaluator.Image_Evaluator(images)

inception_results = evaluator.inception_score(n_splits=50)

print(inception_results)

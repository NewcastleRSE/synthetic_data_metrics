from skimage.transform import resize
from numpy import asarray
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from math import floor
from numpy.random import shuffle
from scipy import stats


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# function for creating own data split with train_test_split?


def get_inception_features(images, n_splits=20, eps=1e-16) -> float:

    print('retrieving inception softmax scores')

    shuffle(images)

    model = InceptionV3()

    # enumerate splits of images/predictions
    softmax_scores = list()
    n_part = floor(images.shape[0] / n_splits)

    for i in range(n_splits):
        print('iteration {} of {}'.format(i+1, n_splits))

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


# prepare timeseries data in windows of fixed size.
def prep_data_updated(X, y, window_size, step):
    data = []
    labels = []
    for i in range(0, X.shape[0] - window_size, step):
        _data = X.values[i: i + window_size]
        _y = stats.mode(y[i: i + window_size])[0][0]
        data.append(_data)
        labels.append(_y)
    return data, labels

# check if a column is categorical
def is_categorical(col):
    return col.dtype.name == 'object'
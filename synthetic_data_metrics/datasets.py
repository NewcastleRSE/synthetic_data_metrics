from keras.datasets import cifar10
import pandas as pd

DATASETS = {
    'cifar10': 'load_cifar10',
    'timeseries': 'load_timeseries'
}


def load_cifar10():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    return x_train, y_train, x_test, y_test


def load_timeseries():
    real = pd.read_csv(
        'https://figshare.com/ndownloader/files/39144212?'
        'private_link=5c282677d58e00da7d5c'
        )
    synth = pd.read_csv(
        'https://figshare.com/ndownloader/files/39144203?'
        'private_link=5c282677d58e00da7d5c'
        )
    return real, synth

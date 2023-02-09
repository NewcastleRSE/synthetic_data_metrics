from keras.datasets import cifar10
import pandas as pd


def load_cifar10():
    """
    Loads the CIFAR-10 dataset, an image dataset.
    For more information rgearding the dataset see:
    https://www.cs.toronto.edu/~kriz/cifar.html

        Parameters:
            None

        Returns:
            x_train (np.ndarray): 4D array containing data with `uint8` type.
            y_train (np.ndarray): 2D array containing data with `uint8` type.
            x_test (np.ndarray): 4D array containing data with `uint8` type.
            y_test (np.ndarray): 2D array containing data with `uint8` type.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    return x_train, y_train, x_test, y_test


def load_wisdm():
    """
    Loads the WISDM dataset, a time-series dataset of
    activity and biometrics recognition, and a syntheticly
    generated equivalent produced by Naif Alzahrani (Newcastle University).
    For more information see:
    https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+

        Parameters:
            None

        Returns:
            real: (pd.DataFrame):
                Shape:
                    6311 x 15
                Index:
                    RangeIndex
                Columns:
                    Name: time, dtype: object
                    Name: ACTIVITY, dtype: float64
                    Name: XAVG, dtype: float64
                    Name: YAVG, dtype: float64
                    Name: ZAVG, dtype: float64
                    Name: XPEAK, dtype: float64
                    Name: YPEAK, dtype: float64
                    Name: ZPEAK, dtype: float64
                    Name: XABSOLDEV, dtype: float64
                    Name: YABSOLDEV, dtype: float64
                    Name: ZABSOLDEV, dtype: float64
                    Name: XSTANDEV, dtype: float64
                    Name: YSTANDEV, dtype: float64
                    Name: ZSTANDEV, dtype: float64
                    Name: RESULTANT, dtype: float64
            synth: (pd.DataFrame):
                Shape:
                    15 x 5750
                Index:
                    RangeIndex
                Columns:
                    Name: XAVG, dtype: float64
                    Name: YAVG, dtype: float64
                    Name: ZAVG, dtype: float64
                    Name: XPEAK, dtype: float64
                    Name: YPEAK, dtype: float64
                    Name: ZPEAK, dtype: float64
                    Name: XABSOLDEV, dtype: float64
                    Name: YABSOLDEV, dtype: float64
                    Name: ZABSOLDEV, dtype: float64
                    Name: XSTANDEV, dtype: float64
                    Name: YSTANDEV, dtype: float64
                    Name: ZSTANDEV, dtype: float64
                    Name: RESULTANT, dtype: float64
                    Name: time, dtype: object
                    Name: ACTIVITY, dtype: float64
    """
    real = pd.read_csv(
        'https://figshare.com/ndownloader/files/39144212?'
        'private_link=5c282677d58e00da7d5c'
        )
    synth = pd.read_csv(
        'https://figshare.com/ndownloader/files/39144203?'
        'private_link=5c282677d58e00da7d5c'
        )
    return real, synth

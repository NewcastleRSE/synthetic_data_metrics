from keras.datasets import cifar10

def load_cifar10():
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    return x_train, y_train, x_test, y_test


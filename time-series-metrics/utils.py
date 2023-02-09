from scipy import stats

# prepare train/test in windows of fixed size.
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
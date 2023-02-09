import pandas as pd
def load_timeseries():
    real = pd.read_csv('https://figshare.com/ndownloader/files/39144212?private_link=5c282677d58e00da7d5c') #noqa
    synth = pd.read_csv('https://figshare.com/ndownloader/files/39144203?private_link=5c282677d58e00da7d5c') #noqa
    return real, synth

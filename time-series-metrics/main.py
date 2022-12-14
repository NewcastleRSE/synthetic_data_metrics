"""Driver class to run multiple metrics for Multivariate Time-series data

Example:
python main.py

this will run the script with the default arguments
provided below. There are three synthetic datasets
provided in /data/ folder.
To pass a different synthetic dataset, or to use all
three synthetic dataset, simply uncomment and edit the
following
line:

config['synth_list'] = ['data/synth/gan_synth.csv',
                        'data/synth/par_synth.csv',
                        'data/synth/timeGAN_synth.csv']

"""
import pandas as pd
from t_SNE import t_sne_2d
from discriminative_score import dis_score
import argparse

parser = argparse.ArgumentParser(description="PLOTTING t-SNE EMBEDDINGS",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter) # noqa
parser.add_argument("--real", type=str,
                    default='data/real/REAL_WISDM.csv',
                    help="full path to real dataset",)
parser.add_argument("--synth_list", nargs="*", type=str,
                    default=['data/synth/gan_synth.csv'],
                    help="list of paths to synth datasets")
parser.add_argument("--per", type=int, default=30,
                    help='perplexity value for t-SNE')
parser.add_argument("--sample_size", type=int, default=500,
                    help='number of samples to plot')
parser.add_argument("--target", type=str, default=None,
                    help='name of the target column')
parser.add_argument("--save_plot", default=True,
                    help='to save the t-SNE plot to disk')
args = parser.parse_args()
config = vars(args)

target = config['target']
# this example dataset contains a target column
# called 'ACTIVITY', so I'll override it
target = 'ACTIVITY'
[print(f'target column is {target}') if target is not None
 else print("No target column was provided")]

sample_size = config['sample_size']
perplexity = config['per']
save_plot = config['save_plot']
real = pd.read_csv(config['real'])
synth = []
# config['synth_list'] = ['data/synth/gan_synth.csv',
#                         'data/synth/par_synth.csv',
#                         'data/synth/timeGAN_synth.csv']
for file in config['synth_list']:
    synth_data = pd.read_csv(file)
    synth.append(synth_data)
# t-SNE plot
t_sne_2d(real, synth, target=target,  sample_size=sample_size,
         perplexity=perplexity, save_plot=save_plot, tag='')
# Discriminative Score
# Note that this metric accept only one synthetic dataset at a time,
# thus synth[0] to use the first synthetic dataset only.
n = dis_score(real, synth[0], target=target)
print('DS scores: ', n)

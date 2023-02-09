"""This function plots 2-D embeddings for multiple datasets
for visual inspection of similarity.

It receives the real dataset (DataFrame) and a list of synthetic
datasets (could also be one synthetic dataset). If the dataset
includes a target (label) column, it must be passed to the function.

Exaple:

from t-SNE import scrambled_2
scrambled_2(real, [synth], sample_size=600, save_plot=True)

This line will take 600 samples from each dataset and
plot the 2-d t-SNE representaions of them and save
the plot to folder /plots/

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.manifold import TSNE
import uuid
import matplotlib.colors as mcolors
import math
from pathlib import Path
from utils import is_categorical
warnings.filterwarnings("ignore", category=FutureWarning)

def tsne_3(data, perplexity=30):
    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=500, random_state=123)
    all_data = np.concatenate(data)
    tsne_results = pd.DataFrame(tsne.fit_transform(all_data))
    return tsne_results

def t_sne_2d(real, synth, target: str = None,
             sample_size=500, perplexity=40,
             save_plot: bool = False, tag=''):
    """A function to plot tSNE 2d embeddings of multiple
        generated datasets along with the original dataset.

    Args:
        real (_type_): The real dataset.
        synth: The synthetic datasets.
        target (str): name of the target (label) column. Defaults to None.
        sample_size (int, optional): take only s subset of this length from
                                    each label. Defaults to 500.
        perplexity (int, optional): internal t-SNE hyperparameter.
                                    Defaults to 500.
        save_plot (bool, optional): whether to save the plot.
                                    Defaults to False.
        tag (str, optional): to be added to the name of
                            the saved plot. Defaults to ''.
    """

    print('Plotting t-SNE of the real and synthetic data')
    # creat a folder to save plots
    real = real.copy()
    synth = synth.copy()
    if save_plot:
        Path("plots").mkdir(parents=True, exist_ok=True)
    # Naively remove the time channel if it exists
    for col in ['time', 'Time', 'Date', 'date']:
        if col in real.columns:
            real.drop(col, axis=1, inplace=True)
        if col in synth.columns:
            synth.drop(col, axis=1, inplace=True)
    for col in real.columns:
        if is_categorical(real[col]):
            real[col] = pd.factorize(real[col])[0]
            synth[col] = pd.factorize(synth[col])[0]
    # get number of unique labels
    num_labels = len(real[target].unique())
    if num_labels > 1:
        print(f'Splitting datasets by label column ({target})...')
        labels = real[target].unique()
        colors = list(mcolors.BASE_COLORS)
        ncols = 2
        nrows = math.ceil(len(labels)/ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15),
                                constrained_layout=True)
        for label, ax in enumerate(fig.axes):
            if label < len(labels):
                # print(f'Plotting label {label}')
                chosen = [label]
                all_data = []
                real_temp = real.loc[real[target].isin(chosen)].copy()
                assert sample_size <= len(real_temp), "sample size"
                "should be smaller"
                "than the number of samples for each label"
                real_temp = real_temp[:sample_size]
                all_data.append(real_temp)
                synth_temp = synth.loc[synth[target].isin(chosen)].copy()
                synth_temp = synth_temp[:sample_size]
                synth_temp.dropna(inplace=True)
                all_data.append(synth_temp)
                for dataset in all_data:
                    dataset.drop([target], axis=1, inplace=True)
                tsne_results = tsne_3(all_data,
                                      perplexity=perplexity)
                ax.set_title(f'label {label}', fontsize=10,
                             color='black', pad=10)
                labels_for_legend = []
                for n, i in enumerate(np.arange(start=0,
                                                stop=len(tsne_results),
                                                step=sample_size)):
                    if n == 0:
                        ax_label = 'REAL'
                    else:
                        ax_label = 'SYNTH'
                    labels_for_legend.append(ax_label)
                    ax.scatter(tsne_results.iloc[i:i+sample_size, 0].values,
                               tsne_results.iloc[i:i+sample_size, 1].values,
                               c=colors[n], alpha=0.2, label=ax_label)
                fig_dim = 45
                plt.xlim(-fig_dim, fig_dim)
                plt.ylim(-fig_dim, fig_dim)
                fig.suptitle(f'sample_size={sample_size}'
                             f' perplexity={perplexity}',
                             fontsize=15, color='black')
                # ax.axis('scaled')
                # ax.legend()
                leg = ax.legend()
                for lh in leg.legendHandles:
                    lh.set_alpha(1)
        if len(labels) % 2 != 0:
            axs.flat[-1].set_visible(False)  # to remove last plot
        # plt.legend()
        if save_plot:
            plt.savefig(f'plots/{tag} 2D_tSNE_{uuid.uuid4()}')
        # plt.show()
        return plt
    else:
        real.drop('ACTIVITY', axis=1, inplace=True)
        synth.drop('ACTIVITY', axis=1, inplace=True)
        assert sample_size <= len(real), "sample size should be smaller than the size of the dataset" # noqa
        print(f'Plotting {sample_size} samples from each dataset...')
        colors = list(mcolors.BASE_COLORS)
        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        all_data = []
        real_temp = real[:sample_size]
        all_data.append(real_temp)
        synth_temp = synth[:sample_size]
        all_data.append(synth_temp)
        tsne_results = tsne_3(all_data,  perplexity=perplexity)
        for n, i in enumerate(np.arange(start=0, stop=len(tsne_results),
                                        step=sample_size)):
            label = 'real' if n == 0 else 'synth'
            plt.scatter(tsne_results.iloc[i:i+sample_size, 0].values,
                        tsne_results.iloc[i:i+sample_size, 1].values,
                        c=colors[n], alpha=0.2, label=label)
        plt.title(f'sample_size={sample_size}, perplexity={perplexity}',
                  fontsize=15, color='black')
        # plt.legend()
        # legend = plt.legend(loc="upper right", edgecolor="black")
        # legend.get_frame().set_alpha(None)
        # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
        leg = plt.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        if save_plot:
            plt.savefig(f'plots/{tag} 2D_tSNE_{uuid.uuid4()}')
        # plt.show()
        return plt
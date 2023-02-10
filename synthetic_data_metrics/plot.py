import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import math
import uuid
import numpy as np
from synthetic_data_metrics.utils import calculate_tsne


def plot_tsne(real, synth, sample_size, perplexity,
              target, save_plot, tag):
    if save_plot:
        Path("plots").mkdir(parents=True, exist_ok=True)
    num_labels = len(real[target].unique())
    if num_labels > 1:
        labels = real[target].unique()
        colors = list(mcolors.BASE_COLORS)
        ncols = 2
        nrows = math.ceil(len(labels)/ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15),
                                constrained_layout=True)
        for label, ax in enumerate(fig.axes):
            if label < len(labels):
                chosen = [label]
                all_data = []
                real_temp = real.loc[real[target].isin(chosen)].copy()
                assert sample_size <= len(real_temp), "sample size"
                "should be smaller"
                "than the number of samples for each label"
                real_temp = real_temp[:sample_size]
                all_data.append(real_temp)
                synth_temp = synth.loc[synth[target].isin(chosen)].copy() # noqa
                synth_temp = synth_temp[:sample_size]
                synth_temp.dropna(inplace=True)
                all_data.append(synth_temp)
                for dataset in all_data:
                    dataset.drop([target], axis=1, inplace=True)
                tsne_results = calculate_tsne(all_data, perplexity=perplexity)
                ax.set_title(f'label {label}', fontsize=10, color='black',
                             pad=10)
                labels_for_legend = []
                for n, i in enumerate(np.arange(start=0,
                                                stop=len(tsne_results),
                                                step=sample_size)):
                    if n == 0:
                        ax_label = 'REAL'
                    else:
                        ax_label = 'SYNTH'
                    labels_for_legend.append(ax_label)
                    ax.scatter(tsne_results.iloc[i:i+sample_size, 0].values, # noqa
                               tsne_results.iloc[i:i+sample_size, 1].values, # noqa
                               c=colors[n], alpha=0.2, label=ax_label)
                fig_dim = 45
                plt.xlim(-fig_dim, fig_dim)
                plt.ylim(-fig_dim, fig_dim)
                fig.suptitle(f'sample_size={sample_size}'
                             f' perplexity={perplexity}',
                             fontsize=15, color='black')
                leg = ax.legend()
                for lh in leg.legendHandles:
                    lh.set_alpha(1)
        if len(labels) % 2 != 0:
            axs.flat[-1].set_visible(False)  # to remove last plot
        if save_plot:
            plt.savefig(f'plots/{tag} 2D_tSNE_{uuid.uuid4()}')
        return plt
    else:
        real.drop(target, axis=1, inplace=True)
        synth.drop(target, axis=1, inplace=True)
        assert sample_size <= len(real), "sample size should be smaller than the size of the dataset" # noqa
        colors = list(mcolors.BASE_COLORS)
        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        all_data = []
        real_temp = real[:sample_size]
        all_data.append(real_temp)
        synth_temp = synth[:sample_size]
        all_data.append(synth_temp)
        tsne_results = calculate_tsne(all_data,  perplexity=perplexity)
        for n, i in enumerate(np.arange(start=0, stop=len(tsne_results),
                                        step=sample_size)):
            label = 'real' if n == 0 else 'synth'
            plt.scatter(tsne_results.iloc[i:i+sample_size, 0].values,
                        tsne_results.iloc[i:i+sample_size, 1].values,
                        c=colors[n], alpha=0.2, label=label)
        plt.title(f'sample_size={sample_size}, perplexity={perplexity}',
                  fontsize=15, color='black')
        leg = plt.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        if save_plot:
            plt.savefig(f'plots/{tag} 2D_tSNE_{uuid.uuid4()}')
        return plt


def plot_2pca():
    return None

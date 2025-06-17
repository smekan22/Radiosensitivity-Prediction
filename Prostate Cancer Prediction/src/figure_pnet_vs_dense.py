import os
import pandas as pd
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class DataLoaderComparativeAnalysis:

    def __init__(self, results_file_path):
        self.results_file_path = results_file_path

    def load_data(self):
        results = pd.read_csv(self.results_file_path)
        return results


class FigureComparativeAnalysisConfiguration(Enum):
    plot_size = (10, 7)

    pnet_auc_color = 'blue'
    dense_auc_color = 'orange'

    marker = "."
    marker_size = 14

    pnet_label = 'P-NET'
    dense_label = 'Dense'

    legend_fontsize = 12
    legend_frame = False

    top_spine_visibility = False
    bottom_spine_visibility = True
    left_spine_visibility = True
    right_spine_visibility = True

    spine_thickness = 1
    tick_size = 14
    label_size = 14

    right_spine_color = 'red'

    y_axis_limit = (0.4, 1)

class ComparativeAnalysis:

    def __init__(self, results):
        self.results = results
        self.config = FigureComparativeAnalysisConfiguration

    def process_results(self):
        self.number_of_samples = np.array(self.results['number_of_samples'])
        self.pnet_auc = np.array(self.results['pnet_auc'])
        self.dense_auc = np.array(self.results['dense_auc'])
        self.pnet_lower_bound = np.array(self.results['pnet_lower_bound'])
        self.pnet_upper_bound = np.array(self.results['pnet_upper_bound'])
        self.dense_lower_bound = np.array(self.results['dense_lower_bound'])
        self.dense_upper_bound = np.array(self.results['dense_upper_bound'])
        self.statistically_significant = np.array(self.results['statistically_significant'])

    def compute_xticks(self):
        maximum_x_value = max(self.number_of_samples)
        minimum_x_value = min(self.number_of_samples)
        space = int(maximum_x_value/len(self.number_of_samples))-1
        x_ticks = range(minimum_x_value, maximum_x_value, space)
        return x_ticks
    
    def compute_xtickslabels(self):
        significance = ['*' if sig==1 else 'NS' for sig in self.statistically_significant]
        x_ticks_labels = [str(nb_sample) + '\n' + sig for nb_sample, sig in zip(self.number_of_samples, significance)]
        return x_ticks_labels
    
    def compute_yticks_and_ytickslabels(self):
        y_values = self.pnet_auc.tolist() + self.dense_auc.tolist()
        minimum_y_value = min(y_values)
        maximum_y_value = max(y_values)
        lower = round(minimum_y_value,1)
        mid = round((minimum_y_value+maximum_y_value)/2,1)
        upper = round(maximum_y_value, 1)
        y_ticks = [lower, mid, upper]
        y_ticks_labels = y_ticks
        return [0.6, 0.7, 0.8, 0.9], [0.6, 0.7, 0.8, 0.9] #y_ticks, y_ticks_labels
    
    def _configure_spine_visibility(self, ax):
        ax.spines['right'].set_visible(self.config.right_spine_visibility.value)
        ax.spines['left'].set_visible(self.config.left_spine_visibility.value)
        ax.spines['top'].set_visible(self.config.top_spine_visibility.value)
        ax.spines['bottom'].set_visible(self.config.bottom_spine_visibility.value)
        return ax
    
    def _configure_spine_linewidth(self, ax):
        ax.spines['left'].set_linewidth(self.config.spine_thickness.value)
        ax.spines['bottom'].set_linewidth(self.config.spine_thickness.value)
        ax.spines['right'].set_linewidth(self.config.spine_thickness.value)
        return ax
    
    def format_spines(self, ax):
        ax = self._configure_spine_visibility(ax)
        ax = self._configure_spine_linewidth(ax)

        ax.set_ylim(*self.config.y_axis_limit.value)

        ax.tick_params(axis='x', labelsize=self.config.tick_size.value)
        ax.tick_params(axis='y', labelsize=self.config.tick_size.value)

        ax.set_xlabel('Number of samples', fontsize=self.config.label_size.value)
        ax.set_ylabel('AUC', fontsize=self.config.label_size.value)

        ax.spines['right'].set_color('red')

        ax2 = ax.twinx()
        ax2 = self._configure_spine_visibility(ax2)
        ax2 = self._configure_spine_linewidth(ax2)
        ax2.spines['right'].set_color('red')
        ax2.set_yticks([])
        ax2.set_ylabel('Performance increase',
                       color=self.config.right_spine_color.value,
                       fontsize=self.config.label_size.value,
                       labelpad=6)

        return ax

    def plot(self, filename, dense_label=None, save=False, save_dir=None, show=False):
        self.process_results()

        fig, ax = plt.subplots(figsize=self.config.plot_size.value)

        x_ticks = self.compute_xticks()
        y_ticks, y_ticks_labels = self.compute_yticks_and_ytickslabels()
        x_ticks_labels = self.compute_xtickslabels()

        ax.plot(x_ticks,
                self.pnet_auc,
                c=self.config.pnet_auc_color.value,
                marker=self.config.marker.value,
                markersize=self.config.marker_size.value,
                label = self.config.pnet_label.value)
        
        ax.plot(x_ticks,
                self.dense_auc,
                c=self.config.dense_auc_color.value,
                marker=self.config.marker.value,
                markersize=self.config.marker_size.value,
                label = self.config.dense_label.value if dense_label is None else dense_label)
        
        ax.legend(loc='upper left',
                  frameon=self.config.legend_frame.value,
                  fontsize=self.config.legend_fontsize.value)
        
        for i in range(len(self.number_of_samples)-1):
            x = self.number_of_samples[i]
            next_x = self.number_of_samples[i+1]
            lower_bound = self.pnet_lower_bound[i]
            next_lower_bound = self.pnet_lower_bound[i+1]
            upper_bound = self.pnet_upper_bound[i]
            next_upper_bound = self.pnet_upper_bound[i+1]
            ax.fill_between([x, next_x],
                            [lower_bound, next_lower_bound],
                            [upper_bound, next_upper_bound],
                            alpha=0.15,
                            color=self.config.pnet_auc_color.value,
                            edgecolor=None)

        for i in range(len(self.number_of_samples)-1):
            x = self.number_of_samples[i]
            next_x = self.number_of_samples[i+1]
            lower_bound = self.dense_lower_bound[i]
            next_lower_bound = self.dense_lower_bound[i+1]
            upper_bound = self.dense_upper_bound[i]
            next_upper_bound = self.dense_upper_bound[i+1]
            ax.fill_between([x, next_x],
                            [lower_bound, next_lower_bound],
                            [upper_bound, next_upper_bound],
                            alpha=0.15,
                            color=self.config.dense_auc_color.value,
                            edgecolor=None)

        ax = self.format_spines(ax)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_ticks_labels)
        ax.set_yticklabels(y_ticks_labels)

        if save:
            save_dir = os.getcwd() if (save_dir is None) else save_dir
            save_file_path = os.path.join(save_dir, filename)
            fig.savefig(save_file_path)

        if show:
            plt.show()
        else:
            plt.close()
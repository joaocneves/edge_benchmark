import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_stats(RESULTS_DIR):

    aux = pd.DataFrame()

    for i in os.listdir(RESULTS_DIR):
        if os.path.isfile(os.path.join(RESULTS_DIR, i)) and 'stats_' in i:
            aux = pd.concat([aux, pd.read_csv(os.path.join(RESULTS_DIR, i))], axis=0, ignore_index=True)

    return aux

def create_barh(df1, df2):

    name = df1.index
    avg_time1 = df1['Average Time']
    std_time1 = df1['Std Time']
    avg_time2 = df2['Average Time']
    std_time2 = df2['Std Time']

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    bar_width = 0.4
    ind = np.arange(len(name))

    # Horizontal Bar Plot
    ax.barh(ind, avg_time1, bar_width, color=[31 / 255.0, 119 / 255.0, 180 / 255.0], label='venv')
    ax.barh(ind + bar_width, avg_time2, bar_width, color=[180 / 255.0, 31 / 255.0, 119 / 255.0, ], label='conda')
    ax.set(yticks=ind + bar_width/2, yticklabels=name, ylim=[2 * bar_width - 1, len(ind)])
    ax.legend()

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(True)

    plt.xlabel('Inference Time (ms)')

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=1,
            alpha=0.2)

    ax.set_axisbelow(True)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.25,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Show Plot
    plt.show()

    return fig

if __name__  == '__main__':

    RESULTS_DIR = '..//'
    stats = load_stats(RESULTS_DIR)

    # ------ Detection ------- #

    # filter1 = stats["Device"].str.contains('CPU', na=False)
    # filter2 = stats["Model Type"] == 'Detection'
    # filter3 = stats["AVX"] == 0
    # filter4 = stats["Environment"] == 'venv'
    # # filtering data
    # df1 = stats.loc[filter1 & filter2 & filter3 & filter4]
    # df1 = df1.groupby('Framework').mean()
    #
    # filter1 = stats["Device"].str.contains('CPU', na=False)
    # filter2 = stats["Model Type"] == 'Detection'
    # filter3 = stats["AVX"] == 0
    # filter4 = stats["Environment"] == 'conda'
    # # filtering data
    # df2 = stats.loc[filter1 & filter2 & filter3 & filter4]
    # df2 = df2.groupby('Framework').mean()
    #
    # fig = create_barh(df1,df2)
    #
    # fig.savefig("performance_by_framework_Detection.pdf", bbox_inches='tight')

    # ------ Classification ------- #

    filter1 = stats["Device"].str.contains('CPU', na=False)
    filter2 = stats["Model Type"] == 'Classification'
    filter3 = stats["AVX"] == 0
    filter4 = stats["Environment"] == 'venv'
    # filtering data
    df1 = stats.loc[filter1 & filter2 & filter3 & filter4]
    df1 = df1.groupby('Framework').mean()

    filter1 = stats["Device"].str.contains('CPU', na=False)
    filter2 = stats["Model Type"] == 'Classification'
    filter3 = stats["AVX"] == 0
    filter4 = stats["Environment"] == 'conda'
    # filtering data
    df2 = stats.loc[filter1 & filter2 & filter3 & filter4]
    df2 = df2.groupby('Framework').mean()

    fig = create_barh(df1, df2)

    fig.savefig("performance_by_framework_Classification.pdf", bbox_inches='tight')


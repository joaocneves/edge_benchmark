import os
import pandas as pd
from matplotlib import pyplot as plt

def load_stats(RESULTS_DIR):

    aux = pd.DataFrame()

    for i in os.listdir(RESULTS_DIR):
        if os.path.isfile(os.path.join(RESULTS_DIR, i)) and 'stats_' in i:
            aux = pd.concat([aux, pd.read_csv(os.path.join(RESULTS_DIR, i))], axis=0, ignore_index=True)

    return aux

def create_barh(df):

    name = df.index
    avg_time = df['Average Time']
    std_time = df['Std Time']

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(name, avg_time, color=[31 / 255.0, 119 / 255.0, 180 / 255.0])

    locs, labels = plt.yticks()
    labels = [str.replace(' (', '\n(') for str in name.values]
    plt.yticks(locs, labels)

    # ax.barh(name, price, xerr=std_price, ecolor='grey', error_kw=dict(lw=4, capsize=4, capthick=2))

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
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Show Plot
    plt.show()

    return fig

if __name__  == '__main__':

    RESULTS_DIR = '..//'
    stats = load_stats(RESULTS_DIR)

    filter1 = stats["Framework"].str.startswith('Tensorflow 1.14.0', na=False)
    filter2 = stats["Model Type"] == 'Detection'
    # filtering data
    df = stats.loc[filter1 & filter2]
    df = df.groupby('Architecture').mean()

    fig = create_barh(df)

    fig.savefig("performance_by_model_Detection.pdf", bbox_inches='tight')

    filter1 = stats["Framework"].str.startswith('Tensorflow 1.14.0', na=False)
    filter2 = stats["Model Type"] == 'Classification'
    # filtering data
    df = stats.loc[filter1 & filter2]
    df = df.groupby('Architecture').mean()

    fig = create_barh(df)

    fig.savefig("performance_by_model_Classification.pdf", bbox_inches='tight')


import os
import pandas as pd
pd.options.display.float_format = '${:,.2f}'.format

def load_stats(RESULTS_DIR):

    aux = pd.DataFrame()

    for i in os.listdir(RESULTS_DIR):
        if os.path.isfile(os.path.join(RESULTS_DIR, i)) and 'stats_' in i:
            aux = pd.concat([aux, pd.read_csv(os.path.join(RESULTS_DIR, i))], axis=0, ignore_index=True)

    return aux

if __name__  == '__main__':

    RESULTS_DIR = '..//'
    stats = load_stats(RESULTS_DIR)
    stats['AVX'] = stats['AVX'].replace({0.0: 'NO-AVX', 1.0: 'AVX'})

    filter1 = stats["Framework"].str.startswith('T', na=False)
    filter2 = stats["Model Type"] == 'Classification'
    filter3 = stats["Device"].str.contains('CPU', na=False)
    filter4 = stats["Architecture"] == 'Inception-v4'

    # filtering data
    df = stats.loc[filter1 & filter2 & filter3 & filter4]
    df = df.groupby(['Framework', 'AVX']).mean().reset_index()

    df['Average Time'] = df['Average Time'].map('{:,.2f}'.format)
    df['Std Time'] = df['Std Time'].map('{:,.2f}'.format)
    df['Average Time'] = df['Average Time'].astype('str') + ' pm ' + df['Std Time'].astype('str')
    df1 = df.drop(columns=['Flops', 'Unnamed: 0', 'Std Time'])
    print(df1.to_latex(index=False, float_format='%.2f'))

    filter1 = stats["Framework"].str.startswith('T', na=False)
    filter2 = stats["Model Type"] == 'Detection'
    filter3 = stats["Device"].str.contains('CPU', na=False)
    filter4 = stats["Architecture"] == 'SSD (Inception-v2)'

    # filtering data
    df = stats.loc[filter1 & filter2 & filter3 & filter4]
    df = df.groupby(['Framework', 'AVX']).mean().reset_index()

    df['Average Time'] = df['Average Time'].map('{:,.2f}'.format)
    df['Std Time'] = df['Std Time'].map('{:,.2f}'.format)
    df['Average Time'] = df['Average Time'].astype('str') + ' pm ' + df['Std Time'].astype('str')
    df2 = df.drop(columns=['Flops', 'Unnamed: 0', 'Std Time'])
    print(df2.to_latex(index=False, float_format='%.2f'))

    result = pd.concat([df1, df2['Average Time']], axis=1)
    print(result.to_latex(index=False, float_format='%.2f'))

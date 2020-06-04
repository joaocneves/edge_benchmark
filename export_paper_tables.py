import os
import pandas as pd

RESULTS_DIR = '.'


aux = pd.DataFrame()

for i in os.listdir(RESULTS_DIR):
    if os.path.isfile(os.path.join(RESULTS_DIR,i)) and 'stats_' in i:
        aux = pd.concat([aux, pd.read_csv(os.path.join(RESULTS_DIR,i))], axis=0)

model_framework_table = aux
models = model_framework_table['Model']
frameworks = model_framework_table['Framework']

uniq_models = list(set(models))
uniq_frameworks = list(set(frameworks))

idx = 0
out_file = open("model_framework_table", "w")

# header

print(" & ", end='', file=out_file)
for f in uniq_frameworks:
    if f == uniq_frameworks[-1]:
        print("%s \n" % f, end='', file=out_file)
    else:
        print("%s & " % f, end='', file=out_file)

# body
for m in uniq_models:
    print("%s & " % model_framework_table.iloc[idx]['Model'], end='', file=out_file)
    for f in uniq_frameworks:
        if f == uniq_frameworks[-1]:
            print("%d \plusminus %d \n" %
                  (model_framework_table.iloc[idx]['Average Time'], model_framework_table.iloc[idx]['Std Time']),
                  end='', file=out_file)
        else:
            print("%d \plusminus %d & " %
                  (model_framework_table.iloc[idx]['Average Time'], model_framework_table.iloc[idx]['Std Time']),
                  end='', file=out_file)

        idx = idx + 1

out_file.close()
import pandas as pd
import numpy as np
import glob
files1 = glob.glob('training_setA/*.psv')
files2 = glob.glob('training_setB/*.psv')
files = np.concatenate((files1, files2))

df_list = []
for i, f in enumerate(files):
    id = f.split('/')[1].split('.')[0]
    df = pd.read_csv(f, sep='|')
    df = df.assign(patient=id)
    if i % 1000 == 0:
        print(i)
    df_list.append(df)

df = pd.concat(df_list)
df = df.reset_index(drop=True)
df.to_pickle('data.pkl')
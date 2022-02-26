import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import trimap
import pacmap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import plotly.express as px

def feature_selection(st, df, correlated=True, missing=True):
    if(missing):
        df = remove_missing(df)

    if(correlated):
        df = remove_correlated(st, df)

    return df

def impute_data(st, df, method='linear'): # https://stackoverflow.com/questions/37057187/pandas-interpolate-within-a-groupby /
    # https://stackoverflow.com/questions/55718026/how-to-interpolate-missing-values-with-groupby
    st.write(f"Interpolating with the method: {method}")
    df = df.groupby('patient').apply(lambda x: x.interpolate(method=method, limit_direction='both'))
    st.write("")
    st.write("Let's make sure that we don't have any more missing values")
    missing = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)
    st.write(missing)
    st.write("So we still have missing data, fill these in with the median values")
    for i in df.columns[df.isnull().any(axis=0)]: # https://stackoverflow.com/questions/37057187/pandas-interpolate-within-a-groupby
        df[i].fillna(df[i].mean(),inplace=True)
    st.write("")
    st.write("Do we have missing data now?")
    missing = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)
    st.write(missing)
    st.write("")
    st.write("Looks much better!")
    return scale(st, df)

def remove_correlated(st, df):
    fig = plt.figure(figsize=(16, 6))
    # Remove upper triangle
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Correlation Matrix');
    st.pyplot(fig)
    abs_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = abs_matrix.where(np.triu(np.ones(abs_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    st.write(f"Dropping {len(to_drop)} columns because they are highly correlated.")
    df.drop(to_drop, axis=1, inplace=True)
    return df

def remove_missing(df, percentage=0.61): # https://stackoverflow.com/questions/45515031/how-to-remove-columns-with-too-many-missing-values-in-python
    df = df.dropna(thresh=len(df) * percentage, axis=1)
    return df

def scale(st, df):
    df['ts'] = df.groupby(['patient']).cumcount()
    df.drop(['patient'], axis=1, inplace=True)
    st.write(df.head(100))
    st.write("Let's scale the data and then we're done with the dataset")
    cols = list(df.columns)
    cols.remove('SepsisLabel')
    sc = MinMaxScaler()
    df[cols] = sc.fit_transform(df[cols])
    st.write(df.head(100))
    return df, sc

def dimensionality_reduction(st, df, method='PaCMAP'):
    st.write(f"Let's use dimensionality reduction with the {method} method")
    df = df.head(500000)
    df.dropna(inplace=True)
    if method == 'TriMAP':
        dr_df = trimap.TRIMAP(verbose=True).fit_transform(df.drop(['SepsisLabel'], axis=1).values)
    elif method == 'PaCMAP':
        dr_df = pacmap.PaCMAP(verbose=True).fit_transform(df.drop(['SepsisLabel'], axis=1).values)
    else:
        st.write("No valid method selected")
        return None

    dr_df = pd.DataFrame(dr_df, columns=['x1','x2'])
    df['x1'] = dr_df['x1']
    df['x2'] = dr_df['x2']
    st.write(df.head(100))
    return df

def run_dr(st, input_df, name):
    for impute_method in ['linear', 'none']:
        df = input_df.copy()
        fname_impute = f'imputed_{impute_method}_{name}.pkl'
        if os.path.isfile(fname_impute):
            df = pd.read_pickle(fname_impute)
            sc = joblib.load(f'{impute_method}_{name}_scaler.bin')
        else:
            if impute_method != 'none':
                df, sc = impute_data(st, df, method=impute_method)
            else:
                df, sc = scale(st, df)
            df.to_pickle(fname_impute)
            joblib.dump(sc, f'{impute_method}_{name}_scaler.bin', compress=True)

        for dr_method in ['PaCMAP', 'TriMAP']:
            fname_dr = f'dr_{impute_method}_{dr_method}_{name}.pkl'
            if os.path.isfile(fname_dr):
                feature_df = pd.read_pickle(fname_dr)
            else:
                feature_df = dimensionality_reduction(st, df, method=dr_method)
                feature_df.to_pickle(fname_dr)
            st.write(f"Done running dimensionality reduction with imputation: {impute_method}, dr_method: {dr_method}")
            st.write("Let's see if we can find any clusters")
            for dim in ['SepsisLabel','Age','Gender']:
                st.write(f"Can we see clusters for {dim}")
                image_name = f"images/plot_{impute_method}_{dr_method}_{name}_{dim}.png"
                if os.path.isfile(image_name):
                    st.image(image_name)
                else:
                    f = px.scatter(feature_df, x='x1', y='x2', color=dim)
                    f.write_image(image_name)
                    st.plotly_chart(f)
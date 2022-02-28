import dimensionality_reduction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np

def classify(st, df, name, method):
    df = df.head(100000)
    cols = df.columns
    st.write(f"Let's start classifying our {name} dataset")
    st.write("First we divide it into a train and test dataset")
    if method == 'under':
        st.write("Let's undersample")
        X = df.drop(['SepsisLabel'], axis=1).values
        y = df['SepsisLabel'].values
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        X_res = pd.DataFrame(X_res)
        y_res = pd.DataFrame(y_res)
        df = pd.concat([X_res, y_res], axis=1)
        df.columns = cols
        st.write(f"Resampled shape: {df.shape}")
    elif method == 'over':
        st.write("Let's oversample")
        X = df.drop(['SepsisLabel'], axis=1).values
        y = df['SepsisLabel'].values
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        X_res = pd.DataFrame(X_res)
        y_res = pd.DataFrame(y_res)
        df = pd.concat([X_res, y_res], axis=1)
        df.columns = cols
        st.write(f"Resampled shape: {df.shape}")
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size,:]
    test_df = df.iloc[train_size+1:,:]
    train_df, train_sc = dimensionality_reduction.impute_data(st, train_df, method='linear')
    test_df, test_sc = dimensionality_reduction.impute_data(st, test_df, method='linear')
    train_df.to_pickle(f"data/classifier_imputed_train_{method}_{name}.pkl")
    test_df.to_pickle(f"data/classifier_imputed_test_{method}_{name}")
    df.dropna(inplace=True)
    neigh = KNeighborsClassifier(n_neighbors=3)
    try:
        train_df.drop('patient', axis=1, inplace=True)
        test_df.drop('patient', axis=1, inplace=True)
    except Exception:
        print('patient not found in columns')
    neigh.fit(train_df.drop(['SepsisLabel'], axis=1), train_df['SepsisLabel'].values)
    y_pred = neigh.predict(test_df.drop(['SepsisLabel'], axis=1))
    y_true = test_df['SepsisLabel'].values
    st.write(f"The F1 Score for {method} is: {f1_score(y_true, y_pred, average='weighted')}")
    st.write(f"The Accuracy Score for {method} is: {accuracy_score(y_true, y_pred)}")
    return [train_df, test_df]

def run_imbalanced(st, input_df, name):
    result = []
    for method in ['normal','under','over']:
        df = input_df.copy()
        result.append(classify(st, df, name, method))
    return result


import streamlit as st
import pandas as pd
import exploration
import dimensionality_reduction
import clustering
import subspace_clustering
import imbalanced


st.write("Data Challenges")
# st.write("Wessel van de Goor")

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_data():
    return pd.read_pickle('data/data.pkl')

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def explore(st, df):
    return exploration.explore(st, df)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def feature_selection(st, df):
    return dimensionality_reduction.feature_selection(st, df)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def impute_data(st, df, method='linear'):
    return dimensionality_reduction.impute_data(st, df, method=method)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def run_dr(st, df, name):
    return dimensionality_reduction.run_dr(st, df, name)

def run_cluster(st, df, method, name):
    return clustering.run_cluster(st, df, method, name)

def run_subspace_cluster(st, df, method, name):
    return subspace_clustering.run_subspace_cluster(st, df, method, name)

def run_imbalanced(st, df, name):
    return imbalanced.run_imbalanced(st, df, name)

def run_time_series(st, df, name):
    return None

df_start = load_data()
df = df_start.copy()

page = st.selectbox("Choose your page", ["Exploration", "Feature Selection & Extraction",
                                         "Clustering", "Subspace Clustering", "Imbalanced Learning",
                                         "Time Series Classification"])
if page == "Exploration":
    explore(st, df)

elif page == "Feature Selection & Extraction":
    df = feature_selection(st, df)
    run_dr(st, df, 'default')
    st.write("Let's do the same - only for just males")
    run_dr(st, df.query("Gender == 1"), 'male')

elif page == "Clustering":
    st.write("Let's try clustering with the full dataset")
    df = df_start.copy()
    run_cluster(st, df, 'DBSCAN', 'full')
    run_cluster(st, df, 'k-means', 'full')
    st.write("Let's try clustering after feature selection")
    df = dimensionality_reduction.feature_selection(st, df)
    run_cluster(st, df, 'DBSCAN', 'selected')
    run_cluster(st, df, 'k-means', 'selected')
    st.write("Let's try clustering after imputation")
    df = pd.read_pickle('data/imputed_linear_default.pkl')
    run_cluster(st, df, 'DBSCAN', 'imputed')
    run_cluster(st, df, 'k-means', 'imputed')

elif page == "Subspace Clustering":
    st.write("Let's try subspace clustering with the full dataset")
    df = df_start.copy()
    run_subspace_cluster(st, df, 'ensc', 'full')
    # df = df_start.copy()
    # run_subspace_cluster(st, df, 'Spectral', 'full')
    st.write("Trying after imputation")
    df = pd.read_pickle('data/imputed_linear_default.pkl')
    run_subspace_cluster(st, df, 'ensc', 'imputed')
    # df = pd.read_pickle('imputed_linear_default.pkl')
    # run_subspace_cluster(st, df, 'Spectral', 'imputed')

elif page == "Imbalanced Learning":
    df = dimensionality_reduction.feature_selection(st, load_data())
    run_imbalanced(st, df, 'selected')

elif page == "Time Series Classification":
    print("See other file")


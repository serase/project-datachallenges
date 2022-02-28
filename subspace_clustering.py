import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import pandas as pd
from cluster.selfrepresentation import ElasticNetSubspaceClustering
from sklearn.cluster import SpectralBiclustering
import dimensionality_reduction

def subspace_cluster(st, df, method, name):
    if name == 'full' or name == 'selected':
        for i in df.columns[df.isnull().any(axis=0)]: # https://stackoverflow.com/questions/37057187/pandas-interpolate-within-a-groupby
            df[i].fillna(df[i].mean(), inplace=True)
        df = df.head(5000)
        df, sc = dimensionality_reduction.scale(st, df)
    else:
        df = df.head(5000)
    st.write(f"Running {method}")
    fname = f"data/subspace_cluster_{method}_{name}.pkl"
    if os.path.isfile(fname):
        df = pd.read_pickle(fname)
    else:
        st.write('We use the Silhouette Score')
        sh_explanation = """
        1: Means clusters are well apart from each other and clearly distinguished.
        0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
        -1: Means clusters are assigned in the wrong way.
    
        Silhouette Score = (b-a)/max(a,b)
        where
        a= average intra-cluster distance i.e the average distance between each point within a cluster.
        b= average inter-cluster distance i.e the average distance between all clusters."""
        # st.write(sh_explanation)
        image_name = f"images/subspace_cluster_{name}_{method}_parameter.png"
        comb_file = f"data/subspace_cluster_{name}_{method}_best.txt"
        try:
            df.drop('patient', axis=1, inplace=True)
        except Exception:
            print('patient not found in columns')

        if os.path.isfile(image_name):
            st.image(image_name)
            with open(comb_file, 'r') as f:
                best_comb = f.readline()
        else:
            # Create empty lists
            S = [] # this is to store Silhouette scores
            comb = [] #` this is to store combinations of epsilon / min_samples

            if method == 'ensc':
                plt.xlabel('Clusters | Gamma')
                # Define ranges to explore
                cluster_range=range(3,6)
                gamma_range=range(3,6)

                for n in cluster_range:
                    for g in gamma_range:
                        model = ElasticNetSubspaceClustering(n_clusters=n,
                                                             algorithm='lasso_lars',
                                                             gamma=g).fit(df.drop(['SepsisLabel'], axis=1))
                        # Fit the model
                        clm = model.fit(df.drop(['SepsisLabel'], axis=1))
                        # Calculate Silhoutte Score and append to a list
                        S.append(metrics.silhouette_score(df.drop(['SepsisLabel'], axis=1), clm.labels_, metric='euclidean'))
                        comb.append(str(n)+"|"+str(g)) # axis values for the graph
                plt.figure(figsize=(16,8), dpi=300)
                plt.plot(comb, S, 'bo-', color='black')
                plt.xlabel('Clusters | Gamma')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score based on different combination of Hyperparameters')
                plt.savefig(image_name)
                st.pyplot(plt)

            elif method == 'Spectral':
                plt.xlabel('N_clusters')
                cluster_range = range(2,12)
                df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
                for k in cluster_range:
                    model = SpectralBiclustering(n_clusters=k, random_state=0)
                    clm = model.fit(df.drop(['SepsisLabel'], axis=1))
                    S.append(metrics.silhouette_score(df.drop(['SepsisLabel'], axis=1), clm.row_labels_, metric='euclidean'))
                    comb.append(k)
                plt.figure(figsize=(16,8), dpi=300)
                plt.plot(comb, S, 'bo-', color='black')
                plt.xlabel('Clusters')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score based on different combination of Hyperparameters')
                plt.savefig(image_name)
                st.pyplot(plt)
            max_index = S.index(max(S))
            best_comb = comb[max_index]
            with open(comb_file, 'w') as f:
                f.write(best_comb)
        st.write(f"The best performing model: {best_comb}")
        if method == 'ensc':
            model = ElasticNetSubspaceClustering(n_clusters=int(best_comb.split("|")[0]),
                                                 algorithm='lasso_lars',
                                                 gamma=int(best_comb.split("|")[1])).fit(df.drop(['SepsisLabel'],
                                                                                                      axis=1))
            result = model.fit(df.drop(['SepsisLabel'], axis=1))
            df['cluster'] = result.labels_
        if method == 'Spectral':
            model = SpectralBiclustering(n_clusters=int(best_comb), random_state=0)
            result = model.fit(df.drop(['SepsisLabel'], axis=1))
            df['cluster'] = result.row_labels_
        df = df.sort_values(by=['SepsisLabel'])
        df.to_pickle(fname)
    return df

def analyze_cluster(st, df):
    st.write("Do we see a correlation with the label to find out if there are any clusters to be found")
    fig = plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(df.corr().abs(), dtype=np.bool))
    corr = df.corr().abs()
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Correlation Heatmap')
    st.pyplot(fig)
    highest_corr = list(corr['cluster'].nlargest(4).index)
    highest_corr.remove('cluster')
    st.write(f"The 3 features who are correlating the best with cluster are: {(',').join(highest_corr)}")

    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x=df[highest_corr[0]], y=df[highest_corr[1]], z=df[highest_corr[2]],
                        opacity=1, color=df['cluster'].astype(str),
                        color_discrete_sequence=['black']+px.colors.qualitative.Plotly,
                        hover_data=df.columns,
                        width=900, height=900
                        )
    st.plotly_chart(fig)

def run_subspace_cluster(st, input_df, method, name):
    df = input_df.copy()
    df = subspace_cluster(st, df, method=method, name=name)
    analyze_cluster(st, df)
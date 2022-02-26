import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import pandas as pd
import dimensionality_reduction

def cluster(st, df, method, name):
    if name == 'full' or name == 'selected':
        for i in df.columns[df.isnull().any(axis=0)]: # https://stackoverflow.com/questions/37057187/pandas-interpolate-within-a-groupby
            df[i].fillna(df[i].mean(), inplace=True)
        df = df.head(5000)
        df, sc = dimensionality_reduction.scale(st, df)
    else:
        df = df.head(5000)
    st.write(f"Running {method}")
    fname = f"cluster_{method}_{name}.pkl"
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

        image_name = f"images/cluster_{name}_{method}_parameter.png"

        if os.path.isfile(image_name):
            st.image(image_name)
            if method == 'DBSCAN':
                best_comb = '29|4'
            elif method == 'k-means':
                best_comb = 3
        else:
            # Create empty lists
            S = [] # this is to store Silhouette scores
            comb = [] #` this is to store combinations of epsilon / min_samples
            if method == 'DBSCAN':
                # Define ranges to explore
                eps_range = range(12,30) # note, we will scale this down by 100 as we want to explore 0.06 - 0.11 range
                minpts_range = range(3,5)

                for k in eps_range:
                    for j in minpts_range:
                        # Set the model and its parameters
                        model = DBSCAN(eps=k/100, min_samples=j, n_jobs=-1)
                        # Fit the model
                        clm = model.fit(df.drop(['SepsisLabel'], axis=1).values)
                        S.append(metrics.silhouette_score(df.drop(['SepsisLabel'], axis=1),
                                                          clm.labels_, metric='euclidean'))
                        comb.append(str(k)+"|"+str(j))
                plt.figure(figsize=(16,8), dpi=300)
                plt.plot(comb, S, 'bo-', color='black')
                plt.xlabel('Epsilon/100 | MinPts')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score based on different combination of Hyperparameters')
                plt.savefig(image_name)
                st.pyplot(plt)

            elif method == 'k-means':
                cluster_range = range(2,12)
                for k in cluster_range:
                    model = KMeans(init="random",
                                   n_clusters=k,
                                   n_init=10,
                                   max_iter=300,
                                   random_state=42
                                   )
                    clm = model.fit(df.drop(['SepsisLabel'], axis=1))
                    S.append(metrics.silhouette_score(df.drop(['SepsisLabel'], axis=1), clm.labels_, metric='euclidean'))
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
        st.write(f"The best performing model: {best_comb}")
        if method == 'DBSCAN':
            model = DBSCAN(eps=int(best_comb.split("|")[0])/100, min_samples=int(best_comb.split("|")[1]),
                           n_jobs=-1)
            result = model.fit(df.drop(['SepsisLabel'], axis=1))
        if method == 'k-means':
            model = KMeans(init="random", n_clusters=best_comb, n_init=10, max_iter=300, random_state=42)
            result = model.fit(df.drop(['SepsisLabel'], axis=1))
        df['cluster'] = result.labels_
        df = df.sort_values(by=['SepsisLabel'])
        df.to_pickle(fname)
    return df

def analyze_cluster(st, df):
    st.write("Do we see a correlation with the label to find out if there are any clusters to be found")
    fig = plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    corr = df.corr()
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Correlation Heatmap')
    st.pyplot(fig)
    highest_corr = list(corr['cluster'].nlargest(4).index)
    highest_corr.remove('cluster')
    st.write(f"The 3 features who are correlating the best with cluster are: {(',').join(highest_corr)}'")

    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x=df[highest_corr[0]], y=df[highest_corr[1]], z=df[highest_corr[2]],
                        opacity=1, color=df['cluster'].astype(str),
                        color_discrete_sequence=['black']+px.colors.qualitative.Plotly,
                        hover_data=df.columns,
                        width=900, height=900
                        )
    st.plotly_chart(fig)

def run_cluster(st, input_df, method, name):
    df = input_df.copy()
    df = cluster(st, df, method=method, name=name)
    analyze_cluster(st, df)
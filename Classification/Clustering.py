### CLUSTERING ###

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
 
def kmeans_clustering(df, target, n_clusters=(2,10), init='k-means++', metric='distortion', max_iter=100, random_seed=1234):
    X = df.drop(columns=target)
    kmeans = KElbowVisualizer(KMeans(random_state=random_seed), k=n_clusters, metric=metric, timings=False).fit(X)

    print('{} score for {:.0f} clusters is {:.2f}'.format(metric, kmeans.elbow_value_, kmeans.elbow_score_))
    df_output = df.copy()
    df_output['Cluster'] = pd.Categorical(KMeans(kmeans.elbow_value_, init=init, max_iter=max_iter).fit_predict(X))
    return df_output

def hierarchical_clustering():

    return

def GMM_clustering(df, target, n_clusters=(2,10), metric='bic', random_seed=1234):
    X = df.drop(columns=target)
    list_GMM, list_labels, list_bic = {}, {}, {}
    for k in range(*n_clusters):
        list_GMM[k] = GaussianMixture(n_components=k, random_state=random_seed).fit(X)
        list_labels[k] = list_GMM[k].predict(X)
        list_bic[k] = list_GMM[k].bic(X)
    k_optimal = min(list_bic, key=list_bic.get)

    print('{} score for {:.0f} clusters is {:.2f}'.format(metric, k_optimal, list_bic[k_optimal]))
    df_output = df.copy()
    df_output['Cluster'] = pd.Categorical(list_labels[k_optimal])
    return df_output

def SOM_clustering():

    return

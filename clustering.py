import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from metrics import scatterPlot


class clustering():

    def cluster_fit(train_x, test_x, train_y, test_y):
        
        # Addestra il classificatore basato su clustering
        X = np.concatenate((train_x, test_x), axis=0)
        y = np.concatenate((train_y, test_y), axis=0)
        
        # Applicare k-means
        n_clusters = 100  # Dal momento che ci sono 100 specie
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Valutare il clustering
        ari = adjusted_rand_score(y, clusters)
        silhouette_avg = silhouette_score(X, clusters)
        
        

        return ari, silhouette_avg, X, clusters
    
    
        
import numpy as np 
import faiss
from prettytable import PrettyTable
from rich import print
from sklearn.metrics.cluster import silhouette_score

def extract_candidate_features(image_embs, cluster_range):
    print(f"Running FAISS Spherical K-Means with {cluster_range} clusters")
    t = PrettyTable(["clusters", "silhouette"])

    best_config = {
        'n_clusters': -np.inf, 
        'silhouette_score': -np.inf, 
        'clusters': -np.inf, 
        'centroids': -np.inf
    }

    for n in cluster_range:
        kmeans = faiss.Kmeans(
            d=image_embs.shape[1], 
            k=n, 
            niter=20, 
            verbose=True, 
            spherical=True
        )
        kmeans.train(image_embs)
        _, clusters = kmeans.index.search(image_embs, 1)
        clusters = clusters.flatten()

        score = silhouette_score(image_embs, clusters, metric="cosine", sample_size=10000)
        if score > best_config['silhouette_score']:
            best_config = {
                'n_clusters': n, 
                'silhouette_score': score, 
                'clusters': clusters, 
                'centroids': kmeans.centroids,
            }
        t.add_row([n, np.round(score,3)])
    
    print(t)
    print(f"Selected {best_config['n_clusters']} clusters\n")

    return best_config['clusters'], best_config['centroids']

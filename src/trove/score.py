import numpy as np
from .utils import extract_candidate_features, discover_biases

def score(df, class_labels): 
    # TRoVe - Step 1: Extract candidate static features
    print(f"=> Starting TRoVe - Step 1: Extract candidate static features")
    cluster_range = np.arange(len(class_labels)*2, len(class_labels)*6, len(class_labels))
    clusters, medoids = extract_candidate_features(
        image_embs = np.concatenate(df['image_embs']),
        cluster_range = cluster_range
    )

    # TRoVe - Step 2: Discover error-inducing static feature biases
    print(f"=> Starting TRoVe - Step 2: Discover error-inducing static feature biases")
    out = discover_biases(
        df=df,
        class_labels=class_labels,
        clusters=clusters,
        medoids=medoids,
    )

    print(f"=> Completed execution")
    return out


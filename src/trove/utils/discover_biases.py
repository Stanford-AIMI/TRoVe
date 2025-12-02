import numpy as np
from collections import defaultdict

def score_error_contribution(
        image_idx, 
        sequence_pred, 
        sequence_label, 
        image_idx_to_sequence_idx, 
        epsilon=0.1,
        min_freq=0.15,
    ): 
    ec_scores = {}

    # Obtain all ground-truth labels for sequences with at least one
    # constituent image in cluster c
    all_labels_in = set([sequence_label[image_idx_to_sequence_idx[i]] for i in image_idx])

    for y in all_labels_in: 
        # Identify sequences with at least one constituent image in cluster c 
        # and ground-truth label y
        sequence_idx_in = sorted(set(
            [
                image_idx_to_sequence_idx[i] for i in image_idx
                if sequence_label[image_idx_to_sequence_idx[i]] == y
            ]
        ))

        # Identify sequences with no constituent images in cluster c 
        # and ground-truth label y
        sequence_idx_out = [
            i 
            for i in set(image_idx_to_sequence_idx.values())
            if (i not in sequence_idx_in) and (sequence_label[i] == y)
        ]

        if len(sequence_idx_in) == 0 or len(sequence_idx_out) == 0: continue
        assert (len(set(sequence_idx_out).intersection(set(sequence_idx_in))) == 0) 

        in_acc = (sequence_pred[sequence_idx_in] == sequence_label[sequence_idx_in]).mean()
        out_acc = (sequence_pred[sequence_idx_out] == sequence_label[sequence_idx_out]).mean()

        # Compute error contribution score
        ec = out_acc - in_acc

        # Exclude cases where the error contribution score is too small (below epsilon)
        if ec <= epsilon: continue

        # Exclude cases where feature c is rare, as this means that there are too
        # few sequences with c in order to reliably identify error patterns. Similarly, 
        # exclude cases where feature c is very common, as this means that there are too
        # few sequences without c in order to reliably identify error patterns.
        frequency_of_c = len(sequence_idx_in) / (len(sequence_idx_in) + len(sequence_idx_out))
        if frequency_of_c < min_freq or frequency_of_c > (1-min_freq): continue

        ec_scores[y] = ec

    return ec_scores

def score_static_bias( 
        image_idx,
        sequence_pred, 
        sequence_label, 
        image_idx_to_sequence_idx, 
        image_probs,
        class_labels,
    ):
    sb_scores = defaultdict(list)

    for i in image_idx:
        sequence_idx = image_idx_to_sequence_idx[i]
        pred, label = sequence_pred[sequence_idx], sequence_label[sequence_idx]

        # Compute image-level prediction confidence
        if pred != label:
            image_score_on_pred_class = image_probs[i][class_labels.index(pred)]
            sb_scores[label].append(image_score_on_pred_class)

    sb_scores = {k: np.mean(v) for k, v in sb_scores.items()}

    # Exclude class labels where the static bias score is less than or equal to random chance
    sb_scores = {k: v for k,v in sb_scores.items() if v > (1/len(class_labels))}

    return sb_scores


def discover_biases(
    df,
    class_labels,
    clusters,
    medoids,
): 
    sequence_pred = df['sequence_probs'].apply(lambda x: class_labels[x.argmax()]).values
    sequence_label = df["label"].values

    image_idx_to_sequence_idx = dict(zip(
        np.arange(df['image_probs'].explode().shape[0]), 
        df['image_probs'].explode().index            
    ))
    
    results = {
        l: {"cluster_idx": [], "ec_score": [], "sb_score": [], "trove_score": []} 
        for l in class_labels
    }

    for c in sorted(set(clusters)):
        image_idx = np.where(clusters == c)[0]

        ec_scores = score_error_contribution(
            image_idx = image_idx, 
            sequence_pred = sequence_pred, 
            sequence_label = sequence_label, 
            image_idx_to_sequence_idx = image_idx_to_sequence_idx,
        )

        sb_scores = score_static_bias(
            image_idx,
            sequence_pred, 
            sequence_label, 
            image_idx_to_sequence_idx, 
            np.concatenate(df['image_probs']),
            class_labels,
        )

        for y in ec_scores:
            if y not in sb_scores: continue
            results[y]["cluster_idx"].append(c)
            results[y]["ec_score"].append(ec_scores[y])
            results[y]["sb_score"].append(sb_scores[y])
            results[y]["trove_score"].append(ec_scores[y] + sb_scores[y])


    ranked_images = {y: [] for y in results}
    for y in results:
        ranked_features = np.argsort(results[y]['trove_score'])[::-1]
        for f in ranked_features:
            
            c = results[y]['cluster_idx'][f]
            medoid = medoids[c].reshape(1, -1)
            image_idx = np.where(clusters == c)[0]
            image_emb = np.concatenate(df['image_embs'])[image_idx]
            ranked_image_idx = image_idx[np.argsort((image_emb @ medoid.T).flatten())[::-1]]

            ranked_images[y].append(
                {k: results[y][k][f] for k in results[y]} | {'ranked_image_idx': ranked_image_idx}
            )

    return ranked_images



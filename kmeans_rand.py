import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

def kmeans_rand(n_clusters, encoded_vec, names, dir_res_model):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(encoded_vec)
    # print(kmeans.labels_)

    # Clustering performance evaluation
    unique_names, indexed_names = np.unique(names, return_inverse=True)
    # print(unique_names, indexed_names)

    labels_true = indexed_names
    labels_pred = kmeans.labels_
    rand_index = metrics.rand_score(labels_true, labels_pred)
    print("Rand index:", rand_index)
    fn = os.path.join(dir_res_model, "rand_index.txt")
    with open(fn, "w") as text_file:
        text_file.write("Rand index: ")
        text_file.write("{0:.2f}".format(round(rand_index, 2)))

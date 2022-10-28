import os
import time
import numpy as np
from matplotlib import pyplot as plt

def kNN_classification_flow(encoded_vec_2d, names, title):

    labels = [] # data and labels
    data = np.zeros((0, encoded_vec_2d.shape[1])) # (0,2) after projections

    n_neigh = 17
    #print("kNN, n_neigh =", n_neigh)
    #print("decision is based on", int(n_neigh/2)+1, "votes")

    for i in range(encoded_vec_2d.shape[0]):
        if (names[i].find(" l") != -1):
            data = np.append(data, encoded_vec_2d[i:i+1], axis=0)
            labels.append(0) # laminar
            #print(names[i], 0)
        else:
            if (names[i].find(" t") != -1):
                data = np.append(data, encoded_vec_2d[i:i+1], axis=0)
                labels.append(1) # turbulent
                #print(names[i], 1)
            # else: # without labels
            #     data = np.append(data, encoded_vec_2d[i:i+1], axis=0)
            #     labels.append(2)
            #     #print(names[i], 2)

    # print(len(data), len(labels))
    #print(data[0:5])
    #print(encoded_vec_2d[0:5])

    #print(data.shape)

    # draw a scatterplot with colored labels
    #fig, ax = plt.subplots()
    #plt.scatter(data[:, 0], data[:, 1], c=labels, label=np.unique(labels))
    #plt.suptitle(title)
    #plt.show(block=False)
    #plt.close()

    # from sklearn.neighbors import KNeighborsClassifier
    # neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')

    # neigh.fit(data[0:50], labels[0:50])
    # predicted_labels = neigh.predict(data)
    # #print("prediction", predicted_labels)
    # #print("ground truth", labels)

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neigh+1) # +1 is self
    neigh.fit(data)

    pred_kneighbors = neigh.kneighbors(data, return_distance=False)
    #print(pred_kneighbors)
    #print(pred_kneighbors.shape)

    predicted_labels = []
    vote = 0

    for i in range(pred_kneighbors.shape[0]):
        vote = 0
        for j in range(pred_kneighbors.shape[1] - 1):
            #print(pred_kneighbors[i,j+1])
            index = pred_kneighbors[i,j+1] # ignore self
            vote += labels[index] # 1 for turb
            
        if (vote >= int(n_neigh/2) + 1): # out of n_neigh
            predicted_labels.append(1) # turb
        else:
            predicted_labels.append(0) # laminar

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(labels, predicted_labels)
    #print("Accuracy:", acc)

    return acc, labels

    # correct = 0.
    # for i in range(len(labels)):
    #     if (predicted_labels[i] == labels[i]):
    #         correct += 1

    # accuracy = correct / len(labels)
    # print(accuracy)

    #return labels

def kNN_classification_droplet(encoded_vec_2d, names, title):

    #data = np.zeros((0, encoded_vec_2d.shape[1])) # (0,2) after projections

    n_neigh = 11
    #print("kNN, n_neigh =", n_neigh)
    #print("decision is based on", int(n_neigh/2)+1, "votes")

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neigh+1) # +1 is self
    neigh.fit(encoded_vec_2d)

    pred_kneighbors = neigh.kneighbors(encoded_vec_2d, return_distance=False)
    #print(pred_kneighbors)
    print(pred_kneighbors.shape)

    predicted_labels = []
    votes = [] # add labels here

    for i in range(pred_kneighbors.shape[0]):
        votes = []
        for j in range(pred_kneighbors.shape[1] - 1):
            #print(pred_kneighbors[i,j+1])
            index = pred_kneighbors[i,j+1] # ignore self
            votes.append(names[index]) # current vote

        from collections import Counter
        c = Counter(votes)
        #print("most common set:", c.most_common())
        value, count = c.most_common()[0]
        #print("most common:", value)
        predicted_labels.append(value)

    #print(names)
    #print(predicted_labels)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(names, predicted_labels)
    #print("Accuracy:", acc)

    return acc

import os
import time
import numpy as np
from matplotlib import pyplot as plt

def variance_flow(encoded_vec_2d, names):
    # Get distances from cluster class centers

    # vecs must be scaled to [0,1] to be on the same scale
    #print("Min:", encoded_vec_2d.min())
    #print("Max:", encoded_vec_2d.max())
    # scale to range [0, 1]
    encoded_vec_2d = (encoded_vec_2d - encoded_vec_2d.min()) / (encoded_vec_2d.max() - encoded_vec_2d.min())
    #print("After scaling to [0,1]")
    #print("Min:", encoded_vec_2d.min())
    #print("Max:", encoded_vec_2d.max())

    turb_vec, lam_vec = [], [] # turbilent and laminar vectors

    for i in range(encoded_vec_2d.shape[0]):
        if names[i].find(" l") == -1: # turbulent
            turb_vec.append(encoded_vec_2d[i])
            #print(names[i])
        else:
            lam_vec.append(encoded_vec_2d[i])
            #print(names[i])
        
    #print("turb: ", len(turb_vec))
    #print("lam: ", len(lam_vec))

    lam_center = np.mean(lam_vec, axis=0)
    turb_center = np.mean(turb_vec, axis=0)

    center = []
    center.append(lam_center)
    center.append(turb_center)
    #print(center)

    center_l = center[0]
    #print("tsne laminar center: ", center_l)
    center_t = center[1]
    #print("tsne turbulent center: ", center_t)

    # for each point in the proj measure and accumulate corresponding distance
    dist_to_laminar, dist_to_turbulent = [], []
    for i in range(encoded_vec_2d.shape[0]):
        if names[i].find(" l") == -1: # turbulent
            dist_to_turbulent.append(np.linalg.norm(center_t - encoded_vec_2d[i]))
        else:
            dist_to_laminar.append(np.linalg.norm(center_l - encoded_vec_2d[i]))

    dist_to_turbulent = np.mean(dist_to_turbulent, axis=0)
    dist_to_laminar = np.mean(dist_to_laminar, axis=0)

    #return dist_to_laminar, dist_to_turbulent

    dist_to_centers_mean = (dist_to_turbulent + dist_to_laminar)*0.5
    #print("dist_to_centers_mean:", dist_to_centers_mean)

    return dist_to_centers_mean

def variance_droplet(encoded_vec_2d, names):
    # Get distances from cluster class centers

    # vecs must be scaled to [0,1] to be on the same scale
    # print("Min:", encoded_vec_2d.min())
    # print("Max:", encoded_vec_2d.max())
    # scale to range [0, 1]
    encoded_vec_2d = (encoded_vec_2d - encoded_vec_2d.min()) / (encoded_vec_2d.max() - encoded_vec_2d.min())
    # print("After scaling to [0,1]")
    # print("Min:", encoded_vec_2d.min())
    # print("Max:", encoded_vec_2d.max())

    # get unique set of classes
    # loop for classes: append vectors to clusters; np.mean for centers
    # for loop: measure dist from each vec to centers

    unique_names = np.unique(names)
    print(unique_names)

    # get the centers for each cluster
    centers = []
    for cluster_name in unique_names:
        dist_to_center = []
        for i in range(encoded_vec_2d.shape[0]):
            if names[i] == cluster_name:
                dist_to_center.append(encoded_vec_2d[i])
        
        centers.append(np.mean(dist_to_center, axis=0))
    # print("centers:", centers)
    centers = zip(unique_names, centers)
    centers = dict(centers)
    # print("(dict) dist and centers:", centers)
    # print(centers["drop"])

    # for each point in the proj measure and accumulate distance to its cluster
    dist_to_centers = []
    for cluster_name in unique_names:
        dist_to_center = []
        for i in range(encoded_vec_2d.shape[0]):
            if names[i] == cluster_name:
                dist_to_center.append(np.linalg.norm(centers[cluster_name] - encoded_vec_2d[i]))
        
        #print(dist_to_center)
        dist_to_centers.append(np.mean(dist_to_center, axis=0))

    print("dist_to_centers:", dist_to_centers)

    dist_to_centers_mean = np.mean(dist_to_centers, axis=0)
    #print("dist_to_centers_mean:", dist_to_centers_mean)

    return dist_to_centers_mean

def variance_mnist(encoded_vec_2d, names):
    # Get distances from cluster class centers

    # vecs must be scaled to [0,1] to be on the same scale
    #print("Min:", encoded_vec_2d.min())
    #print("Max:", encoded_vec_2d.max())
    # scale to range [0, 1]
    encoded_vec_2d = (encoded_vec_2d - encoded_vec_2d.min()) / (encoded_vec_2d.max() - encoded_vec_2d.min())
    #print("After scaling to [0,1]")
    #print("Min:", encoded_vec_2d.min())
    #print("Max:", encoded_vec_2d.max())

    # get unique set of classes
    # loop for classes: append vectors to clusters; np.mean for centers
    # for loop: measure dist from each vec to centers

    unique_names = np.unique(names)

    # get the centers for each cluster
    centers = []
    for cluster_name in unique_names:
        dist_to_center = []
        for i in range(encoded_vec_2d.shape[0]):
            if names[i] == cluster_name:
                dist_to_center.append(encoded_vec_2d[i])
        
        centers.append(np.mean(dist_to_center, axis=0))
    #print("centers:", centers)
    centers = zip(unique_names, centers)
    centers = dict(centers)
    #print("(dict) dist and centers:", centers)
    #print(centers["drop"])
    print("centers:", centers)

    # for each point in the proj measure and accumulate distance to its cluster
    dist_to_centers = []
    for cluster_name in unique_names:
        dist_to_center = []
        for i in range(encoded_vec_2d.shape[0]):
            if names[i] == cluster_name:
                dist_to_center.append(np.linalg.norm(centers[cluster_name] - encoded_vec_2d[i]))
        
        #print(dist_to_center)
        dist_to_centers.append(np.mean(dist_to_center, axis=0))

    #print("dist_to_centers:", dist_to_centers)

    dist_to_centers_mean = np.mean(dist_to_centers, axis=0)
    #print("dist_to_centers_mean:", dist_to_centers_mean)

    return dist_to_centers_mean

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

    n_neigh = 17 # 17 11 7
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

def kNN_classification_mnist(encoded_vec_2d, names, title):

    #data = np.zeros((0, encoded_vec_2d.shape[1])) # (0,2) after projections

    n_neigh = 17 # 11
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

def kNN_fraction_flow(encoded_vec_2d, names, title):
    #print("Neighborhood hit flow")

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

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neigh+1) # +1 is self
    neigh.fit(data)

    pred_kneighbors = neigh.kneighbors(data, return_distance=False)
    #print(pred_kneighbors)
    #print(pred_kneighbors.shape)

    fraction = 0.
    for i in range(pred_kneighbors.shape[0]):
        votes = []
        for j in range(pred_kneighbors.shape[1] - 1):
            #print(pred_kneighbors[i,j+1])
            index = pred_kneighbors[i,j+1] # ignore self
            votes.append(labels[index]) # current vote

        from collections import Counter
        c = Counter(votes)
        #print("most common set:", c.most_common())
        #print("true label:", labels[i])
        
        for label_count in c.most_common():
            value, count = label_count
            #print(value)
            if (value == labels[i]): # count the fraction
                fraction += count/n_neigh
                #print(count/n_neigh)
                break

    fraction /= len(labels) # or len(names)

    return fraction

def kNN_fraction_droplet(encoded_vec_2d, names, title):
    #print("Neighborhood hit droplet")

    #data = np.zeros((0, encoded_vec_2d.shape[1])) # (0,2) after projections

    n_neigh = 17 # 11
    #print("kNN, n_neigh =", n_neigh)
    #print("decision is based on", int(n_neigh/2)+1, "votes")

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neigh+1) # +1 is self
    neigh.fit(encoded_vec_2d)

    pred_kneighbors = neigh.kneighbors(encoded_vec_2d, return_distance=False)
    #print(pred_kneighbors)
    print(pred_kneighbors.shape)

    votes = [] # add labels here

    fraction = 0.
    for i in range(pred_kneighbors.shape[0]):
        votes = []
        for j in range(pred_kneighbors.shape[1] - 1):
            #print(pred_kneighbors[i,j+1])
            index = pred_kneighbors[i,j+1] # ignore self
            votes.append(names[index]) # current vote

        from collections import Counter
        c = Counter(votes)
        #print("most common set:", c.most_common())
        #print("true label:", names[i])
        
        for label_count in c.most_common():
            value, count = label_count
            #print(value)
            if (value == names[i]): # count the fraction
                fraction += count/n_neigh
                #print(count/n_neigh)
                break

    fraction /= len(names)

    return fraction

def kNN_fraction_mnist(encoded_vec_2d, names, title):
    #print("Neighborhood hit droplet")

    #data = np.zeros((0, encoded_vec_2d.shape[1])) # (0,2) after projections

    n_neigh = 17 # 11
    #print("kNN, n_neigh =", n_neigh)
    #print("decision is based on", int(n_neigh/2)+1, "votes")

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neigh+1) # +1 is self
    neigh.fit(encoded_vec_2d)

    pred_kneighbors = neigh.kneighbors(encoded_vec_2d, return_distance=False)
    #print(pred_kneighbors)
    print(pred_kneighbors.shape)

    votes = [] # add labels here

    fraction = 0.
    for i in range(pred_kneighbors.shape[0]):
        votes = []
        for j in range(pred_kneighbors.shape[1] - 1):
            #print(pred_kneighbors[i,j+1])
            index = pred_kneighbors[i,j+1] # ignore self
            votes.append(names[index]) # current vote

        from collections import Counter
        c = Counter(votes)
        #print("most common set:", c.most_common())
        #print("true label:", names[i])
        
        for label_count in c.most_common():
            value, count = label_count
            #print(value)
            if (value == names[i]): # count the fraction
                fraction += count/n_neigh
                #print(count/n_neigh)
                break

    fraction /= len(names)

    return fraction
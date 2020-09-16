
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn

from pca_projection import pca_projection
from tsne_projection import tsne_projection
from umap_projection import umap_projection

from visualization import visualize_keract, visualize_keras
from latent_sp_interpolation import interpolate, find_interpolatable

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
import time
from progress.bar import Bar
import pickle
import json
from sklearn.model_selection import train_test_split

from keras.layers import Activation, Input, Dense, Conv2D, Conv2DTranspose
from keras.layers import Flatten, Reshape, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers

def cropping_output(decoded, input_shape):

    dec_shape = K.int_shape(decoded)
    print('Before cropping:', dec_shape) 
    cropWidth, cropHeight = dec_shape[1] - input_shape[1], dec_shape[2] - input_shape[2]
    print(cropWidth, cropHeight)
    cropLeft, cropTop = int(cropWidth / 2), int(cropHeight / 2)
    cropRight, cropBot = cropWidth - cropLeft, cropHeight - cropTop
    # cropping the output
    decoded = Cropping2D(cropping=((cropLeft, cropRight), (cropTop, cropBot)))(decoded)
    dec_shape = K.int_shape(decoded)
    print('After cropping:', dec_shape) 

    return decoded

def load_labelled_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("droplet_labelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    print("Loading names from pickle dump...")
    pkl_file = open("droplet_labelled_names.pkl", 'rb')
    names = []
    names = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    print(data.shape)
    print(len(names))
    if data.shape[0] != len(names):
        input("!!! Inconstintency in data and names !!!")
    # data = data[0:3000]
    # names = names[0:3000]
    from sklearn.utils import shuffle
    data, names = shuffle(data, names, random_state=0)
    print("Shuffled test set")
    print(data.shape)
    print(len(names))
    data = data[0:600]
    names = names[0:600]

    # #data = data[:600,...]
    # test_idx = np.random.randint(data.shape[0], size=600)
    # #print(test_idx)
    # data = data[test_idx,]
    # #names = names[test_idx,]
    # names_new = []
    # for idx in test_idx:
    #     names_new.append(names[idx])
    # names = names_new
    # print("Randomized sampling from test data")
    # print(data.shape)
    # print(len(names))
    # print(names[0])
    # print(np.unique(names))

    # for _ in np.unique(names):
    #     print(_)

    return data, names

def load_unlabelled_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("droplet_unlabelled_data.pkl", 'rb')
    #pkl_file = open("droplet-part.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    data = data[0:12000]
    print(data.shape)

    return data

def brightness_normalization(data):

    for i in range(data.shape[0]):
        data[i] = (data[i] - data[i].mean()) / data[i].std()

    # for i in range(10):
    #     print(data[i].mean(), data[i].std())

    print("All samples were normalized individually!")
    return data

def load_preprocess():

    data_test, names = load_labelled_data() # for test only, 
    data_train = load_unlabelled_data() # for train only, 

    data = np.zeros((data_test.shape[0]+data_train.shape[0], 160, 224, 1))
    data[:data_train.shape[0],] = data_train
    data[data_train.shape[0]:,] = data_test

    dataset = "droplet" # for pca t-sne umap preprocessing vis

    visualize_data = False
    data, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    print(data.shape)

    data = brightness_normalization(data) # test

    # cropping:
    crop_left = int(data.shape[2]*0.1) # start from left
    crop_right = int(data.shape[2]*0.85) # end at right
    data = data[:,:,crop_left:crop_right,:]
    print("data cropped: ", data.shape)

    # visualize_data = False
    # data, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    # print(data.shape)
    data_test_vis = data_test # unnormalized, uncropped

    data_train = data[:data_train.shape[0],]
    data_test = data[data_train.shape[0]:,]
    print("test:", data_test.shape)

    data_train, data_val = train_test_split(data_train, test_size=0.2, random_state=1)
    print('train & val', data_train.shape, data_val.shape)

    x_train, x_test, x_val = data_train, data_test, data_val

    return x_train, x_test, x_val, names, data_mean, data_std, data_test_vis

def main():

    x_train, x_test, x_val, names, data_mean, data_std, data_test_vis = load_preprocess()
    dataset = "droplet" 

    # network parameters
    #input_shape = (image_size, image_size, 1)
    batch_size = 16
    kernel_size = 3
    filters = 64
    generic = False
    dense_dim = 1024
    latent_dim = 256
    epochs = 0 # 500
    conv_layers = 4
    stride = 2
    latent_vector = True
    project = True
    regularization = False
    dropout_sparcity = False
    denoising = False

    # skript: combine 5 experiment: m arc, loss val, all results
    # everything must be fully automated!

    model_names = {"baseline_norm_crop_relu.h5"}

    model_names = {"baseline_norm_crop_1.h5", "baseline_norm_crop_2.h5", \
    "baseline_norm_crop_3.h5", "baseline_norm_crop_4.h5", "baseline_norm_crop_5.h5"}# baseline cropped norm
    for model_name in model_names:
        print("model_name:", model_name)

        dir_res = "Results/Baseline"
        model = model_name[:-5]
        dir_res = os.path.join(dir_res, model)
        print("Saved here:", dir_res)
        model = model_name[:-3]
        dir_res_model = os.path.join(dir_res, model)
        os.makedirs(dir_res_model, exist_ok=True)

        # print("test_data, encoded_vec, decoded_imgs")
        # print('normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
        # print('normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())
        # # un-normalize all using data_mean, data_std
        # test_data_vis = test_data_vis * data_std + data_mean
        # #encoded_vec = encoded_vec * data_std + data_mean
        # decoded_imgs = decoded_imgs * data_std + data_mean
        # print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
        # print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

        test_data_vis = x_test # baseline
        # print(test_data_vis.min(), test_data_vis.max())
        # test_data_vis = test_data_vis * data_std + data_mean
        # print(test_data_vis.min(), test_data_vis.max())

        dataset = "droplet" # for pca t-sne vis
        title = 'Raw data ' # for baseline subtitle

        encoded_vec = np.zeros((300, 128))
        encoded_vec = 0

        if (project == True):
            # project using PCA (then t-sne) and visualize the scatterplot
            #print("PCA projection")
            #pca_projection(encoded_vec, test_data_vis, latent_vector, title, dataset, names)

            # project using t-sne and visualize the scatterplot
            print("t-SNE projection")
            #title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
            title_tsne = title + '-> t-SNE scatterplot, perp='
            #tsne_projection(encoded_vec, test_data_vis, latent_vector, title_tsne, dataset, names, perp=20)
            tsne_projection(encoded_vec, test_data_vis, latent_vector, title_tsne, dir_res_model, dataset, names, perp=30)
            #tsne_projection(encoded_vec, test_data_vis, latent_vector, title_tsne, dataset, names, perp=40)

            # project using UMAP and visualize the scatterplot
            print("UMAP projection")
            #title_umap = title + 'Latent -> UMAP scatterplot'
            title_umap = title + '-> UMAP scatterplot'
            umap_projection(encoded_vec, test_data_vis, latent_vector, title_umap, dir_res_model, dataset, names)


if __name__ == '__main__':
    main()


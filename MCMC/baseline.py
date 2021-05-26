
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils import model_directories, models_metrics_stability_mcmc
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
    pkl_file = open("mcmc_labelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    print("Loading names from pickle dump...")
    pkl_file = open("mcmc_labelled_names.pkl", 'rb')
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
    # data = data[0:1000]
    # names = names[0:1000]

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
    pkl_file = open("mcmc.pkl", 'rb')
    #pkl_file = open("droplet-part.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    data = data[0:12000] # 12000
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

    data = np.zeros((data_test.shape[0]+data_train.shape[0], 50, 50))
    data[:data_train.shape[0],] = data_train
    data[data_train.shape[0]:,] = data_test

    dataset = "mcmc" # for pca t-sne umap preprocessing vis

    visualize_data = False
    data, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    print(data.shape)

    data = brightness_normalization(data) # test

    # visualize_data = False
    # data, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    # print(data.shape)
    data_test_vis = data_test # unnormalized, uncropped

    data_train = data[:data_train.shape[0],]
    data_test = data[data_train.shape[0]:,]
    print("test:", data_test.shape)

    data_train, data_val = train_test_split(data_train, test_size=0.2, random_state=1)
    print('train & val', data_train.shape, data_val.shape)

    data_train = np.expand_dims(data_train, axis=-1)
    data_test = np.expand_dims(data_test, axis=-1)
    data_val = np.expand_dims(data_val, axis=-1)

    x_train, x_test, x_val = data_train, data_test, data_val

    return x_train, x_test, x_val, names, data_mean, data_std, data_test_vis

def main():

    x_train, x_test, x_val, names, data_mean, data_std, data_test_vis = load_preprocess()
    dataset = "mcmc" 

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

    # model_names = {"baseline_norm_1.h5", "baseline_norm_2.h5", "baseline_norm_3.h5", "baseline_norm_4.h5", "baseline_norm_5.h5"}
    mod_nam = {"baseline_norm"}

    # metrics stability add-on
    model_names = models_metrics_stability_mcmc(mod_nam)

    # model_names_all = []
    # for m_n in mod_nam:
    #     for i in range(5):    
    #         m_n_index = m_n + "_" + str(i+1) + ".h5"
    #         model_names_all.append(m_n_index)

    # model_names = model_names_all
    # print(model_names)

    for model_name in model_names:
        print("model_name:", model_name)

        dir_res = "Results/Baseline"
        dir_res_model = model_directories(dir_res, model_name)
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

        # metrics stability add-on
        step = 250
        for lab in reversed(range(500,2500+step, step)):
            # print(lab)
            to_remove = "_" + str(lab)
            if to_remove in model_name:
                x_test_ = x_test[:lab,...] # #labels to consider
                names_ = names[:lab] # #labels to consider
                print("Labels considered:", x_test_.shape[0])
                model_name = model_name.replace(to_remove, '')
        print(model_name)
        ###

        test_data_vis = x_test_ # baseline
        train_data = x_train[0:8000]
        # print(test_data_vis.min(), test_data_vis.max())
        # test_data_vis = test_data_vis * data_std + data_mean
        # print(test_data_vis.min(), test_data_vis.max())
        train_test_data = np.concatenate((train_data, test_data_vis), axis=0)
        # test_data_vis = train_test_data # full
        encoded_vec_train = 0
        encoded_vec_train_test = 0
        train_data = 0
        # train_test_data = 0

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
            # tsne_projection(encoded_vec, test_data_vis, latent_vector, title_tsne, dir_res_model, dataset, names, perp=30)
            #tsne_projection(encoded_vec, test_data_vis, latent_vector, title_tsne, dataset, names, perp=40)

            # project using UMAP and visualize the scatterplot
            print("UMAP projection")
            #title_umap = title + 'Latent -> UMAP scatterplot'
            title_umap = title + '-> UMAP scatterplot'
            #umap_projection(encoded_vec, test_data_vis, latent_vector, title_umap, dir_res_model, dataset, names)
            umap_projection(encoded_vec, encoded_vec_train, encoded_vec_train_test, test_data_vis, train_data, train_test_data, latent_vector, title_umap, dir_res_model, dataset, names_)



if __name__ == '__main__':
    main()


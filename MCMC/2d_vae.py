"""
27.02
generate smth new - sample random vector from latent space and see +
beta-VAE +-

22.04
try to add Dropout -

such as interpolating between two samples +
sampling in the vicinity of a sample -
exploring differences between a pair of samples applied to a third sample -
"""

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils import model_directories, models_metrics_stability, model_name_metrics_stability
from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn
from reparameterization_trick import sampling

from pca_projection import pca_projection
from tsne_projection import tsne_projection
from umap_projection import umap_projection
from kmeans_rand import kmeans_rand

from visualization import visualize_keract, visualize_keras
from latent_sp_interpolation import interpolate, find_interpolatable, latent_dim_traversal

import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
import time
from progress.bar import Bar
import pickle
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from keras.layers import Activation, Input, Dense, Conv2D, Conv2DTranspose
from keras.layers import Flatten, Reshape, Cropping2D, Lambda, Dropout
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

    print(data.shape)
    print(len(names))

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    if data.shape[0] != len(names):
        input("!!! Inconstintency in data and names !!!")

    from sklearn.utils import shuffle
    data, names = shuffle(data, names, random_state=0)
    print("Shuffled test set")
    print(data.shape)
    print(len(names))
    data = data[0:2500]
    names = names[0:2500]

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

    from sklearn.utils import shuffle
    data = shuffle(data, random_state=0)

    data = data[0:20000]
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

    # data = load_unlabelled_data()

    dataset = "mcmc" # for pca t-sne umap preprocessing vis

    visualize_data = False
    data, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    print(data.shape)

    data = brightness_normalization(data) # test

    # cropping:
    # crop_left = int(data.shape[2]*0.1) # start from left
    # crop_right = int(data.shape[2]*0.85) # end at right
    # data = data[:,:,crop_left:crop_right,:]
    # print("data cropped: ", data.shape)

    # data_test_vis = data_test # unnormalized, uncropped
    data_test_vis = data
    # data_mean = 0
    # data_std = 1
    # names = ""

    # data_train_size = 16000
    # data_train = data[:data_train_size,]
    # data_test = data[data_train_size:,]
    # print("test:", data_test.shape)

    data_train = data[:data_train.shape[0],]
    data_test = data[data_train.shape[0]:,]

    data_train, data_val = train_test_split(data_train, test_size=0.2, random_state=1)
    print('train & val', data_train.shape, data_val.shape)

    data_train = np.expand_dims(data_train, axis=-1)
    data_test = np.expand_dims(data_test, axis=-1)
    data_val = np.expand_dims(data_val, axis=-1)

    x_train, x_test, x_val = data_train, data_test, data_val

    return x_train, x_test, x_val, names, data_mean, data_std, data_test_vis
        
def main():

    dir_res = "Results/2D_VAE" # directory with all models

    # Load data and subsequently encoded vectors in 2D representation
    # for this save before x_test and encoded vec after tsne and umap
    load_data = False
    if load_data:
        # load test_data from pickle and later encoded_vec_2d
        fn = os.path.join(dir_res, "test_data.pkl")
        pkl_file = open(fn, 'rb')
        data_test = pickle.load(pkl_file)
        print("Test data were loaded from pickle")
        pkl_file.close

        fn = os.path.join(dir_res, "train_data.pkl")
        pkl_file = open(fn, 'rb')
        data_train = pickle.load(pkl_file)
        print("Train data were loaded from pickle")
        pkl_file.close

        fn = os.path.join(dir_res, "test_labels.pkl")
        pkl_file = open(fn, 'rb')
        labels = pickle.load(pkl_file)
        print("Test labels were loaded from pickle")
        pkl_file.close

        test_data = np.asarray(data_test)
        print(test_data.shape)
        train_data = np.asarray(data_train)
        print(train_data.shape)

        names = labels
        print(len(names))

        train_test_data = np.concatenate((train_data, test_data), axis=0)

        encoded_vec = 0 # don't need it
        encoded_vec_train = 0 # don't need it
        encoded_vec_train_test = 0 # don't need it

    else:
        # preprocess the data and save test subset as pickle
        x_train, x_test, x_val, names, data_mean, data_std, data_test_vis = load_preprocess()
        print("Test data:", x_test.shape)

        # fn = os.path.join(dir_res, "test_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(x_test, pkl_file)
        # print("Test data were saved as pickle")
        # pkl_file.close

        # train_data = x_train[0:8000]
        # fn = os.path.join(dir_res, "train_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(train_data, pkl_file)
        # print("Train data were saved as pickle")
        # pkl_file.close

        # fn = os.path.join(dir_res, "test_labels.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(names, pkl_file)
        # print("Test labels were saved as pickle")
        # pkl_file.close

    model_names = {"2d_beta10_vae_cropped_128_relu_norm_1.h5", "2d_beta10_vae_cropped_128_relu_norm_2.h5", \
    "2d_beta6_vae_cropped_128_relu_norm_1.h5", "2d_beta6_vae_cropped_128_relu_norm_2.h5", \
    "2d_beta8_vae_cropped_128_relu_norm_1.h5", "2d_beta8_vae_cropped_128_relu_norm_2.h5", \
    "2d_beta100_vae_cropped_128_relu_norm_1.h5", "2d_beta100_vae_cropped_128_relu_norm_2.h5", "2d_beta100_vae_cropped_128_relu_norm_3.h5", \
    "2d_beta8_vae_cropped_256_relu_norm_1.h5", "2d_beta8_vae_cropped_256_relu_norm_2.h5", \
    "2d_beta2_vae_cropped_256_relu_norm_1.h5", "2d_beta2_vae_cropped_256_relu_norm_2.h5", "2d_beta2_vae_cropped_256_relu_norm_3.h5", \
    "2d_beta100_vae_cropped_256_relu_norm_1.h5", "2d_beta100_vae_cropped_256_relu_norm_2.h5", "2d_beta100_vae_cropped_256_relu_norm_3.h5",
    "2d_beta20_vae_cropped_128_relu_norm_1.h5", "2d_beta20_vae_cropped_128_relu_norm_2.h5", "2d_beta20_vae_cropped_128_relu_norm_3.h5", \
    "2d_beta20_vae_cropped_256_relu_norm_1.h5", "2d_beta20_vae_cropped_256_relu_norm_2.h5", "2d_beta20_vae_cropped_256_relu_norm_3.h5", \
    "2d_beta2_vae_cropped_64_relu_norm_1.h5", "2d_beta2_vae_cropped_64_relu_norm_2.h5", "2d_beta2_vae_cropped_64_relu_norm_3.h5", \
    "2d_beta2_vae_cropped_32_relu_norm_1.h5", "2d_beta2_vae_cropped_32_relu_norm_2.h5", "2d_beta2_vae_cropped_32_relu_norm_3.h5", \
    "2d_beta8_vae_cropped_64_relu_norm_1.h5", "2d_beta8_vae_cropped_64_relu_norm_2.h5", "2d_beta8_vae_cropped_64_relu_norm_3.h5", \
    "2d_beta8_vae_cropped_32_relu_norm_1.h5", "2d_beta8_vae_cropped_32_relu_norm_2.h5", "2d_beta8_vae_cropped_32_relu_norm_3.h5", \
    "2d_beta20_vae_cropped_64_relu_norm_1.h5", "2d_beta20_vae_cropped_64_relu_norm_2.h5", "2d_beta20_vae_cropped_64_relu_norm_3.h5", \
    "2d_beta20_vae_cropped_32_relu_norm_1.h5", "2d_beta20_vae_cropped_32_relu_norm_2.h5", "2d_beta20_vae_cropped_32_relu_norm_3.h5",
    "2d_beta8_vae_cropped_256_relu_norm_3.h5", "2d_beta6_vae_cropped_128_relu_norm_3.h5"}

    # model_names = {"2d_vae_cropped_128_relu_norm_3.h5", "2d_vae_cropped_128_relu_norm_4.h5", \
    # "2d_vae_cropped_256_relu_norm_1.h5", "2d_vae_cropped_256_relu_norm_2.h5", \
    # "2d_vae_cropped_256_relu_norm_3.h5", "2d_vae_cropped_256_relu_norm_4.h5", "2d_vae_cropped_256_relu_norm_5.h5", \
    # "2d_beta_vae_cropped_128_relu_norm_2.h5", "2d_beta_vae_cropped_128_relu_norm_3.h5", "2d_beta_vae_cropped_128_relu_norm_4.h5",
    # "2d_beta2_vae_cropped_128_relu_norm_3.h5", "2d_beta2_vae_cropped_128_relu_norm_4.h5",
    # "2d_beta2_vae_cropped_128_relu_norm_1.h5", "2d_beta2_vae_cropped_128_relu_norm_2.h5", "2d_beta2_vae_cropped_128_relu_norm_5.h5",
    # "2d_beta6_vae_cropped_128_relu_norm_4.h5", "2d_beta6_vae_cropped_128_relu_norm_5.h5",
    # "2d_beta8_vae_cropped_128_relu_norm_4.h5", "2d_beta8_vae_cropped_128_relu_norm_5.h5", "2d_beta8_vae_cropped_128_relu_norm_3.h5",
    # "2d_beta10_vae_cropped_128_relu_norm_4.h5", "2d_beta10_vae_cropped_128_relu_norm_5.h5",
    # "2d_beta20_vae_cropped_128_relu_norm_4.h5", "2d_beta20_vae_cropped_128_relu_norm_5.h5",
    # "2d_beta_vae_cropped_256_relu_norm_1.h5", "2d_beta_vae_cropped_256_relu_norm_2.h5",
    # "2d_beta_vae_cropped_256_relu_norm_3.h5", "2d_beta_vae_cropped_256_relu_norm_4.h5", "2d_beta_vae_cropped_256_relu_norm_5.h5",
    # "2d_beta6_vae_cropped_256_relu_norm_1.h5", "2d_beta6_vae_cropped_256_relu_norm_2.h5",
    # "2d_beta6_vae_cropped_256_relu_norm_3.h5", "2d_beta6_vae_cropped_256_relu_norm_4.h5", "2d_beta6_vae_cropped_256_relu_norm_5.h5",
    # "2d_beta10_vae_cropped_256_relu_norm_1.h5", "2d_beta10_vae_cropped_256_relu_norm_2.h5",
    # "2d_beta10_vae_cropped_256_relu_norm_3.h5", "2d_beta10_vae_cropped_256_relu_norm_4.h5", "2d_beta10_vae_cropped_256_relu_norm_5.h5"}

    dataset = "mcmc"
    title = '2D VAE: ' # for subtitle

    mod_nam = {"2d_vae_cropped_128_relu_norm", "2d_vae_cropped_256_relu_norm", 
    "2d_beta_vae_cropped_128_relu_norm", "2d_beta_vae_cropped_256_relu_norm", 
    "2d_beta2_vae_cropped_32_relu_norm", "2d_beta2_vae_cropped_64_relu_norm", 
    "2d_beta2_vae_cropped_128_relu_norm", "2d_beta2_vae_cropped_256_relu_norm", 
    "2d_beta6_vae_cropped_128_relu_norm", "2d_beta6_vae_cropped_256_relu_norm", 
    "2d_beta8_vae_cropped_32_relu_norm", "2d_beta8_vae_cropped_64_relu_norm",
    "2d_beta8_vae_cropped_128_relu_norm", "2d_beta8_vae_cropped_256_relu_norm",
    "2d_beta10_vae_cropped_128_relu_norm", "2d_beta10_vae_cropped_256_relu_norm",
    "2d_beta20_vae_cropped_32_relu_norm", "2d_beta20_vae_cropped_64_relu_norm",
    "2d_beta20_vae_cropped_128_relu_norm", "2d_beta20_vae_cropped_256_relu_norm",
    "2d_beta100_vae_cropped_128_relu_norm", "2d_beta100_vae_cropped_256_relu_norm"}

    mod_nam = {"2d_beta8_vae_cropped_128_relu_norm", "2d_beta2_vae_cropped_128_relu_norm"}

    # metrics stability add-on
    stability_study = True
    if (stability_study):
        print("Stability Study")
        model_names = models_metrics_stability(mod_nam, dataset)
    else:
        model_names_all = []
        for m_n in mod_nam:
            for i in range(5):    
                m_n_index = m_n + "_" + str(i+1) + ".h5"
                model_names_all.append(m_n_index)

        model_names = model_names_all
        print(model_names)

    # model_names = {"2d_beta8_vae_cropped_32_relu_norm_1.h5", "2d_beta8_vae_cropped_128_relu_norm_1.h5",
    # "2d_beta_vae_cropped_256_relu_norm_1.h5", "2d_beta2_vae_cropped_128_relu_norm_1.h5"}

    for model_name in model_names:
        print("model_name:", model_name)

        dir_res_model = model_directories(dir_res, model_name)
        os.makedirs(dir_res_model, exist_ok=True)

        filename = os.path.join(dir_res_model, "model_structure.txt")

        project = True
        interpolation = False
        if load_data:
            interpolation = False
        latent_vector = True

        #load_data = False
        if not load_data:
            # encoded_vec will be predicted with encoder

            # network parameters
            #input_shape = (image_size, image_size, 1)
            batch_size = 16
            kernel_size = 3
            filters = 64
            generic = False
            dense_dim = 1024
            latent_dim = 256
            epochs = 0 # 500
            conv_layers = 2
            stride = 2
            latent_vector = True
            beta_vae = False
            beta = 4
            regularization = False

            # Grid search in: latent_dim, activation, beta
            if("relu" in model_name):
                activation = 'relu'
            elif("tanh" in model_name):
                activation = 'tanh'

            if(activation == 'relu'):
                kernel_initializer = 'he_uniform'
            elif(activation == 'tanh'):
                kernel_initializer = 'glorot_uniform'

            if("vae" in model_name):
                beta_vae = False
            if("beta" in model_name):
                beta_vae = True
                beta = 4
            if("beta0.5" in model_name):
                beta = 0.5
            if("beta2" in model_name):
                beta = 2
            if("beta6" in model_name):
                beta = 6
            if("beta8" in model_name):
                beta = 8
            if("beta10" in model_name):
                beta = 10
            if("beta20" in model_name):
                beta = 20
            if("beta100" in model_name):
                beta = 100

            if("256" in model_name):
                dense_dim = 1024
                latent_dim = 256
            if("128" in model_name):
                dense_dim = 512
                latent_dim = 128
            if("_64_" in model_name):
                dense_dim = 256
                latent_dim = 64
            if("_32_" in model_name):
                dense_dim = 256
                latent_dim = 32

            # build encoder model
            inputs = Input(shape=(x_train.shape[1], x_train.shape[2], 1), name='encoded_input')
            encoded = inputs

            for _ in range(conv_layers):
                encoded = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            strides=stride,
                            padding='same')(encoded)
                if (generic == True): filters //= 2
                #encoded = BatchNormalization()(encoded)
            # num of conv param: kernel*kernel * channels_in * kernel_num + bias

            # Shape info needed to build Decoder Model
            encoded_shape = K.int_shape(encoded)
            print(encoded_shape)

            if (latent_vector == True): # generate dense layer
                encoded = generate_dense_layers(encoded, dense_dim, encoded_shape, activation, kernel_initializer)

            #latent_dim = 256 # 1024 -> 256; 512 -> 128
            z_mean = Dense(latent_dim, activation=activation, kernel_initializer=kernel_initializer, name='z_mean')(encoded)
            z_log_var = Dense(latent_dim, activation=activation, kernel_initializer=kernel_initializer, name='z_log_var')(encoded)
            # no regularization applied here, wrong! no need for vae +

            # use reparameterization trick to push the sampling out as input
            z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

            # Instantiate Encoder Model
            encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            print('Shape:',encoder.layers[-1].output_shape[1:])
            #encoder.summary()
            # with open(filename, "w") as text_file:
            #     encoder.summary(print_fn=lambda x: text_file.write(x + '\n'))

            # build decoder model
            if (latent_vector == True): # generate the latent vector
                decoded_input = Input(shape=(latent_dim, ), name='decoded_input')
                decoded = generate_fully_conn(decoded_input, encoded_shape, activation, kernel_initializer)
            else:
                decoded_input = Input(shape=(encoded_shape[1], encoded_shape[2], encoded_shape[3]), name='decoded_input')
                decoded = decoded_input
            
            for _ in range(conv_layers):
                if (generic == True): filters *= 2
                decoded = Conv2DTranspose(filters=filters,
                                    kernel_size=kernel_size,
                                    activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    strides=stride,
                                    padding='same')(decoded)
                #decoded = BatchNormalization()(decoded)

            decoded = Conv2DTranspose(filters=1,
                                kernel_size=kernel_size,
                                padding='same')(decoded)

            # Crop the decoded output so that dimensions are equal to the input
            decoded = cropping_output(decoded, x_train.shape)          

            outputs = Activation('linear', name='decoder_output')(decoded) # linear or sigmoid [0,1]
            # sigmoid - slow training? yes, as the derivative is small

            # instantiate decoder model
            decoder = Model(decoded_input, outputs, name='decoder')
            #decoder.summary()
            # with open(filename, "a") as text_file:
            #     decoder.summary(print_fn=lambda x: text_file.write(x + '\n'))

            # instantiate VAE model
            outputs = decoder(encoder(inputs)[2])
            vae = Model(inputs, outputs, name='vae')
            #vae.summary()
            # with open(filename, "a") as text_file:
            #     vae.summary(print_fn=lambda x: text_file.write(x + '\n'))

            from pathlib import Path
            my_file = Path(filename)
            if not my_file.is_file():
                print("no such file,", filename)
                with open(filename, "w") as text_file:
                    encoder.summary(print_fn=lambda x: text_file.write(x + '\n'))
                    decoder.summary(print_fn=lambda x: text_file.write(x + '\n'))
                    vae.summary(print_fn=lambda x: text_file.write(x + '\n'))

            try:
                # metrics stability add-on
                if (stability_study):
                    print("Stability Study")
                    model_name, dir_model_name, x_test_, names_ = model_name_metrics_stability(model_name, x_test, names, dataset)
                else:
                    dir_model_name = os.path.join("weights", model_name)

                vae.load_weights(dir_model_name)
                print("Loaded", dir_model_name, "model from disk")
                # input("!!!")
                # continue # skip existing models
            except IOError:
                print(dir_model_name, "model not accessible")
                epochs = 20 # train if no weights found

            # input("!!!")

            #autoencoder.compile(optimizer='adadelta', loss='mse') #
            lr = 0.0005
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False) # lr=0.0005

            from keras.losses import mse, binary_crossentropy, categorical_crossentropy
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            # E[log P(X|z)]
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))  # mean is returned
            # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
            #kl = 0.5 * K.sum( K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1 ) # sum
            kl_loss = -0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))

            kld_coeff = 1. / (x_train.shape[1] * x_train.shape[2] / latent_dim)

            if(beta_vae):
                kld_coeff *= beta # beta-VAE, use Lagrangian multiplier β under the KKT condition
                print("beta-VAE with β =", beta)

            print("KLD coeff: ", kld_coeff)
            kl_loss *= kld_coeff
            # kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss = K.mean(kl_loss, axis=-1)

            vae_loss = K.mean(reconstruction_loss + kl_loss)
            vae.add_loss(vae_loss)

            vae.compile(optimizer=adam)
            # K.optimizer.Adam(lr=0.001) 'adadelta'

            #from keras.callbacks import TensorBoard

            from keras.callbacks import EarlyStopping
            early_stopping = [EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=2, mode='auto',
                                restore_best_weights=True)]
            
            # train the whole autoencoder
            history_callback = vae.fit(x_train, #norm - not forget renormalize
                        epochs=epochs,
                        batch_size=batch_size, # batch size & learning rate
                        shuffle=True, verbose=2,
                        callbacks=early_stopping,
                        validation_data=(x_val, None)) # divide properly
                        #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]))
            
            if(epochs):
                vae.save_weights(dir_model_name)
                print("Saved", dir_model_name, "model weights to disk")

                loss_history = history_callback.history
                np_loss_history = np.array(loss_history)
                #print(np_loss_history)
                #np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
                with open(filename, "a") as text_file:
                    text_file.write("loss_history: ")
                    text_file.write(str(np_loss_history))

            if (stability_study):
                print("Stability Study")
                test_data = x_test_
                test_names = names_
            else:
                test_data = x_test # x_test x_train x_test_
                test_names = names
            train_data = x_train[0:8000]

            latent_representation = encoder.predict(test_data)
            encoded_vec = latent_representation[2]
            print('encoded_vec after reparam trick:', encoded_vec.shape) # (batch-size, latent_dim)
            print('encoded_vec max:', encoded_vec.max())
            print('encoded_vec min:', encoded_vec.min())
            # fig=plt.figure()
            # plt.tight_layout()
            # #fig.set_size_inches(8, 6)
            # plt.suptitle('2d_vae: Latent vectors')
            # plt.imshow(encoded_vec)
            # fig.savefig('{}/latent.png'.format(dir_res_model))
            # plt.close(fig)

            # clustering perf eval in the feature space
            n_clusters = 5
            # kmeans_rand(n_clusters, encoded_vec, names_, dir_res_model)
            # continue

            decoded_imgs = vae.predict(test_data)
            # print('decoded_imgs:', decoded_imgs.shape)
            # print('dec max:', decoded_imgs.max())
            # print('dec min:', decoded_imgs.min())

            #print(test_data.mean()) # 0
            #print(test_data.std()) # 1

            # print('normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            # print('normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())
            # un-normalize all using data_mean, data_std
            #test_data = test_data * data_std + data_mean
            # encoded_vec = encoded_vec * data_std + data_mean
            #decoded_imgs = decoded_imgs * data_std + data_mean
            # print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            # print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

            # draw original and reconstructed data
            # draw_orig_reconstr(test_data, decoded_imgs, title, dir_res_model, dataset)

            # test_data = data_test_vis # visualize the original

            # load_data = False # for now
            train_test_data = 0
            # encoded_vec = 0 # don't need it
            encoded_vec_train = 0 # don't need it
            encoded_vec_train_test = 0 # don't need it

            # train data
            latent_representation = encoder.predict(train_data)
            encoded_vec_train = latent_representation[2]
            print('encoded_vec_train after reparam trick:', encoded_vec_train.shape) # (batch-size, latent_dim)

            decoded_imgs = vae.predict(train_data)
            print('decoded_imgs:', decoded_imgs.shape)
            # print('dec max:', decoded_imgs.max())
            # print('dec min:', decoded_imgs.min())

            train_test_data = np.concatenate((train_data, test_data), axis=0)
            # train and test data
            latent_representation = encoder.predict(train_test_data)
            encoded_vec_train_test = latent_representation[2]
            print('encoded_vec_train_test after reparam trick:', encoded_vec_train_test.shape) # (batch-size, latent_dim)

            decoded_imgs = vae.predict(train_test_data)
            print('decoded_imgs:', decoded_imgs.shape)
            # print('dec max:', decoded_imgs.max())
            # print('dec min:', decoded_imgs.min())


        if (project == True):
            # project using PCA (then t-sne) and visualize the scatterplot
            #print("PCA projection")
            #pca_projection(encoded_vec, test_data, latent_vector, title, dataset)

            # project using UMAP and visualize the scatterplot
            print("UMAP projection")
            title_umap = title + 'Latent -> UMAP scatterplot'
            # umap_projection(encoded_vec, test_data, latent_vector, title_umap, dir_res_model, dataset, names)
            umap_projection(encoded_vec, encoded_vec_train, encoded_vec_train_test, test_data, train_data, train_test_data, latent_vector, title_umap, dir_res_model, dataset, test_names)

            # project using t-sne and visualize the scatterplot
            print("t-SNE projection")
            title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=20)
            # tsne_projection(encoded_vec, test_data, latent_vector, title_tsne, dir_res_model, dataset, names, perp=30)
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=40)

        if (interpolation == True):

            # spherical interpolation in the latent space
            
            test_data = x_test # x_test x_train

            # Interpolation in the latent space

            #spherical interpolation
            dec_interpol_sample_slerp, dec_interpol_sample_traversal = interpolate(encoded_vec, decoder, dir_res_model)

            # # find images that are close to the interpolatable
            find_interpolatable(dec_interpol_sample_slerp, test_data, dir_res_model, dataset)

            # latent dimension traversal with one seed image
            # seed_img = x_test[0] #
            # print(seed_img.shape)
            # seed_img.resize(1, x_test.shape[1], x_test.shape[2], 1)
            # latent_representation = encoder.predict(seed_img)
            # encoded_vec = latent_representation[2]
            
            # latent_dim_traversal(encoded_vec, decoder, dir_res_model)

        K.clear_session()

if __name__ == '__main__':
    main()



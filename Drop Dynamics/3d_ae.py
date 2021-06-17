import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils import model_directories, models_metrics_stability, model_name_metrics_stability
from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn

from pca_projection import pca_projection
from tsne_projection import tsne_projection
from umap_projection import umap_projection
from kmeans_rand import kmeans_rand

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

from keras.layers import Activation, Input, Dense, Conv3D, Conv3DTranspose
from keras.layers import Flatten, Reshape, Cropping3D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers

def cropping_output(decoded, input_shape):

    dec_shape = K.int_shape(decoded)
    print(dec_shape) 
    cropWidth, cropHeight = dec_shape[2] - input_shape[2], dec_shape[3] - input_shape[3]
    print('cropping:', cropWidth, cropHeight)
    cropLeft, cropTop = int(cropWidth / 2), int(cropHeight / 2)
    cropRight, cropBot = cropWidth - cropLeft, cropHeight - cropTop
    # cropping the output
    decoded = Cropping3D(cropping=((0, 0), (cropLeft, cropRight), (cropTop, cropBot),))(decoded)
    dec_shape = K.int_shape(decoded)
    print(dec_shape)  

    return decoded

def load_labelled_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("droplet_sampled_labelled_data.pkl", 'rb') # droplet_labelled_data_2
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    print("Loading names from pickle dump...")
    pkl_file = open("droplet_sampled_labelled_names.pkl", 'rb') # droplet_labelled_names_2
    names = []
    names = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    # if data.shape[0] != len(names):
    #     input("!!! Inconstintency in data and names !!!")
    
    data_sampled = data[:1800,...] # 1800 in total
    names_sampled = names[:1800]
    # test_idx = np.random.randint(data.shape[0], size=900) #3D
    # #print(test_idx)
    # data = data[test_idx,]
    # #names = names[test_idx,]
    # names_new = []
    # for idx in test_idx:
    #     names_new.append(names[idx])
    # names = names_new
    # print("Randomized sampling from test data")
    print(data.shape)
    print(len(names))
    print(names[0])
    print(np.unique(names))

    # for _ in np.unique(names):
    #     print(_)

    print("Loading data from pickle dump...")
    pkl_file = open("droplet_labelled_data_2.pkl", 'rb') # droplet_labelled_data_2
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    print("Loading names from pickle dump...")
    pkl_file = open("droplet_labelled_names_2.pkl", 'rb') # droplet_labelled_names_2
    names = []
    names = pickle.load(pkl_file)
    pkl_file.close

    print("All", data.shape[0], "labelled frames were loaded successfully")

    data = data[:5400,...] # 5682 in total
    names = names[:5400]

    data = np.concatenate((data, data_sampled), axis=0)
    names = names + names_sampled

    print(data.shape)
    print(len(names))
    print(names[0])
    print(np.unique(names))

    return data, names

def load_unlabelled_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("droplet_unlabelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    data = data[0:15000,...] #15K
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
    print(data.shape)

    dataset = "droplet" # for pca t-sne umap preprocessing vis
    temporal = True

    # empty names
    empty_names = []
    for _ in range(data_train.shape[0]):
        empty_names.append("")
    all_names = empty_names + names # empty + test

    visualize_data = False
    data, data_mean, data_std, all_names = preprocess(dataset, visualize_data, data, all_names, temporal) # reshape, visualize, normalize, scale
    print(data.shape)

    data = brightness_normalization(data) # test

    train_shape = int(data_train.shape[0]/3) # after 3 in 1 preprocessing
    names = all_names[train_shape:]
    data_train = data[:train_shape,]
    data_test = data[train_shape:,]

    from sklearn.utils import shuffle
    data_test, names = shuffle(data_test, names, random_state=0)
    print("Shuffled test set")
    print(data_test.shape)

    data_test_vis = data_test # unnormalized -, uncropped +

    # cropping:
    crop_left = int(data.shape[3]*0.15) # start from left 10 15
    crop_right = int(data.shape[3]*0.8) # end at right 15 20
    crop_bottom = int(data.shape[2]*0.83) # remove bottom 10 15 18 17+ 16- 15still 12bad
    data = data[:,:,:,crop_left:crop_right,:]
    data_train = data_train[:,:,:,crop_left:crop_right,:]
    data_train = data_train[:,:,:crop_bottom,:,:]
    data_test = data_test[:,:,:,crop_left:crop_right,:]
    data_test = data_test[:,:,:crop_bottom,:,:]
    print("train set cropped: ", data_train.shape)
    print("test set:", data_test.shape, len(names))

    # data_test_vis = data_test # unnormalized, uncropped # this is 2d since before preprocess()

    # # randomize the test samples
    # test_idx = np.random.randint(data_test.shape[0], size=600)
    # data_test = data_test[test_idx,]
    # data_test_vis = data_test_vis[test_idx,]
    # names_new = []
    # for idx in test_idx:
    #     names_new.append(names[idx])
    # names = names_new
    # print("Randomized sampling from test data")
    # print(data_test.shape)

    data_train, data_val = train_test_split(data_train, test_size=0.2, random_state=1)
    print('train & val', data_train.shape, data_val.shape)

    x_train, x_test, x_val = data_train, data_test, data_val

    return x_train, x_test, x_val, names, data_mean, data_std, data_test_vis

def main():

    dir_res = "Results/3D_AE" # directory with all models

    # Load data and subsequently encoded vectors in 2D representation
    # for this save before x_test and encoded vec after tsne and umap
    load_data = False
    if load_data: 
        dir_res = "Results/3D_VAE" # same as for 3D VAE
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
        print("Test data: ", test_data.shape)
        train_data = np.asarray(data_train)
        print("Train data: ", train_data.shape)

        names = labels
        print(len(names))

        train_test_data = np.concatenate((train_data, test_data), axis=0)

        encoded_vec = 0 # don't need it
        encoded_vec_train = 0 # don't need it
        encoded_vec_train_test = 0 # don't need it
    else:
        #dir_res = "Results/3D_VAE" # test data is same for vae and ae
        # preprocess the data and save test subset as pickle
        x_train, x_test, x_val, names, data_mean, data_std, data_test_vis = load_preprocess()

        # dir_res = "Results/3D_VAE" # same as for 3D VAE
        # fn = os.path.join(dir_res, "test_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(x_test, pkl_file)
        # print("Test data were saved as pickle")
        # pkl_file.close

        # fn = os.path.join(dir_res, "train_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(x_train, pkl_file)
        # print("Train data were saved as pickle")
        # pkl_file.close

        # fn = os.path.join(dir_res, "test_labels.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(names, pkl_file)
        # print("Test labels were saved as pickle")
        # pkl_file.close

        # input("x")

    # model_names = {"3d_ae_cropped_256_relu_norm_1.h5", "3d_ae_cropped_256_relu_norm_2.h5", \
    # "3d_ae_cropped_256_relu_norm_3.h5", "3d_ae_cropped_256_relu_norm_4.h5", "3d_ae_cropped_256_relu_norm_5.h5", \
    # "3d_ae_cropped_512_relu_norm_1.h5", "3d_ae_cropped_512_relu_norm_2.h5", \
    # "3d_ae_cropped_512_relu_norm_3.h5", "3d_ae_cropped_512_relu_norm_4.h5", "3d_ae_cropped_512_relu_norm_5.h5", \
    # "3d_ae_cropped_256_relu_reg_norm_1.h5", "3d_ae_cropped_256_relu_reg_norm_2.h5", \
    # "3d_ae_cropped_256_relu_reg_norm_3.h5", "3d_ae_cropped_256_relu_reg_norm_4.h5", "3d_ae_cropped_256_relu_reg_norm_5.h5", \
    # "3d_ae_cropped_512_relu_reg_norm_1.h5", "3d_ae_cropped_512_relu_reg_norm_2.h5", \
    # "3d_ae_cropped_512_relu_reg_norm_3.h5", "3d_ae_cropped_512_relu_reg_norm_4.h5", "3d_ae_cropped_512_relu_reg_norm_5.h5", \
    # "3d_ae_cropped_128_relu_norm_1.h5", "3d_ae_cropped_128_relu_norm_2.h5", \
    # "3d_ae_cropped_128_relu_norm_3.h5", "3d_ae_cropped_128_relu_norm_4.h5", "3d_ae_cropped_128_relu_norm_5.h5", \
    # "3d_ae_cropped_64_relu_norm_1.h5", "3d_ae_cropped_64_relu_norm_2.h5", \
    # "3d_ae_cropped_64_relu_norm_3.h5", "3d_ae_cropped_64_relu_norm_4.h5", "3d_ae_cropped_64_relu_norm_5.h5", \
    # "3d_ae_cropped_32_relu_norm_1.h5", "3d_ae_cropped_32_relu_norm_2.h5", \
    # "3d_ae_cropped_32_relu_norm_3.h5", "3d_ae_cropped_32_relu_norm_4.h5", "3d_ae_cropped_32_relu_norm_5.h5"}

    model_names = {"3d_ae_croppedb_256_relu_norm_1.h5", "3d_ae_croppedb_256_relu_norm_2.h5", \
    "3d_ae_croppedb_256_relu_norm_3.h5", "3d_ae_croppedb_256_relu_norm_4.h5", "3d_ae_croppedb_256_relu_norm_5.h5", \
    "3d_ae_croppedb_256_relu_reg_norm_1.h5", "3d_ae_croppedb_256_relu_reg_norm_2.h5", \
    "3d_ae_croppedb_256_relu_reg_norm_3.h5", "3d_ae_croppedb_256_relu_reg_norm_4.h5", "3d_ae_croppedb_256_relu_reg_norm_5.h5", \
    "3d_ae_croppedb_128_relu_norm_1.h5", "3d_ae_croppedb_128_relu_norm_2.h5", \
    "3d_ae_croppedb_128_relu_norm_3.h5", "3d_ae_croppedb_128_relu_norm_4.h5", "3d_ae_croppedb_128_relu_norm_5.h5", \
    "3d_ae_croppedb_64_relu_norm_1.h5", "3d_ae_croppedb_64_relu_norm_2.h5", \
    "3d_ae_croppedb_64_relu_norm_3.h5", "3d_ae_croppedb_64_relu_norm_4.h5", "3d_ae_croppedb_64_relu_norm_5.h5"}

    dir_res = "Results/3D_AE" # directory with all models
    dataset = "droplet"
    title = '3D AE: ' # for subtitle

    mod_nam = {"3d_ae_croppedb_256_relu_norm", 
    "3d_ae_croppedb_256_relu_reg_norm", 
    "3d_ae_croppedb_128_relu_norm", 
    "3d_ae_croppedb_64_relu_norm"}

    mod_nam = {"3d_ae_croppedb_256_relu_norm.h5"}
    # mod_nam = {"3d_ae_croppedb_16_relu_norm"}

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

    for model_name in model_names:
        print("model_name:", model_name)

        dir_res_model = model_directories(dir_res, model_name)
        os.makedirs(dir_res_model, exist_ok=True)

        filename = os.path.join(dir_res_model, "model_structure.txt")

        project = True
        interpolation = False
        latent_vector = True

        if not load_data:
            # encoded_vec will be predicted with encoder

            temporal = True
            # network parameters
            #input_shape = (image_size, image_size, 1)
            batch_size = 16
            kernel_size = (3, 3, 3)
            filters = 64
            generic = False
            dense_dim = 128 #1024
            latent_dim = 2 #512
            epochs = 0 # 500
            conv_layers = 4
            stride = (3, 2, 2)
            regularization = False
            dropout_sparcity = False
                
            # Grid search in: latent_dim, activation, regularization
            if("relu" in model_name):
                activation = 'relu'
            elif("tanh" in model_name):
                activation = 'tanh'

            if(activation == 'relu'):
                kernel_initializer = 'he_uniform'
            elif(activation == 'tanh'):
                kernel_initializer = 'glorot_uniform'

            if("reg" in model_name):
                regularization = True
                #epochs = 30

            if("drop" in model_name):
                dropout_sparcity = True
                #epochs = 30

            if("512" in model_name):
                dense_dim = 1024
                latent_dim = 512
            if("256" in model_name):
                dense_dim = 512
                latent_dim = 256
            if("128" in model_name):
                dense_dim = 256
                latent_dim = 128
            if("64" in model_name):
                dense_dim = 256
                latent_dim = 64
            if("32" in model_name):
                dense_dim = 128
                latent_dim = 32
            if("_16_" in model_name):
                dense_dim = 128
                latent_dim = 16

            # build encoder model
            inputs = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), name='encoded_input')
            encoded = inputs

            for _ in range(conv_layers):
                if (_ == 1): 
                    kernel_size = (1, 3, 3)
                    stride = (1, 2, 2)
                encoded = Conv3D(filters=filters,
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

            if(dropout_sparcity):
                encoded = Dropout(0.5, seed=1)(encoded)

            if (latent_vector == True): # generate dense layer and latent space
                encoded = generate_dense_layers(encoded, dense_dim, encoded_shape, activation, kernel_initializer,
                                                    latent_dim, regularization, temporal)
            
            #encoded = BatchNormalization()(encoded) # for relu

            #latent_dim = 256
            #encoded = Dense(latent_dim, activation='tanh', name='latent_vec')(encoded)

            # Instantiate Encoder Model
            encoder = Model(inputs, encoded, name='encoder')
            print('Shape:',encoder.layers[-1].output_shape[1:])
            #encoder.summary()

            # build decoder model
            if (latent_vector == True): # generate the latent vector
                decoded_input = Input(shape=(latent_dim, ), name='decoded_input')
                decoded = generate_fully_conn(decoded_input, encoded_shape, activation, kernel_initializer, temporal)
            else:
                decoded_input = Input(shape=(encoded_shape[1], encoded_shape[2], encoded_shape[3]), name='decoded_input')
                decoded = decoded_input
            
            for _ in range(conv_layers):
                if (generic == True): filters *= 2
                if (_ == conv_layers-1): 
                    kernel_size = (3, 3, 3)
                    stride = (3, 2, 2)
                decoded = Conv3DTranspose(filters=filters,
                                    kernel_size=kernel_size,
                                    activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    strides=stride,
                                    padding='same')(decoded)
                #decoded = BatchNormalization()(decoded)

            decoded = Conv3DTranspose(filters=1,
                                kernel_size=kernel_size,
                                padding='same')(decoded)

            # Crop the decoded output so that dimensions are equal to the input
            decoded = cropping_output(decoded, x_train.shape)               

            outputs = Activation('linear', name='decoder_output')(decoded) # linear or sigmoid [0,1] tanh [-1,1]
            # sigmoid - slow training? yes, as the derivative is small

            # instantiate decoder model
            decoder = Model(decoded_input, outputs, name='decoder')
            #decoder.summary()

            # instantiate Autoencoder model
            outputs = decoder(encoder(inputs))
            #print('output:', outputs.shape)
            autoencoder = Model(inputs, outputs, name='autoencoder')
            #autoencoder.summary()
            # with open(filename, "a") as text_file:
            #     autoencoder.summary(print_fn=lambda x: text_file.write(x + '\n'))

            from pathlib import Path
            my_file = Path(filename)
            if not my_file.is_file():
                print("no such file,", filename)
                with open(filename, "w") as text_file:
                    encoder.summary(print_fn=lambda x: text_file.write(x + '\n'))
                    decoder.summary(print_fn=lambda x: text_file.write(x + '\n'))
                    autoencoder.summary(print_fn=lambda x: text_file.write(x + '\n'))
            
            try:
                # metrics stability add-on
                if (stability_study):
                    print("Stability Study")
                    model_name, dir_model_name, x_test_, names_ = model_name_metrics_stability(model_name, x_test, names, dataset)
                else:
                    dir_model_name = os.path.join("weights", model_name)

                autoencoder.load_weights(dir_model_name)
                print("Loaded", dir_model_name, "model from disk")
            except IOError:
                print(dir_model_name, "model not accessible")
                epochs = 20 # train if no weights found
            
            #autoencoder.compile(optimizer='adadelta', loss='mse') #
            lr = 0.0005
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

            from keras.losses import mse, binary_crossentropy, categorical_crossentropy
            # do not scale to [0,1] if using mse loss
            #mse = mse(K.flatten(inputs), K.flatten(outputs[:,:-diff_x,:-diff_y,:]))
            autoencoder.compile(optimizer=adam, loss='mse') # binary_crossentropy + softmax if normalized [0,1]  mse
            # K.optimizer.Adam(lr=0.001) 'adadelta'

            #from keras.callbacks import TensorBoard

            from keras.callbacks import EarlyStopping
            early_stopping = [EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=2, mode='auto',
                                restore_best_weights=True)]

            # train the whole autoencoder
            history_callback = autoencoder.fit(x_train, x_train, #norm - not forget renormalize
                            epochs=epochs,
                            batch_size=batch_size, # batch size & learning rate
                            shuffle=True, verbose=2,
                            callbacks=early_stopping,
                            validation_data=(x_val, x_val)) # divide properly
                            #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]))

            if(epochs):
                autoencoder.save_weights(dir_model_name)
                print("Saved", model_name, "model weights to disk")

                loss_history = history_callback.history
                np_loss_history = np.array(loss_history)
                #print(np_loss_history)
                #np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
                with open(filename, "a") as text_file:
                    text_file.write("loss_history: ")
                    text_file.write(str(np_loss_history))

            # Keract visualizations
            #visualize(x_train, encoder, decoder)

            if (stability_study):
                print("Stability Study")
                test_data = x_test_
                test_names = names_
            else:
                test_data = x_test # x_test x_train x_test_
                test_names = names
            train_data = x_train
            # names = "" # no labels for x_train
            # test_data = x_train and x_test

            # load_data = False # for now
            train_test_data = 0
            # encoded_vec = 0 # don't need it
            encoded_vec_train = 0 # don't need it
            encoded_vec_train_test = 0 # don't need it

            encoded_vec = encoder.predict(test_data)
            # print(encoded_vec)
            print('encoded_vectors:', encoded_vec.shape) # (batch-size, latent_dim)
            # fig=plt.figure()
            # plt.tight_layout()
            # #fig.set_size_inches(8, 6)
            # plt.suptitle('3d_ae: Latent vectors')
            # plt.imshow(encoded_vec)
            # fig.savefig('{}/latent.png'.format(dir_res_model))
            # plt.close(fig)

            # clustering in the feature space
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=8, random_state=0).fit(encoded_vec)
            # print(kmeans.labels_)

            # clustering perf eval in the feature space
            n_clusters = 8
            # kmeans_rand(n_clusters, encoded_vec, names_, dir_res_model)
            # continue

            decoded_imgs = autoencoder.predict(test_data)
            print('decoded_imgs:', decoded_imgs.shape)
            print('dec max:', decoded_imgs.max())
            print('dec min:', decoded_imgs.min())

            # spherical interpolation in the latent space 
            #dec_interpol_sample_slerp, dec_interpol_sample_traversal = interpolate(encoded_vec, decoder)

            # find images that are close to the interpolatable
            #find_interpolatable(dec_interpol_sample_slerp, test_data)

            # un-normalize all using data_mean, data_std
            # print("test_data, encoded_vec, decoded_imgs")
            print('normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            print('normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())
            #test_data = test_data * data_std + data_mean
            #encoded_vec = encoded_vec * data_std + data_mean
            #decoded_imgs = decoded_imgs * data_std + data_mean
            # print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            # print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

            # test_data = data_test_vis # visualize the original

            # draw original and reconstructed data
            # draw_orig_reconstr(test_data, decoded_imgs, title, dir_res_model, dataset, temporal)

            #test_data = data_test_vis # visualize the original

            # # train data
            # encoded_vec_train = encoder.predict(train_data)
            # print('encoded_vec_train after reparam trick:', encoded_vec_train.shape) # (batch-size, latent_dim)

            # decoded_imgs = autoencoder.predict(train_data)
            # print('decoded_imgs:', decoded_imgs.shape)
            # print('dec max:', decoded_imgs.max())
            # print('dec min:', decoded_imgs.min())

            train_test_data = np.concatenate((train_data, test_data), axis=0)
            # train and test data
            encoded_vec_train_test = encoder.predict(train_test_data)
            print('encoded_vec_train_test after reparam trick:', encoded_vec_train_test.shape) # (batch-size, latent_dim)

            decoded_imgs = autoencoder.predict(train_test_data)
            print('decoded_imgs:', decoded_imgs.shape)
            print('dec max:', decoded_imgs.max())
            print('dec min:', decoded_imgs.min())

        if (project == True):
            # project using PCA (then t-sne) and visualize the scatterplot
            #print("PCA projection")
            #pca_projection(encoded_vec, test_data, latent_vector, title, dataset)

            # project using t-sne and visualize the scatterplot
            # print("t-SNE projection")
            #title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=20)
            # tsne_projection(encoded_vec, test_data, latent_vector, title_tsne, dir_res_model, dataset, names, temporal=True, perp=30)
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=40)

            # project using UMAP and visualize the scatterplot
            print("UMAP projection")
            title_umap = title + 'Latent -> UMAP scatterplot'
            umap_projection(encoded_vec, encoded_vec_train, encoded_vec_train_test, test_data, train_data, train_test_data, latent_vector, title_umap, dir_res_model, dataset, test_names, temporal=True)

        if (interpolation == True):
            #Interpolation in the latent space

            #spherical interpolation
            dec_interpol_sample_slerp, dec_interpol_sample_traversal = interpolate(encoded_vec, decoder, dir_res_model, temporal)

            # find images that are close to the interpolatable
            test_data = x_test # to compare interpolation with original samples
            find_interpolatable(dec_interpol_sample_slerp, test_data, dir_res_model, dataset, temporal)

            # #latent dimension traversal with one seed image
            # seed_img = x_test[10] #
            # #print(seed_img)
            # #print(seed_img.shape)
            # seed_img = np.zeros((3, 160, 168, 1)) # no!
            # #print(seed_img)
            # #print(type(seed_img))
            # #print(seed_img.shape)
            # #seed_img.resize(1, 3, x_test.shape[2], x_test.shape[3], 1) 
            # #print(x_test.shape[2])
            # seed_img.resize(1, 3, seed_img.shape[1], seed_img.shape[2], 1)
            # latent_representation = encoder.predict(seed_img)
            # encoded_vec = latent_representation[2]

            # latent_dim_traversal(encoded_vec, decoder, dir_res_model, dataset, temporal)

        K.clear_session()

if __name__ == '__main__':
    main()


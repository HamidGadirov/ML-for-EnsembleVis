"""
spatial-temporal conv, determine where -
"""

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn

#from pca_projection import pca_projection
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
from sklearn.model_selection import train_test_split

from keras.layers import Activation, Input, Dense, Conv3D, Conv3DTranspose
from keras.layers import Flatten, Reshape, Cropping3D, Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers

def load_labelled_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("sampled-300_labelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    print("Loading names from pickle dump...")
    pkl_file = open("sampled-300_labelled_names.pkl", 'rb')
    names = []
    names = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    print(data.shape) # 900,
    print(len(names))
    print(names[0])

    return data, names

def load_unlabelled_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("sampled-300_unlabelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    print("Loading names from pickle dump...")
    pkl_file = open("sampled-300_unlabelled_names.pkl", 'rb')
    names = []
    names = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All", data.shape[0], "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    data = data[0:9000]
    names = names[0:9000]
    print(data.shape)
    print(len(names))
    print(names[0])

    return data, names

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

def load_preprocess():
        
    data_test, cylinder_names_test = load_labelled_data() # for test only, 900
    data_train, names_train = load_unlabelled_data() # for train only

    data = np.zeros((data_test.shape[0]+data_train.shape[0], 441, 84, 1))
    data[:data_train.shape[0],] = data_train
    data[data_train.shape[0]:,] = data_test

    names = names_train + cylinder_names_test

    dataset = "flow"
    temporal = True

    visualize = False
    data, data_mean, data_std, names = preprocess(dataset, visualize, data, names, temporal) # reshape, visualize, normalize, scale
    print(data.shape)

    train_shape = int(data_train.shape[0]/3)

    data_train = data[:train_shape,]
    data_test = data[train_shape:,]

    cylinder_names_train = names[:train_shape]
    cylinder_names_test = names[train_shape:]

    from sklearn.utils import shuffle
    data_test, cylinder_names_test = shuffle(data_test, cylinder_names_test, random_state=0)
    print("Shuffled test set")
    print(data_test.shape)
    print(len(cylinder_names_test))

    data_train, data_val, cylinder_names_train, cylinder_names_val = train_test_split(data_train, cylinder_names_train, test_size=0.2, random_state=1)
    print('train & val', data_train.shape, data_val.shape)

    x_train, x_test, x_val = data_train, data_test, data_val

    return x_train, x_test, x_val, cylinder_names_test, data_mean, data_std, data_test

def main():

    dir_res = "Results/3D_AE" # directory with all models

    # Load data and subsequently encoded vectors in 2D representation
    # for this save before x_test and encoded vec after tsne and umap
    load_data = True
    if load_data: 
        dir_res = "Results/3D_VAE" # test data is same for vae and ae
        # load test_data from pickle and later encoded_vec_2d
        fn = os.path.join(dir_res, "test_data.pkl")
        pkl_file = open(fn, 'rb')
        data = pickle.load(pkl_file)
        print("Test data were loaded from pickle")
        pkl_file.close

        fn = os.path.join(dir_res, "test_labels.pkl")
        pkl_file = open(fn, 'rb')
        labels = pickle.load(pkl_file)
        print("Test labels were loaded from pickle")
        pkl_file.close

        test_data = np.asarray(data)
        print(test_data.shape)
        test_data_vis = test_data

        names = labels
        print(len(names))

        encoded_vec = 0 # don't need it
    else:
        # preprocess the data and save test subset as pickle
        x_train, x_test, x_val, names, data_mean, data_std, data_test_vis = load_preprocess()

        # crop the input to remove cylinder feature; data_test is to visualize original data
        cropping = True
        if (cropping):
            cut_value = int(0.3*x_train.shape[2])
            x_train = x_train[:,:,cut_value:,:,:]
            x_test = x_test[:,:,cut_value:,:,:]
            x_val = x_val[:,:,cut_value:,:,:]
            print("x_train cropped: ", x_train.shape)

        # print('normalized:', x_test.min(), x_test.max())
        # test_data_vis = x_test
        # test_data_vis = test_data_vis * data_std + data_mean
        # x_test = test_data_vis
        # print('unnormalized:', x_test.min(), x_test.max())

        # fn = os.path.join(dir_res, "test_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(x_test, pkl_file)
        # print("Test data were saved as pickle")
        # pkl_file.close

        # fn = os.path.join(dir_res, "test_labels.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(names, pkl_file)
        # print("Test labels were saved as pickle")
        # pkl_file.close

        # x_test = (x_test - x_test.mean()) / x_test.std() # back to unnorm

    model_names = {"3d_ae_cropped_256_relu_reg_1.h5", "3d_ae_cropped_256_relu_reg_2.h5", \
    "3d_ae_cropped_256_relu_reg_3.h5", "3d_ae_cropped_256_relu_reg_4.h5", "3d_ae_cropped_256_relu_reg_5.h5", \
    "3d_ae_cropped_256_relu_1.h5", "3d_ae_cropped_256_relu_2.h5", \
    "3d_ae_cropped_256_relu_3.h5", "3d_ae_cropped_256_relu_4.h5", "3d_ae_cropped_256_relu_5.h5", \
    "3d_ae_cropped_512_relu_reg_1.h5", "3d_ae_cropped_512_relu_reg_2.h5", \
    "3d_ae_cropped_512_relu_reg_3.h5", "3d_ae_cropped_512_relu_reg_4.h5", "3d_ae_cropped_512_relu_reg_5.h5", \
    "3d_ae_cropped_512_relu_1.h5", "3d_ae_cropped_512_relu_2.h5", \
    "3d_ae_cropped_512_relu_3.h5", "3d_ae_cropped_512_relu_4.h5", "3d_ae_cropped_512_relu_5.h5"}

    dir_res = "Results/3D_AE" # directory with all models
    dataset = "flow"
    title = '3D AE: ' # for subtitle

    #model_names = {"3d_ae_cropped_256_tanh_1.h5"} #

    for model_name in model_names:
        print("model_name:", model_name)

        model = model_name[:-5]
        dir_res_m = os.path.join(dir_res, model)
        print("Saved here:", dir_res_m)
        model = model_name[:-3]
        dir_res_model = os.path.join(dir_res_m, model)
        os.makedirs(dir_res_model, exist_ok=True)

        filename = os.path.join(dir_res_model, "model_structure.txt")

        project = True
        latent_vector = True
        temporal = True

        if not load_data:
            # encoded_vec will be predicted with encoder

            # network parameters
            #input_shape = (image_size, image_size, 1)
            batch_size = 16
            kernel_size = (3, 3, 3)
            filters = 64
            generic = False
            dense_dim = 1024
            latent_dim = 512
            epochs = 0 # 500
            conv_layers = 4
            stride = (3, 2, 2)
            latent_vector = True
            project = True
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

            if("512" in model_name):
                dense_dim = 1024
                latent_dim = 512
            if("256" in model_name):
                dense_dim = 512
                latent_dim = 256

            if("reg" in model_name):
                regularization = True
                # epochs = 20

            if("drop" in model_name):
                dropout_sparcity = True
                # epochs = 40

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

            """ # try diff pos of conv / trmp 
            encoded = Conv3D(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(encoded)
            encoded = Conv3D(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(encoded)
            encoded = Conv3D(filters=filters,kernel_size=(3,3,3),activation=activation,strides=(3,2,2),padding='same')(encoded)
            encoded = Conv3D(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(encoded)
            encoded = Conv3D(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(encoded)
            encoded = BatchNormalization()(encoded)
            """
            # Shape info needed to build Decoder Model
            encoded_shape = K.int_shape(encoded)
            print(encoded_shape)

            if(dropout_sparcity):
                encoded = Dropout(0.5, seed=1)(encoded)

            if (latent_vector == True): # generate dense layer and latent space
                encoded = generate_dense_layers(encoded, dense_dim, encoded_shape, activation, kernel_initializer,
                                                    latent_dim, regularization, temporal)
        
            # Instantiate Encoder Model
            encoder = Model(inputs, encoded, name='encoder')
            print('Shape:',encoder.layers[-1].output_shape[1:])
            #encoder.summary()
            # with open(filename, "w") as text_file:
            #     encoder.summary(print_fn=lambda x: text_file.write(x + '\n'))

            # build decoder model
            if (latent_vector == True): # generate fully conn layer
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
            """
            decoded = Conv3DTranspose(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(decoded)
            decoded = Conv3DTranspose(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(decoded)
            decoded = Conv3DTranspose(filters=filters,kernel_size=(3,3,3),activation=activation,strides=(3,2,2),padding='same')(decoded)
            decoded = Conv3DTranspose(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(decoded)
            decoded = Conv3DTranspose(filters=filters,kernel_size=(1,3,3),activation=activation,strides=(1,2,2),padding='same')(decoded)
            decoded = BatchNormalization()(decoded)
            """
            decoded = Conv3DTranspose(filters=1,
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
                f = open(model_name)
                autoencoder.load_weights(model_name)
                print("Loaded", model_name, "model from disk")
            except IOError:
                print(model_name, "model not accessible")

            #autoencoder.compile(optimizer='adadelta', loss='mse') #
            lr = 0.0005
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False) # lr=0.005

            from keras.losses import mse, binary_crossentropy, categorical_crossentropy
            # do not scale to [0,1] if using mse loss
            #mse = mse(K.flatten(inputs), K.flatten(outputs[:,:-diff_x,:-diff_y,:]))
            autoencoder.compile(optimizer=adam, loss='mse') # binary_crossentropy if normalized [0,1]  mse

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
                autoencoder.save_weights(model_name)
                print("Saved", model_name, "model weights to disk")

                loss_history = history_callback.history
                np_loss_history = np.array(loss_history)
                #print(np_loss_history)
                #np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
                with open(filename, "a") as text_file:
                    text_file.write("loss_history: ")
                    text_file.write(str(np_loss_history))

            test_data = x_test # x_test x_train
            #test_data_vis = data_test # data_test data_train
            test_data_vis = x_test # vis what the model sees

            encoded_vec = encoder.predict(test_data) # x_test
            print('encoded_vectors:', encoded_vec.shape) # (batch-size, latent_dim)
            fig=plt.figure()
            plt.suptitle('3d_ae: Latent vectors')
            plt.imshow(encoded_vec)
            fig.savefig('{}/latent.png'.format(dir_res_model), dpi=300)

            decoded_imgs = autoencoder.predict(test_data) # x_test
            print('decoded_imgs:', decoded_imgs.shape)
            print('dec max:', decoded_imgs.max())
            print('dec min:', decoded_imgs.min())

            # spherical interpolation in the latent space
            #dec_interpol_sample_slerp, dec_interpol_sample_traversal = interpolate(encoded_vec, decoder)

            # find images that are close to the interpolatable
            #find_interpolatable(dec_interpol_sample_slerp, test_data)

            #print('normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            #print('normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())
            # un-normalize all using data_mean, data_std
            test_data_vis = test_data_vis * data_std + data_mean
            #encoded_vec = encoded_vec * data_std + data_mean
            decoded_imgs = decoded_imgs * data_std + data_mean
            #print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            #print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

            # draw original and reconstructed data
            draw_orig_reconstr(test_data_vis, decoded_imgs, title, dir_res_model, dataset, temporal)

        if (project == True):
            # project using PCA (then t-sne) and visualize the scatterplot
            #print("PCA projection")
            #pca_projection(encoded_vec, test_data_vis, latent_vector, title, dataset, names, temporal)

            # project using t-sne and visualize the scatterplot
            print("t-SNE projection")
            title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=20)
            tsne_projection(encoded_vec, test_data_vis, latent_vector, title_tsne, dir_res_model, dataset, names, temporal, perp=30)
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=40)

            # project using UMAP and visualize the scatterplot
            print("UMAP projection")
            title_umap = title + 'Latent -> UMAP scatterplot'
            umap_projection(encoded_vec, test_data_vis, latent_vector, title_umap, dir_res_model, dataset, names, temporal)


if __name__ == '__main__':
    main()

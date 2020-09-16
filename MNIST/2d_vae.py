
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn
from reparameterization_trick import sampling

#from pca_projection import pca_projection
from tsne_projection import tsne_projection
from umap_projection import umap_projection

from visualization import visualize_keract, visualize_keras
from latent_sp_interpolation import interpolate, find_interpolatable, latent_dim_traversal

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
import time
from progress.bar import Bar
import pickle
from sklearn.model_selection import train_test_split

from keras.layers import Activation, Input, Dense, Conv2D, Conv2DTranspose
from keras.layers import Flatten, Reshape, Cropping2D, Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers     


def main():
    model_name = "2d_vae_32_relu_1.h5"
    #model_name = "2d_beta_vae_32_relu_1.h5"
    #model_name = "2d_beta2_vae_32_relu_1.h5"
    #model_name = "2d_beta0.1_vae_32_relu_1.h5"

    dataset = "mnist"
    title = '2D VAE: ' # for subtitle

    dir_res = "Results/2D_VAE"
    model = model_name[:-5]
    dir_res_m = os.path.join(dir_res, model)
    print("Saved here:", dir_res_m)
    model = model_name[:-3]
    dir_res_model = os.path.join(dir_res_m, model)
    os.makedirs(dir_res_model, exist_ok=True)

    project = True
    latent_vector = True

    # load data and vis
    # for this save before x_test and encoded after tsne and umap
    load_data = True
    if load_data: 
        # load test_data from pickle and later encoded_vec_2d
        fn = os.path.join(dir_res, "test_data.pkl")
        pkl_file = open(fn, 'rb')
        data = pickle.load(pkl_file)
        print("Test data is loaded from pickle")
        pkl_file.close

        fn = os.path.join(dir_res, "test_labels.pkl")
        pkl_file = open(fn, 'rb')
        labels = pickle.load(pkl_file)
        print("Test data is loaded from pickle")
        pkl_file.close

        test_data = np.asarray(data)
        print(test_data.shape)

        print(labels.shape)

        encoded_vec = 0 # don't need it
    else:
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #x_train = x_train.astype('float32') / 255.
        #x_test = x_test.astype('float32') / 255.
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
        #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
        print(x_train.shape)

        data = x_train[:30000,...]
        print(data.shape)

        visualize = False
        data, data_mean, data_std = preprocess(dataset, visualize, data) # reshape, visualize, normalize, scale

        x_train = data[:20000,...]
        print(x_train.shape)
        x_test = data[20000:24000,...]
        y_test = y_train[20000:24000,...]
        print(x_test.shape)
        print(y_test.shape)
        labels = y_test

        # fn = os.path.join(dir_res, "test_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(x_test, pkl_file)
        # print("Test data are saved as pickle")
        # pkl_file.close

        # fn = os.path.join(dir_res, "test_labels.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(labels, pkl_file)
        # print("Test labels are saved as pickle")
        # pkl_file.close

        ###############################
        # network parameters
        #input_shape = (image_size, image_size, 1)
        batch_size = 64
        kernel_size = 3
        filters = 64
        generic = False
        dense_dim = 128
        latent_dim = 32 # 2 as small as possible
        epochs = 0 # 500  # 0 epochs - good result
        conv_layers = 2
        stride = 2
        activation = 'relu' # 'relu' 'tanh'
        latent_vector = True
        beta_vae = False
        beta = 4
        regularization = False
        project = True

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
        if("beta0.1" in model_name):
            beta = 0.1
        if("beta2" in model_name):
            beta = 2

        convArgs = {
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
            'padding': 'same',
            'data_format': 'channels_last'
        }

        # build encoder model
        inputs = Input(shape=(x_train.shape[1], x_train.shape[2], 1), name='encoded_input')

        #increase capacity of the model: related to model

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

        if (latent_vector == True): # generate the latent vector
            encoded = generate_dense_layers(encoded, dense_dim, encoded_shape, activation, kernel_initializer)

        #latent_dim = 2
        z_mean = Dense(latent_dim, activation=activation, kernel_initializer=kernel_initializer, name='z_mean')(encoded)
        z_log_var = Dense(latent_dim, activation=activation, kernel_initializer=kernel_initializer, name='z_log_var')(encoded)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # Instantiate Encoder Model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        print('Shape:',encoder.layers[-1].output_shape[1:])
        #encoder.summary()

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

        dec_shape = K.int_shape(decoded)
        print(dec_shape) 
        cropWidth, cropHeight = dec_shape[1] - x_train.shape[1], dec_shape[2] - x_train.shape[2]
        print(cropWidth, cropHeight)
        cropLeft, cropTop = int(cropWidth / 2), int(cropHeight / 2)
        cropRight, cropBot = cropWidth - cropLeft, cropHeight - cropTop
        # cropping the output
        decoded = Cropping2D(cropping=((cropLeft, cropRight), (cropTop, cropBot)))(decoded)
        dec_shape = K.int_shape(decoded)
        print(dec_shape)                

        outputs = Activation('linear', name='decoder_output')(decoded) # linear or sigmoid [0,1]
        # sigmoid - slow training? yes, as the derivative is small

        # instantiate decoder model
        decoder = Model(decoded_input, outputs, name='decoder')
        #decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')
        #vae.summary()

        try:
            f = open(model_name)
            vae.load_weights(model_name)
            print("Loaded", model_name, "model from disk")
        except IOError:
            print(model_name, "model not accessible")

        #autoencoder.compile(optimizer='adadelta', loss='mse') #
        lr = 0.0005
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False) # lr=0.005

        # from keras.losses import mse, binary_crossentropy, categorical_crossentropy

        # """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # # E[log P(X|z)]
        # recon = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
        # #recon = recon * x_train.shape[1] * x_train.shape[2]
        # # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        # kl = 0.5 * K.sum( K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1 )
        # vae_loss = recon + kl

        # # mse = mse(K.flatten(inputs), K.flatten(outputs))
        # # reconstruction_loss = mse * x_train.shape[1] * x_train.shape[2] # * for grad updates

        # # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        # # kl_loss = K.sum(kl_loss, axis=-1)
        # # kl_loss *= -0.5

        # # # beta-VAE, use Lagrangian multiplier β under the KKT condition
        # # kl_loss *= 4 # which impact with recon_loss?

        # # vae_loss = K.mean(reconstruction_loss + kl_loss)

        # vae.add_loss(vae_loss)

        # vae.compile(optimizer=adam)

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

        #from keras.callbacks import TensorBoard

        # create validation and test data
        from sklearn.model_selection import train_test_split
        # x_train, x_test = train_test_split(x_train, test_size=0.2, random_state=1)
        # print('train test', x_train.shape, x_test.shape)
        x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=1)
        print('train val', x_train.shape, x_val.shape)

        from keras.callbacks import EarlyStopping
        early_stopping = [EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=2, mode='auto',
                            restore_best_weights=True)]
        
        # train the whole autoencoder
        vae.fit(x_train, #norm - not forget renormalize
                    epochs=epochs,
                    batch_size=batch_size, # batch size & learning rate
                    shuffle=True, verbose=2,
                    #callbacks=early_stopping,
                    validation_data=(x_val, None)) # divide properly
                    #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]))
        
        if(epochs):
            vae.save_weights(model_name)
            print("Saved", model_name, "model weights to disk")

        # Keract visualizations
        #visualize_keract(x_train[:,int(0.3*x_train.shape[1]):,:,:], encoder, decoder)
        #visualize_keras(x_train, encoder, decoder)

        # How convolutional neural networks see the world
        
        test_data = x_test
        print(test_data.shape)

        latent_representation = encoder.predict(test_data)
        encoded_vec = latent_representation[2]
        print('encoded_vec after reparam trick:', encoded_vec.shape) # (batch-size, latent_dim)
        print('encoded_vec max:', encoded_vec.max())
        print('encoded_vec min:', encoded_vec.min())
        plt.suptitle('2D VAE: Latent vectors')
        plt.imshow(encoded_vec)

        decoded_imgs = vae.predict(test_data)
        print('decoded_imgs:', decoded_imgs.shape)
        print('dec max:', decoded_imgs.max())
        print('dec min:', decoded_imgs.min())

        # spherical interpolation in the latent space
        #interpolate(encoded_vec, decoder)

        print("test_data.mean: ", test_data.mean()) # 
        print("test_data.std: ", test_data.std()) #

        print('normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
        print('normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())
        # un-normalize all using data_mean, data_std
        # test_data = test_data * data_std + data_mean
        # encoded_vec = encoded_vec * data_std + data_mean
        # decoded_imgs = decoded_imgs * data_std + data_mean
        # print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
        # print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

        dataset = "mnist" # for pca t-sne vis
        title = '2D VAE: ' # for subtitle

        # draw original and reconstructed data
        draw_orig_reconstr(test_data, decoded_imgs, title, dir_res_model, dataset)

        # test_data = data_test_vis # visualize the original

    if (project == True):
        # project using PCA (then t-sne) and visualize the scatterplot
        #print("PCA projection")
        #pca_projection(encoded_vec, test_data, latent_vector, title, dataset)

        # project using t-sne and visualize the scatterplot
        print("t-SNE projection")
        title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
        #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=20)
        tsne_projection(encoded_vec, test_data, latent_vector, title_tsne, dir_res_model, dataset, labels, perp=30)
        #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=40)

        # project using UMAP and visualize the scatterplot
        print("UMAP projection")
        title_umap = title + 'Latent -> UMAP scatterplot'
        umap_projection(encoded_vec, test_data, latent_vector, title_umap, dir_res_model, dataset, labels)


if __name__ == '__main__':
    main()



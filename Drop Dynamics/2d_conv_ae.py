import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_latent_vector, generate_fully_conn

from pca_projection import pca_projection
from tsne_projection import tsne_projection
from umap_projection import umap_projection

from visualization import visualize_keract, visualize_keras
from latent_sp_interpolation import interpolate, find_interpolatable

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
import time
from progress.bar import Bar
import pickle
import json

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

def load_data():

    start_time = time.time()
    print("Loading data from pickle dump...")
    pkl_file = open("droplet-part.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)
    pkl_file.close

    elapsed_time = time.time() - start_time
    print("All frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")
    print(data.shape)

    data = data[:5000,...]
    print(data.shape)

    return data

def main():

    data = load_data()
    dataset = "droplet" # for pca t-sne umap preprocessing vis

    visualize_data = False
    x_train, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    print(x_train.shape)

    # network parameters
    #input_shape = (image_size, image_size, 1)
    batch_size = 16
    kernel_size = 3
    filters = 64
    generic = False
    latent_dim = 256 # find the best value
    epochs = 0 # 500
    conv_layers = 4
    stride = 2
    latent_vector = True
    project = True
    regularization = False
    dropout_sparcity = False

    model_name = "2d_ae_relu.h5" # val_loss: 0.0139
    # model_name = "2d_ae_lrelu.h5" # val_loss:
    #model_name = "2d_ae_relu_lr.h5" # val_loss:

    # model_name = "2d_ae_relu_reg.h5" # val_loss: 0.0200
    # model_name = "2d_ae_128_relu_reg.h5" # val_loss: 0.0261
    # model_name = "2d_ae_relu_reg_drop.h5" # val_loss: 0.0372
    # model_name = "2d_ae_128_relu_reg_drop.h5" # val_loss: 0.0326

    # model_name = "2d_ae_tanh.h5" # val_loss: 0.0216
    # model_name = "2d_ae_tanh_reg.h5" # val_loss: 0.0183
    # model_name = "2d_ae_128_tanh_reg.h5" # val_loss: 0.0146
    # model_name = "2d_ae_tanh_reg_drop.h5" # val_loss: 0.0345
    # model_name = "2d_ae_128_tanh_reg_drop.h5" # val_loss: 0.0301

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

    if("drop" in model_name):
        dropout_sparcity = True

    if("128" in model_name):
        latent_dim = 128


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
        encoded = BatchNormalization()(encoded)
    # num of conv param: kernel*kernel * channels_in * kernel_num + bias

    # Shape info needed to build Decoder Model
    encoded_shape = K.int_shape(encoded)
    print(encoded_shape)

    if(dropout_sparcity):
        encoded = Dropout(0.5, seed=1)(encoded)

    if (latent_vector == True): # generate the latent vector, apply regularization for AE
        encoded = generate_latent_vector(encoded, latent_dim, encoded_shape, activation, 
                                            kernel_initializer, regularization)
    
    #encoded = BatchNormalization()(encoded) # for relu

    #latent_dim = 256
    #encoded = Dense(latent_dim, activation='tanh', name='latent_vec')(encoded)

    # Instantiate Encoder Model
    encoder = Model(inputs, encoded, name='encoder')
    print('Shape:',encoder.layers[-1].output_shape[1:])
    encoder.summary()

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
        decoded = BatchNormalization()(decoded)

    decoded = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          padding='same')(decoded)

    # Crop the decoded output so that dimensions are equal to the input
    decoded = cropping_output(decoded, x_train.shape)               

    outputs = Activation('linear', name='decoder_output')(decoded) # linear or sigmoid [0,1] tanh [-1,1]
    # sigmoid - slow training? yes, as the derivative is small

    # instantiate decoder model
    decoder = Model(decoded_input, outputs, name='decoder')
    decoder.summary()

    # instantiate Autoencoder model
    outputs = decoder(encoder(inputs))
    #print('output:', outputs.shape)
    autoencoder = Model(inputs, outputs, name='autoencoder')
    autoencoder.summary()
    
    try:
        f = open(model_name)
        autoencoder.load_weights(model_name)
        print("Loaded", model_name, "model from disk")
    except IOError:
        print(model_name, "model not accessible")
    
    #autoencoder.compile(optimizer='adadelta', loss='mse') #
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) # lr=0.005

    from keras.losses import mse, binary_crossentropy, categorical_crossentropy
    # do not scale to [0,1] if using mse loss
    #mse = mse(K.flatten(inputs), K.flatten(outputs[:,:-diff_x,:-diff_y,:]))
    #mse = mse(K.flatten(inputs), K.flatten(outputs[:,:-diff_x,:-diff_y,:]))
    autoencoder.compile(optimizer=adam, loss='mse') # binary_crossentropy if normalized [0,1]  mse
    # K.optimizer.Adam(lr=0.001) 'adadelta'

    # cluster properly, e.g. same turbulency but diff place

    #from keras.callbacks import TensorBoard
    
    # create validation and test data
    from sklearn.model_selection import train_test_split
    x_train, x_test = train_test_split(x_train, test_size=0.1, random_state=1)
    print('train test', x_train.shape, x_test.shape)
    x_train, x_val = train_test_split(x_train, test_size=0.111, random_state=1)
    print('train val', x_train.shape, x_val.shape)

    from keras.callbacks import EarlyStopping
    early_stopping = [EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=2, mode='auto',
                          restore_best_weights=True)]

    # train the whole autoencoder
    autoencoder.fit(x_train, x_train, #norm - not forget renormalize
                    epochs=epochs,
                    batch_size=batch_size, # batch size & learning rate
                    shuffle=True, verbose=2,
                    callbacks=early_stopping,
                    validation_data=(x_val, x_val)) # divide properly
                    #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]))

    if(epochs):
        autoencoder.save_weights(model_name)
        print("Saved", model_name, "model weights to disk")

    # Keract visualizations
    #visualize(x_train, encoder, decoder)

    
    test_data = x_test # x_test x_train

    encoded_vec = encoder.predict(test_data)
    print('encoded_vectors:', encoded_vec.shape) # (batch-size, latent_dim)
    plt.suptitle('2d_ae: Latent vectors')
    plt.imshow(encoded_vec)

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
    # print('normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
    # print('normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())
    test_data = test_data * data_std + data_mean
    #encoded_vec = encoded_vec * data_std + data_mean
    decoded_imgs = decoded_imgs * data_std + data_mean
    # print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
    # print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

    dataset = "droplet" # for pca t-sne vis
    title = '2D_AE: ' # for subtitle

    # draw original and reconstructed data
    draw_orig_reconstr(test_data, decoded_imgs, title)

    if (project == True):
        # project using PCA (then t-sne) and visualize the scatterplot
        #print("PCA projection")
        #pca_projection(encoded_vec, test_data, latent_vector, title, dataset)

        # project using t-sne and visualize the scatterplot
        print("t-SNE projection")
        title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
        #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=20)
        tsne_projection(encoded_vec, test_data, latent_vector, title_tsne, dataset, perp=30)
        #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=40)

        # project using UMAP and visualize the scatterplot
        print("UMAP projection")
        title_umap = title + 'Latent -> UMAP scatterplot, '
        umap_projection(encoded_vec, test_data, latent_vector, title_umap, dataset)


if __name__ == '__main__':
    main()


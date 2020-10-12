# 3D beta-VAE
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
import time
from progress.bar import Bar
import pickle
from sklearn.model_selection import train_test_split

# import tensorflow as tf
# if tf.config.list_physical_devices('GPU'):
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#     tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("physical_devices-------------", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras.layers import Activation, Input, Dense, Conv3D, Conv3DTranspose
from keras.layers import Flatten, Reshape, Cropping3D, Lambda, Dropout
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

    print(data.shape)
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

def brightness_normalization(data):

    for i in range(data.shape[0]):
        data[i] = (data[i] - data[i].mean()) / data[i].std()

    # for i in range(10):
    #     print(data[i].mean(), data[i].std())

    print("All samples were normalized individually!")
    return data

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

    # data = brightness_normalization(data) # test, no effect

    train_shape = int(data_train.shape[0]/3)

    data_train = data[:train_shape,]
    data_test = data[train_shape:,]

    cylinder_names_train = names[:train_shape]
    cylinder_names_test = names[train_shape:]

    # randomize the test samples
    # test_idx = np.random.randint(data_test.shape[0], size=300)
    # print(test_idx)
    # data_test = data_test[test_idx,] # data (training data) shows good results, overfit ?
    # cylinder_names_test_new = []
    # for idx in test_idx:
    #     print(idx)
    #     cylinder_names_test_new.append(cylinder_names_test[idx])
    # cylinder_names_test = cylinder_names_test_new
    # print("Randomized sampling from test data")
    # print(data_test.shape)
    # print(len(cylinder_names_test))

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

    dir_res = "Results/3D_VAE" # directory with all models

    # Load data and subsequently encoded vectors in 2D representation
    # for this save before x_test and encoded vec after tsne and umap
    load_data = True 
    if load_data: 
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

        print('normalized:', x_test.min(), x_test.max())
        test_data_vis = x_test
        test_data_vis = test_data_vis * data_std + data_mean
        x_test = test_data_vis
        print('unnormalized:', x_test.min(), x_test.max())

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

        x_test = (x_test - x_test.mean()) / x_test.std() # back to unnorm

    model_names = {"3d_vae_cropped_128_relu_1.h5", "3d_vae_cropped_128_relu_2.h5", \
    "3d_vae_cropped_128_relu_3.h5", "3d_vae_cropped_128_relu_4.h5", "3d_vae_cropped_128_relu_5.h5", \
    "3d_beta10_vae_cropped_256_relu_1.h5", "3d_beta10_vae_cropped_256_relu_2.h5", \
    "3d_beta10_vae_cropped_256_relu_3.h5", "3d_beta10_vae_cropped_256_relu_4.h5", "3d_beta10_vae_cropped_256_relu_5.h5", \
    "3d_beta100_vae_cropped_256_relu_1.h5", "3d_beta100_vae_cropped_256_relu_2.h5", \
    "3d_beta100_vae_cropped_256_relu_3.h5", "3d_beta100_vae_cropped_256_relu_4.h5", "3d_beta100_vae_cropped_256_relu_5.h5", \
    "3d_vae_cropped_512_relu_1.h5", "3d_vae_cropped_512_relu_2.h5", \
    "3d_vae_cropped_512_relu_3.h5", "3d_vae_cropped_512_relu_4.h5", "3d_vae_cropped_512_relu_5.h5", \
    "3d_beta0.5_vae_cropped_256_relu_1.h5", "3d_beta0.5_vae_cropped_256_relu_2.h5", \
    "3d_beta0.5_vae_cropped_256_relu_3.h5", "3d_beta0.5_vae_cropped_256_relu_4.h5", "3d_beta0.5_vae_cropped_256_relu_5.h5", \
    "3d_beta_vae_cropped_256_relu_1.h5", "3d_beta_vae_cropped_256_relu_2.h5", \
    "3d_beta_vae_cropped_256_relu_3.h5", "3d_beta_vae_cropped_256_relu_4.h5", "3d_beta_vae_cropped_256_relu_5.h5", \
    "3d_beta2_vae_cropped_256_relu_1.h5", "3d_beta2_vae_cropped_256_relu_2.h5", \
    "3d_beta2_vae_cropped_256_relu_3.h5", "3d_beta2_vae_cropped_256_relu_4.h5", "3d_beta2_vae_cropped_256_relu_5.h5", \
    "3d_vae_cropped_256_relu_1.h5", "3d_vae_cropped_256_relu_2.h5", \
    "3d_vae_cropped_256_relu_3.h5", "3d_vae_cropped_256_relu_4.h5", "3d_vae_cropped_256_relu_5.h5"}

    dataset = "flow"
    title = '3D VAE: ' # for subtitle

    # model_names = {"3d_beta_vae_cropped_256_relu_1.h5", "3d_beta_vae_cropped_256_relu_2.h5"}

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
        interpolation = False
        if load_data:
            interpolation = False
        latent_vector = True
        temporal = True

        #load_data = False
        if not load_data:
            # encoded_vec will be predicted with encoder

            # network parameters
            #input_shape = (image_size, image_size, 1)
            batch_size = 16
            kernel_size = (3, 3, 3)
            filters = 64
            generic = False
            dense_dim = 1024 #
            epochs = 0 # 500
            conv_layers = 4
            stride = (3, 2, 2)
            beta_vae = False
            beta = 4

            # Grid search in: latent_dim, activation, regularization
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
            if("beta10" in model_name):
                beta = 10
            if("beta100" in model_name):
                beta = 100

            if("512" in model_name):
                dense_dim = 1024
                latent_dim = 512
            if("256" in model_name):
                dense_dim = 512
                latent_dim = 256
            if("128" in model_name):
                dense_dim = 256
                latent_dim = 128

            
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

            if (latent_vector == True): # generate fully conn layer
                encoded = generate_dense_layers(encoded, dense_dim, encoded_shape, activation, kernel_initializer,
                                                    temporal=temporal)

            # latent_dim = 512 # 512 -> 256, 1024 -> 512
            z_mean = Dense(latent_dim, activation=activation,
                kernel_initializer=kernel_initializer, name='z_mean')(encoded)
            z_log_var = Dense(latent_dim, activation=activation,
                kernel_initializer=kernel_initializer, name='z_log_var')(encoded)

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
                f = open(model_name)
                vae.load_weights(model_name)
                print("Loaded", model_name, "model from disk")
            except IOError:
                print(model_name, "model not accessible")

            #autoencoder.compile(optimizer='adadelta', loss='mse') #
            lr = 0.0005
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

            from keras.losses import mse, binary_crossentropy, categorical_crossentropy
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            # E[log P(X|z)]
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))  # mean is returned
            # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
            #kl = 0.5 * K.sum( K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1 ) # sum
            kl_loss = -0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            #kl_loss = -0.5 * K.sum((1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)), axis=-1) # sum, according to paper

            kld_coeff = 1. / (x_train.shape[2] * x_train.shape[3] / latent_dim)

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
                vae.save_weights(model_name)
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
            #test_data_vis = x_test # vis what the model sees

            latent_representation = encoder.predict(test_data)
            encoded_vec = latent_representation[2]
            print('encoded_vec after reparam trick:', encoded_vec.shape) # (batch-size, latent_dim)
            plt.suptitle('Latent vectors')
            plt.imshow(encoded_vec)
            #plt.plot(encoded_vec)
            plt.savefig('{}/latent.png'.format(dir_res_model), dpi=300)

            decoded_imgs = vae.predict(test_data)
            print('decoded_imgs:', decoded_imgs.shape)
            print('dec max:', decoded_imgs.max())
            print('dec min:', decoded_imgs.min())

            # if(cropping):
            #     test_data = data_test # data_test data_train
            #     print("Cropping was added, but visualizaing full frames")

            # generate new image by sampling a random vector from the latent space
            #sample = encoded_vec[2]
            #print(sample.shape)
            #dec_sample = decoder.predict(sample)
            #generate_new(sample, dec_sample, temporal)


            #print('normalized max:', test_data_vis.max(), encoded_vec.max(), decoded_imgs.max())
            #print('normalized min:', test_data_vis.min(), encoded_vec.min(), decoded_imgs.min())
            # un-normalize all using data_mean, data_std
            #test_data_vis = test_data_vis * data_std + data_mean # already
            #ncoded_vec = encoded_vec * data_std + data_mean
            decoded_imgs = decoded_imgs * data_std + data_mean
            #print('un-normalized max:', test_data_vis.max(), encoded_vec.max(), decoded_imgs.max())
            #print('un-normalized min:', test_data_vis.min(), encoded_vec.min(), decoded_imgs.min())

            # draw original and reconstructed data
            draw_orig_reconstr(test_data_vis, decoded_imgs, title, dir_res_model, dataset, temporal)

        if (project == True):
            # project using PCA (then t-sne) and visualize the scatterplot
            #print("PCA projection")
            #pca_projection(encoded_vec, test_data, latent_vector, title, dataset, names, temporal)

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

        if (interpolation == True):
            # Interpolation in the latent space

            #spherical interpolation
            dec_interpol_sample_slerp, dec_interpol_sample_traversal = interpolate(encoded_vec, decoder, dir_res_model, temporal)

            # find images that are close to the interpolatable
            test_data = x_test # to compare interpolation with original samples
            find_interpolatable(dec_interpol_sample_slerp, test_data, dir_res_model, dataset, temporal)

            #latent dimension traversal with one seed image
            # seed_img = x_test[10] #
            # print(seed_img.shape)
            # seed_img.resize(1, 3, seed_img.shape[1], seed_img.shape[2], 1)
            # latent_representation = encoder.predict(seed_img)
            # encoded_vec = latent_representation[2]

            # latent_dim_traversal(encoded_vec, decoder, dir_res_model, temporal)


if __name__ == '__main__':
    main()



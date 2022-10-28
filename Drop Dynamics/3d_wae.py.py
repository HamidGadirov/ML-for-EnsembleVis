import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from preprocessing import preprocess
from draw_data import draw_data
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn

#from pca_projection import pca_projection
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
from keras.layers import Flatten, Reshape, Cropping2D, Dropout, Cropping3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers

import keras.utils
from keras.models import load_model, Model
from keras.layers import UpSampling3D, AveragePooling3D
from keras.layers import LeakyReLU,Reshape
from keras.models import save_model
import tensorflow as tf

def generateTheta(L,endim):
    # This function generates L random samples from the unit `ndim'-u
    theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,endim))]
    return np.asarray(theta)

# def generateZ(batchsize):
#     # This function generates 2D samples from a `circle' distribution in 
#     # a 2-dimensional space
#     r=np.random.uniform(size=(batchsize))
#     theta=2*np.pi*np.random.uniform(size=(batchsize))
#     x=r*np.cos(theta)
#     y=r*np.sin(theta)
#     z_=np.array([x,y]).T
#     return z_

def generateZ(batchsize,endim):
    # This function generates samples from a uniform distribution in 
    # the `endim'-dimensional space
    z=2*(np.random.uniform(size=(batchsize,endim))-0.5)
    return z

def stitchImages(I,axis=0):
    n,N,M,K=I.shape
    if axis==0:
        img=np.zeros((N*n,M,K))
        for i in range(n):
            img[i*N:(i+1)*N,:,:]=I[i,:,:,:]
    else:
        img=np.zeros((N,M*n,K))
        for i in range(n):
            img[:,i*M:(i+1)*M,:]=I[i,:,:,:]
    return img

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
    
    data_sampled = data[:1800,...] # 3000
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

    data = data[:5400,...] # 5682
    names = names[:5400]
    # 3600 for saved

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

    data = data[:15000] # 15000
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

    # randomize the test samples
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

    # Load data and subsequently encoded vectors in 2D representation
    # for this save before x_test and encoded vec after tsne and umap
    load_data = True
    if load_data: 
        dir_res = "Results/3D_VAE" # directory with all models
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

    dir_res = "Results/3D_WAE" # directory with all models
    dataset = "droplet"
    title = '3D WAE: ' # for subtitle

    # model_names = {"2d_ae_2_relu_reg_norm_1.h5", "2d_ae_2_relu_reg_norm_2.h5", "2d_ae_2_relu_reg_norm_3.h5", \

    mod_nam = {"3d_wae_256_lrelu_reg_norm", "3d_wae_128_lrelu_reg_norm", "3d_wae_64_lrelu_reg_norm"}
    mod_nam = {"3d_wae_32_lrelu_reg_norm"}

    # metrics stability add-on
    # model_names_all = []
    # step = 400
    # for lab in range(400,2400+step, step): # labels to consider
    #     #print(i)

    #     for m_n in mod_nam:
    #         for i in range(5):    
    #             m_n_index = m_n + "_" + str(lab) + "_" + str(i+1) + ".h5"
    #             model_names_all.append(m_n_index)

    # model_names = model_names_all
    # print(model_names)
    ###

    model_names_all = []
    for m_n in mod_nam:
        for i in range(5):    
            m_n_index = m_n + "_" + str(i+1) + ".h5"
            model_names_all.append(m_n_index)

    model_names = model_names_all
    print(model_names)

    # model_names = {"3d_wae_128_lrelu_reg_norm_1.h5"}

    for model_name in model_names:
        print("model_name:", model_name)

        model = model_name[:-5]
        dir_res_m = os.path.join(dir_res, model)
        print("Saved here:", dir_res_m)
        model = model_name[:-3]
        dir_res_model = os.path.join(dir_res_m, model)
        os.makedirs(dir_res_model, exist_ok=True)

        filename = os.path.join(dir_res_model, "model_structure.txt")

        #draw_orig_reconstr(test_data, decoded_imgs, title, dir_res_model, dataset)
        # draw_data(x_test, x_test, dataset, dir_res_model)
        # input()

        project = True
        latent_vector = True

        #load_data = False
        if not load_data:
            # encoded_vec will be predicted with encoder

            temporal = True
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
            stride = (3, 2, 2)
            latent_vector = True
            regularization = False
            dropout_sparcity = False
            denoising = False

            # model_names = {"2d_ae_cropped_128_relu_denoising_norm.h5"}
        
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
                #epochs *= 2
            if("drop" in model_name):
                dropout_sparcity = True
                #epochs *= 2
            if("denoising" in model_name):
                denoising = True
                #epochs *= 2

            interdim=256 # This is the dimension of intermediate latent variable 
                        #(after convolution and before embedding)
            endim=128 # Dimension of the embedding space

            if("256" in model_name):
                dense_dim = 1024
                latent_dim = 256
            if("128" in model_name):
                dense_dim = 256
                latent_dim = 128
            if("64" in model_name):
                dense_dim = 128
                latent_dim = 64
            if("32" in model_name):
                dense_dim = 128
                latent_dim = 32
            if("16" in model_name):
                dense_dim = 128
                latent_dim = 16
            if("8" in model_name):
                dense_dim = 128
                latent_dim = 8
            if("ae_2" in model_name):
                dense_dim = 128
                latent_dim = 2

            # Defining the Encoder/Decoder as Keras graphs
            img=Input((x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)) #Input image 
            embedd=Input((endim,)) #Keras input to Decoder
            depth=16 # This is a design parameter and in fact it is not the depth!
            L=50 # Number of random projections
            batchsize=50

            # Define Encoder
            x=Conv3D(depth*1, (3, 3, 3), padding='same')(img)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            kernel_size = (1, 3, 3)
            x=Conv3D(depth*1, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=AveragePooling3D(stride, padding='same')(x)
            stride = (1, 2, 2)
            x=Conv3D(depth*2, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=Conv3D(depth*2, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=AveragePooling3D(stride, padding='same')(x)
            x=Conv3D(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=Conv3D(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=AveragePooling3D(stride, padding='same')(x)
            x=Conv3D(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=AveragePooling3D(stride, padding='same')(x)
            x=Flatten()(x)
            x=Dense(interdim,activation='relu')(x)
            encoded=Dense(endim)(x)

            encoder=Model(inputs=[img],outputs=[encoded])
            # encoder.summary()

            # Shape info needed to build Decoder Model
            # encoded_shape = K.int_shape(encoded)
            # print(encoded_shape)

            # Define Decoder
            x=Dense(interdim)(embedd)
            # x=Dense(embedd)
            x=Dense(depth*360,activation='relu')(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=Reshape((1,9,10,4*depth))(x) # fix this!
            x=UpSampling3D(stride)(x)
            x=Conv3DTranspose(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=Conv3DTranspose(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=UpSampling3D(stride)(x)

            x=Conv3DTranspose(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=UpSampling3D(stride)(x)
            x=Conv3DTranspose(depth*4, kernel_size, padding='valid')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=Conv3DTranspose(depth*4, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)

            stride = (3, 2, 2)
            x=UpSampling3D(stride)(x)
            x=Conv3DTranspose(depth*2, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            x=Conv3DTranspose(depth*2, kernel_size, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            # x=BatchNormalization(momentum=0.8)(x)
            decoded=Conv3DTranspose(1, (3, 3, 3), padding='same',activation='sigmoid')(x)

            decoded = cropping_output(decoded, x_train.shape)

            decoder=Model(inputs=[embedd],outputs=[decoded])
            # decoder.summary()

            theta=K.variable(generateTheta(L,endim)) #Define a Keras Variable for \theta_ls
            # z=K.variable(generateZ(batchsize)) #Define a Keras Variable for samples of z
            z=K.variable(generateZ(batchsize,endim)) #Define a Keras Variable for samples of z

            # Generate the autoencoder by combining encoder and decoder
            aencoded=encoder(img)
            ae=decoder(aencoded)
            autoencoder=Model(inputs=[img],outputs=[ae])
            # autoencoder.summary()

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
                # for lab in reversed(range(400,2400+step, step)):
                #     # print(lab)
                #     to_remove = "_" + str(lab)
                #     if to_remove in model_name:
                #         x_test_ = x_test[:lab*3,...] # #labels to consider
                #         names_ = names[:lab*3] # #labels to consider
                #         print("Labels considered:", x_test_.shape[0])
                #         model_name = model_name.replace(to_remove, '')
                # print(model_name)
                ###

                dir_model_name = os.path.join("weights", model_name)
                f = open(dir_model_name)
                autoencoder.load_weights(dir_model_name)
                print("Loaded", dir_model_name, "model from disk")
                # continue # skip existing models
            except IOError:
                print(dir_model_name, "model not accessible")
                epochs = 20 # train if no weights found

            # Let projae be the projection of the encoded samples
            projae=K.dot(aencoded,K.transpose(theta))
            # Let projz be the projection of the $q_Z$ samples
            projz=K.dot(z,K.transpose(theta))
            # Calculate the Sliced Wasserstein distance by sorting 
            # the projections and calculating the L2 distance between
            W2=(tf.nn.top_k(tf.transpose(projae),k=batchsize).values-tf.nn.top_k(tf.transpose(projz),k=batchsize).values)**2
            
            w2weight=K.variable(10.0)
            crossEntropyLoss= (1.0)*K.mean(K.binary_crossentropy(K.flatten(img),K.flatten(ae)))
            L1Loss= (1.0)*K.mean(K.abs(K.flatten(img)-K.flatten(ae)))
            W2Loss= w2weight*K.mean(W2)
            # I have a combination of L1 and Cross-Entropy loss for the first term and then 
            # W2 for the second term
            vae_Loss=L1Loss+crossEntropyLoss+W2Loss
            autoencoder.add_loss(vae_Loss) # Add the custom loss to the model

            #Compile the model
            autoencoder.compile(optimizer='rmsprop')

            #autoencoder.compile(optimizer='adadelta', loss='mse') #
            lr = 0.0005
            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False) # lr=0.0005

            loss=[]
            fig1=plt.figure()
            for epoch in range(15):
                ind=np.random.permutation(x_train.shape[0])    
                if epoch>10:
                    K.set_value(w2weight,1.1*K.eval(w2weight))
                for i in range(int(x_train.shape[0]/batchsize)):
                    Xtr=x_train[ind[i*batchsize:(i+1)*batchsize],...]
                    theta_=generateTheta(L,endim)
                    # z_=generateZ(batchsize)
                    z_=generateZ(batchsize,endim)
                    K.set_value(z,z_)
                    K.set_value(theta,theta_)        
                    loss.append(autoencoder.train_on_batch(x=Xtr,y=None))        
                plt.plot(np.asarray(loss))
                # display.clear_output(wait=True)
                # display.display(plt.gcf()) 
                time.sleep(1e-3)

            # # Test autoencoder
            # en=encoder.predict(x_test)# Encode the images
            # dec=decoder.predict(en) # Decode the encodings

            # # Distribution of the encoded samples
            # plt.figure(figsize=(10,10))
            # unique_names, indexed_names = np.unique(names, return_inverse=True)
            # # print(unique_names, indexed_names)
            # colors = []
            # for i in indexed_names:
            #     colors.append('indigo' if i==0 else 'purple' if i==1 else 'orange' if i==2 else 'darkblue' if i==3 \
            #         else 'mediumblue' if i==4 else 'limegreen' if i==5 else 'yellow' if i==6 else 'dodgerblue')
            # plt.scatter(en[:,0],-en[:,1],c=colors, cmap=plt.cm.Spectral)
            # plt.xlim([-1.5,1.5])
            # plt.ylim([-1.5,1.5])
            # plt.show()

            #from keras.callbacks import TensorBoard

            from keras.callbacks import EarlyStopping
            early_stopping = [EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=2, mode='auto',
                                restore_best_weights=True)]

            # # train the whole autoencoder
            # history_callback = autoencoder.fit(x_train, #norm - not forget renormalize
            #                 epochs=epochs,
            #                 batch_size=batch_size, # batch size & learning rate
            #                 shuffle=True, verbose=2,
            #                 callbacks=early_stopping)
                            #validation_data=(x_val))
                            #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]))

            # if(epochs):
            #     autoencoder.save_weights(dir_model_name)
            #     print("Saved", dir_model_name, "model weights to disk")

            #     loss_history = history_callback.history
            #     np_loss_history = np.array(loss_history)
            #     #print(np_loss_history)
            #     #np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
            #     with open(filename, "a") as text_file:
            #         text_file.write("loss_history: ")
            #         text_file.write(str(np_loss_history))
            
            test_data = x_test # x_test x_train
            # train_data = x_train[0:8000]
            train_data = x_train

            # Test autoencoder
            # en=encoder.predict(x_train)# Encode the images
            # dec=decoder.predict(en) # Decode the encodings

            encoded_vec = encoder.predict(test_data)
            print('encoded_vectors:', encoded_vec.shape) # (batch-size, latent_dim)
            # fig=plt.figure()
            # plt.tight_layout()
            # #fig.set_size_inches(8, 6)
            # plt.suptitle('2d_ae: Latent vectors')
            # plt.imshow(encoded_vec)
            # fig.savefig('{}/latent.png'.format(dir_res_model))
            # plt.close(fig)

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
            # normalized max: 3.448445829885598 0.88685006 1.1053748
            # normalized min: -4.12281306460826 0.0 -2.841378
            #  test_data = test_data * data_std + data_mean
            #encoded_vec = encoded_vec * data_std + data_mean
            #  decoded_imgs = decoded_imgs * data_std + data_mean
            # print('un-normalized max:', test_data.max(), encoded_vec.max(), decoded_imgs.max())
            # print('un-normalized min:', test_data.min(), encoded_vec.min(), decoded_imgs.min())

            title = '3D WAE: ' # for subtitle
            # title = 'Raw data ' # for baseline

            # test_data = data_test_vis # visualize the original

            #draw original and reconstructed data
            # draw_orig_reconstr(test_data, decoded_imgs, title, dir_res_model, dataset, temporal)

            # test_data = data_test_vis # visualize the original
            # test_data = x_test
            # encoded_vec = np.zeros((600, 128))

            # train_test_data = 0
            # encoded_vec = 0 # don't need it
            encoded_vec_train = 0 # don't need it
            # encoded_vec_train_test = 0 # don't need it

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
            print("t-SNE projection")
            #title_tsne = title + 'Latent -> t-SNE scatterplot, perp='
            title_tsne = title + '-> t-SNE scatterplot, perp='
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=20)
            # tsne_projection(encoded_vec, test_data, latent_vector, title_tsne, dir_res_model, dataset, names, perp=30)
            #tsne_projection(encoded_vec, test_data, latent_vector, cylinder_names_test, title, perp=40)

            # project using UMAP and visualize the scatterplot
            print("UMAP projection")
            #title_umap = title + 'Latent -> UMAP scatterplot'
            title_umap = title + '-> UMAP scatterplot'
            # umap_projection(encoded_vec, test_data, latent_vector, title_umap, dir_res_model, dataset, names)
            umap_projection(encoded_vec, encoded_vec_train, encoded_vec_train_test, test_data, train_data, train_test_data, latent_vector, title_umap, dir_res_model, dataset, names, temporal=True)

        K.clear_session()

if __name__ == '__main__':
    main()


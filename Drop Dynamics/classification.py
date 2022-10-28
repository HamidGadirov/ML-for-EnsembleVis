"""
Classification with EfficientNet(v2)
"""

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)

# from utils import model_directories, models_metrics_stability, model_name_metrics_stability
from preprocessing import preprocess
from draw_original_reconstruction import draw_orig_reconstr
from fully_conn import generate_dense_layers, generate_fully_conn
from reparameterization_trick import sampling

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
from sklearn.utils import shuffle

import json
from PIL import Image
import torch
from torchvision import transforms

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 

from keras.layers import Activation, Input, Dense, Conv2D, Conv2DTranspose
from keras.layers import Flatten, Reshape, Cropping2D, Lambda, Dropout
# from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers   

import cv2

def visualize(image, name="result.png", save=False):
    fig = plt.figure()
    ax = fig.gca()
    # plt.imshow(image.astype('uint8'))
    plt.imshow(image)
    plt.show()
    if(save):
        res_path = os.path.join('result', name)
        fig.savefig(res_path)

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
    print("All", data.shape[0], "labelled frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

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
    # data_train = load_unlabelled_data() # for train only, 
    data_train = np.empty([0, 160, 224, 1]) # no need for unlabelled at the moment

    data = np.zeros((data_test.shape[0]+data_train.shape[0], 160, 224, 1))
    data[:data_train.shape[0],] = data_train
    data[data_train.shape[0]:,] = data_test

    dataset = "droplet" # for pca t-sne umap preprocessing vis

    visualize_data = False
    data, data_mean, data_std = preprocess(dataset, visualize_data, data) # reshape, visualize, normalize, scale
    print(data.shape)

    data = brightness_normalization(data) # test

    data_train = data[:data_train.shape[0],]
    data_test = data[data_train.shape[0]:,]

    data_test, names = shuffle(data_test, names, random_state=0)
    print("Shuffled test set")
    print(data_test.shape)

    # # cropping:
    # crop_left = int(data.shape[2]*0.1) # start from left
    # crop_right = int(data.shape[2]*0.85) # end at right
    # data = data[:,:,crop_left:crop_right,:]
    # print("data cropped: ", data.shape)

    # # cropping:
    # crop_left = int(data.shape[2]*0.15) # start from left 10 15
    # crop_right = int(data.shape[2]*0.8) # end at right 15 20
    # crop_bottom = int(data.shape[1]*0.83) # remove bottom 10 15 18 17+ 16- 15still 12bad
    # # data = data[:,:,crop_left:crop_right,:]
    # data_train = data_train[:,:,crop_left:crop_right,:]
    # data_train = data_train[:,:crop_bottom,:,:]
    # data_test = data_test[:,:,crop_left:crop_right,:]
    # data_test = data_test[:,:crop_bottom,:,:]
    # print("train set cropped: ", data_train.shape)
    # print("test set cropped:", data_test.shape, len(names))

    # data_test_vis = data_test # unnormalized, uncropped

    # data_train, data_val = train_test_split(data_train, test_size=0.2, random_state=1)
    # print('train & val', data_train.shape, data_val.shape)

    # x_train, x_test, x_val = data_train, data_test, data_val

    # return x_train, x_test, x_val, names, data_mean, data_std, data_test_vis

    return data_test, names

def upscale(data, shape):
    upscaled_data = []
    for i in range(data.shape[0]):
        upscaled_data.append( cv2.resize(data[i,...], (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC) )
    return np.array(upscaled_data)

def to_rgb(data):
    rgb_data = []
    for i in range(data.shape[0]):
        rgb_data.append( cv2.cvtColor(data[i,...].astype('float32'),cv2.COLOR_GRAY2RGB) )
    return np.array(rgb_data)

def build_model(x, num_classes):
    from tensorflow.keras.applications import EfficientNetB0
    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def plot_hist(hist, dir_res):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    # plt.show()
    plt.savefig('{}/training.png'.format(dir_res))
    plt.close()

def main():

    # try:
    #     tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    #     print("Device:", tpu.master())
    #     strategy = tf.distribute.TPUStrategy(tpu)
    # except ValueError:
    #     print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    #     strategy = tf.distribute.MirroredStrategy()

    strategy = tf.distribute.MirroredStrategy()

    dir_res = "Results/2D_EffNet" # directory with all models
    load_pkl_data = True 
    if load_pkl_data: 
        # load test_data from pickle and later encoded_vec_2d
        fn = os.path.join(dir_res, "test_data.pkl")
        with open(fn, 'rb') as pkl_file:
            data_test = pickle.load(pkl_file)
            print("Test data were loaded from pickle")

        # fn = os.path.join(dir_res, "train_data.pkl")
        # pkl_file = open(fn, 'rb')
        # data_train = pickle.load(pkl_file)
        # print("Train data were loaded from pickle")
        # pkl_file.close

        fn = os.path.join(dir_res, "test_labels.pkl")
        with open(fn, 'rb') as pkl_file:
            labels = pickle.load(pkl_file)
            print("Test labels were loaded from pickle")

        test_data = np.asarray(data_test)
        print(test_data.shape)
        # train_data = np.asarray(data_train)
        # print(train_data.shape)

        names = labels
        test_names = names
        print(len(names))

        # train_test_data = np.concatenate((train_data, test_data), axis=0)

        # encoded_vec = 0 # don't need it
        # encoded_vec_train = 0 # don't need it
        # encoded_vec_train_test = 0 # don't need it
    else:
        # preprocess the data and save test subset as pickle
        # x_train, x_test, x_val, names, data_mean, data_std, data_test_vis = load_preprocess()
        test_data, names = load_preprocess()

        # save it only once
        fn = os.path.join(dir_res, "test_data.pkl")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'wb+') as pkl_file:
            pickle.dump(test_data, pkl_file)
            print("Test data were saved as pickle")

        # fn = os.path.join(dir_res, "train_data.pkl")
        # pkl_file = open(fn, 'wb')
        # pickle.dump(x_train, pkl_file)
        # print("Train data were saved as pickle")
        # pkl_file.close

        fn = os.path.join(dir_res, "test_labels.pkl")
        with open(fn, 'wb+') as pkl_file:
            pickle.dump(names, pkl_file)
            print("Test labels were saved as pickle")

    # test_data
    # test_names
    # train_data 

    # img = test_data[0,...]
    shape = (224, 224)
    # visualize(test_data[10, ...])
    test_data = upscale(test_data, shape)
    test_data = np.asarray(tf.expand_dims(test_data, axis=-1))
    print(test_data.shape)
    print(type(test_data))
    # visualize(test_data[10, ...])
    test_data = to_rgb(test_data)
    print(test_data.shape)
    # visualize(test_data[50, ...])

    num_classes = 8

    # def input_preprocess(image, label):
    # label = tf.one_hot(label, NUM_CLASSES)
    # return image, label

    # print(test_names[0:10])
    # test_names[0] = tf.one_hot(test_names[0], num_classes)
    # print(test_names[0])

    # vocab = ["bubble", "bubble-splash", "column", "crown", "crown-splash", "splash", "drop", "none"]

    unique_names, indexed_names = np.unique(names, return_inverse=True)
    # print(unique_names, indexed_names)

    indices = indexed_names
    labels = np.asarray(tf.one_hot(indices, num_classes))  # output: [3 x 3]
    print(labels.shape)

    data_train, data_rest, labels_train, labels_rest = train_test_split(test_data, labels, test_size=0.2, random_state=1)
    print('train & rest', data_train.shape, data_rest.shape)
    data_val, data_test, labels_val, labels_test = train_test_split(data_rest, labels_rest, test_size=0.5, random_state=1)
    print('val & test', data_val.shape, data_test.shape)

    # input("x")

    # # Preprocess image
    # tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    # img = tfms(img).unsqueeze(0)
    # print(img.shape) # torch.Size([1, 3, 224, 224])

    # with strategy.scope():
    #     model = build_model(test_data, num_classes=8)

    # sys.path.insert(0, './efficientnetv2') 
    # import effnetv2_model

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
    #     effnetv2_model.get_model('efficientnetv2-m', include_top=False),
    #     tf.keras.layers.Dropout(rate=0.2),
    #     tf.keras.layers.Dense(num_classes, activation='softmax'),
    # ])
    # model.summary()

    from EfficientNetv2 import effnet_model
    model = effnet_model()

    # lr = 0.0005
    # adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False) # lr=0.0005

    # loss: flow + categorical_crossentropy
    # flow = smoothness + brightness_constancy
    # flow_loss = loss (from get_loss in tvnet)

    # this is wrong!
    # u1_loss = K.mean(u1_np, axis=-1)
    # u2_loss = K.mean(u2_np, axis=-1)
    # flow_loss = u1_loss + u2_loss

    # class_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # multi_task_loss = K.mean(flow_loss + class_loss)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"]) # binary_crossentropy

    model_name = "effnetv2_20.h5"
    dir_model_name = os.path.join("weights", model_name)

    try:
        model.load_weights(dir_model_name)
        print("Loaded", dir_model_name, "model from disk")
    except IOError:
        print(dir_model_name, "model not accessible")
    
    from keras.callbacks import EarlyStopping
    early_stopping = [EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=5,
                        verbose=1, mode='auto',
                        restore_best_weights=True)]

    epochs = 20  # @param {type: "slider", min:8, max:80}
    batch_size = 8
    hist = model.fit(data_train, labels_train,
                    batch_size=batch_size, 
                    epochs=epochs, verbose=1, 
                    callbacks=early_stopping,
                    validation_data=(data_val, labels_val), 
                    shuffle=True)
    plot_hist(hist, dir_res)

    if(epochs):
        model.save_weights(dir_model_name)
        print("Saved model weights to disk")

        # loss_history = history_callback.history
        # np_loss_history = np.array(loss_history)
        # #print(np_loss_history)
        # #np.savetxt("loss_history.txt", np_loss_history, delimiter=",")
        # with open(filename, "a") as text_file:
        #     text_file.write("loss_history: ")
        #     text_file.write(str(np_loss_history))

    loss = model.evaluate(data_test, labels_test,
                    batch_size=batch_size,
                    verbose=1)
    print(loss)
    # 5: 0.84
    # 10: 0.94
    # aug layers: 10 ep: 0.90

    # input("x")

    # from efficientnet_pytorch import EfficientNet
    # model = EfficientNet.from_pretrained('efficientnet-b4')



    # input("x")

    # # Load ImageNet class names
    # labels_map = json.load(open('labels_map.txt'))
    # labels_map = [labels_map[str(i)] for i in range(1000)]

    # # Classify
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(img)

    # # Print predictions
    # print('-----')
    # for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    #     prob = torch.softmax(outputs, dim=1)[0, idx].item()
    #     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

    dataset = "droplet"
    title = '2D VAE: ' # for subtitle

    K.clear_session()

if __name__ == '__main__':
    main()



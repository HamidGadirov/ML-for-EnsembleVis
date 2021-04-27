import numpy as np
import keras.utils
from keras.layers import Input,Dense, Flatten
from keras.models import load_model, Model
from keras.layers import Conv2D, UpSampling2D, AveragePooling2D
from keras.layers import LeakyReLU,Reshape
from keras.datasets import mnist
from keras.models import save_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
# from IPython import display
import time

def generateTheta(L,endim):
    # This function generates L random samples from the unit `ndim'-u
    theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,endim))]
    return np.asarray(theta)
def generateZ(batchsize):
    # This function generates 2D samples from a `circle' distribution in 
    # a 2-dimensional space
    r=np.random.uniform(size=(batchsize))
    theta=2*np.pi*np.random.uniform(size=(batchsize))
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    z_=np.array([x,y]).T
    return z_
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

img=Input((28,28,1)) #Input image 
interdim=128 # This is the dimension of intermediate latent variable 
             #(after convolution and before embedding)
endim=2 # Dimension of the embedding space
embedd=Input((endim,)) #Keras input to Decoder
depth=16 # This is a design parameter and in fact it is not the depth!
L=50 # Number of random projections
batchsize=500

x=Conv2D(depth*1, (3, 3), padding='same')(img)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Conv2D(depth*1, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=AveragePooling2D((2, 2), padding='same')(x)
x=Conv2D(depth*2, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Conv2D(depth*2, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=AveragePooling2D((2, 2), padding='same')(x)
x=Conv2D(depth*4, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Conv2D(depth*4, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=AveragePooling2D((2, 2), padding='same')(x)
x=Flatten()(x)
x=Dense(interdim,activation='relu')(x)
encoded=Dense(endim)(x)

encoder=Model(inputs=[img],outputs=[encoded])
encoder.summary()


x=Dense(interdim)(embedd)
x=Dense(depth*64,activation='relu')(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Reshape((4,4,4*depth))(x)
x=UpSampling2D((2, 2))(x)
x=Conv2D(depth*4, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Conv2D(depth*4, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
x=UpSampling2D((2, 2))(x)
x=Conv2D(depth*4, (3, 3), padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Conv2D(depth*4, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
x=UpSampling2D((2, 2))(x)
x=Conv2D(depth*2, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
x=Conv2D(depth*2, (3, 3), padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
# x=BatchNormalization(momentum=0.8)(x)
# x=BatchNormalization(momentum=0.8)(x)
decoded=Conv2D(1, (3, 3), padding='same',activation='sigmoid')(x)

decoder=Model(inputs=[embedd],outputs=[decoded])
decoder.summary()

theta=K.variable(generateTheta(L,endim)) #Define a Keras Variable for \theta_ls
z=K.variable(generateZ(batchsize)) #Define a Keras Variable for samples of z

# Generate the autoencoder by combining encoder and decoder
aencoded=encoder(img)
ae=decoder(aencoded)
autoencoder=Model(inputs=[img],outputs=[ae])
autoencoder.summary()


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

(x_train,y_train),(x_test,_)=mnist.load_data()
x_train=np.expand_dims(x_train.astype('float32')/255.,3)


loss=[]
fig1=plt.figure()
for epoch in range(30):
    ind=np.random.permutation(x_train.shape[0])    
    if epoch>10:
        K.set_value(w2weight,1.1*K.eval(w2weight))
    for i in range(int(x_train.shape[0]/batchsize)):
        Xtr=x_train[ind[i*batchsize:(i+1)*batchsize],...]
        theta_=generateTheta(L,endim)
        z_=generateZ(batchsize)
        K.set_value(z,z_)
        K.set_value(theta,theta_)        
        loss.append(autoencoder.train_on_batch(x=Xtr,y=None))        
    plt.plot(np.asarray(loss))
    # display.clear_output(wait=True)
    # display.display(plt.gcf()) 
    time.sleep(1e-3)

# Test autoencoder
en=encoder.predict(x_train)# Encode the images
dec=decoder.predict(en) # Decode the encodings

# Distribution of the encoded samples
plt.figure(figsize=(10,10))
plt.scatter(en[:,0],-en[:,1],c=10*y_train, cmap=plt.cm.Spectral)
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.show()









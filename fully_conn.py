from keras.layers import Dense, Reshape
from keras import optimizers, regularizers 

def generate_dense_layers(encoded, dense_dim, encoded_shape, activation, kernel_initializer,
                            latent_dim=0, regularization=False, temporal=False):

    # activation='sigmoid'
    # kernel_initializer='glorot_uniform'

        # Generate the encoded latent vector
    if (temporal):
        encoded = Reshape((encoded_shape[1]*encoded_shape[2]*encoded_shape[3]*encoded_shape[4], ))(encoded)
    else:
        encoded = Reshape((encoded_shape[1]*encoded_shape[2]*encoded_shape[3], ))(encoded)

    # Dense layer for AE/VAE:
    encoded = Dense(dense_dim, activation=activation, kernel_initializer=kernel_initializer, name='dense_layer')(encoded)

    if (latent_dim): # that is for AE:
        if (regularization): # apply l1 l2 to AE but not to VAE
            encoded = Dense(latent_dim, activation=activation, 
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=regularizers.l1(1e-5),
                        activity_regularizer=regularizers.l2(1e-5), name='latent_space')(encoded)
            print("kernel_regularizer l1=1e-5")
            print("activity_regularizer l2=1e-5") 
            # 10e-5 doesnt't work vell, zero vectors in latent space
        else:
            encoded = Dense(latent_dim, activation=activation, 
                        kernel_initializer=kernel_initializer, name='latent_space')(encoded)
            print("No activity_regularizer applied")

    # output dimension of an encoder is (batch-size, latent_dim)
    # for VAE this will return only dense layer
    return encoded

def generate_fully_conn(decoded_input, encoded_shape, activation, kernel_initializer, temporal=False):

    # Fully connected layer of the decoder
    if (temporal):
        decoded = Dense(encoded_shape[1]*encoded_shape[2]*encoded_shape[3]*encoded_shape[4], activation=activation, 
                    kernel_initializer=kernel_initializer)(decoded_input)
        decoded = Reshape((encoded_shape[1], encoded_shape[2], encoded_shape[3], encoded_shape[4]))(decoded)
    else:
        decoded = Dense(encoded_shape[1]*encoded_shape[2]*encoded_shape[3], activation=activation, 
                    kernel_initializer=kernel_initializer)(decoded_input)
        decoded = Reshape((encoded_shape[1], encoded_shape[2], encoded_shape[3]))(decoded)

    #decoded = Dense(filters, activation=activation)(decoded_input)
    # in fully conn only [filters] neurons (or units)
    # num of param: filters*filters + filters
    #print(decoded.shape)
    
    return decoded

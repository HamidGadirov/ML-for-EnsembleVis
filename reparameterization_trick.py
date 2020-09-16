from keras import backend as K

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    #print('batch, dim: ', batch, dim)
    # by default, random_normal has mean=0 and std=1.0
    #epsilon = K.random_normal(shape=(batch, dim), seed=1)
    epsilon = K.random_normal(shape=(batch, dim))
    #print(epsilon)
    reparameterization = z_mean + K.exp(0.5 * z_log_var) * epsilon
    #print(reparameterization)
    return reparameterization

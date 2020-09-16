import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)

from generate import generate_new

"""
# generate new image by sampling a random vector from the latent space
sample = encoded_vec[2]
print(sample.shape)
dec_sample = decoder.predict(sample)
generate_new(sample, dec_sample)
"""

# spherical interpolation
def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def interpolate(encoded_vec, decoder, dir_res_model, temporal=False):
    # spherical interpolation in the latent space

    print(type(encoded_vec)) # list for VAE, numpy.ndarray for AE

    if (isinstance(encoded_vec, list)):
        print('VAE')
        sample = encoded_vec[2]
        print(sample.shape)
    else:
        print('AE')
        sample = encoded_vec
        print(len(encoded_vec))

    vec1 = 12 # 33
    p1 = sample[vec1]
    vec2 = 32 # 37
    p2 = sample[vec2]

    i = 0
    for t in range(10):
        sample[i] = slerp(p1, p2, t/10) 
        i+=1

    """
    sample[0] = sample[vec1]*0.1 + sample[vec2]*0.9
    sample[1] = sample[vec1]*0.3 + sample[vec2]*0.7
    sample[2] = sample[vec1]*0.5 + sample[vec2]*0.5
    sample[3] = sample[vec1]*0.7 + sample[vec2]*0.3
    sample[4] = sample[vec1]*0.9 + sample[vec2]*0.1
    """
    dec_interpol_sample_slerp = decoder.predict(sample[:10]) # sample[:5]
    #generate_new(sample, dec_interpol_sample, temporal)
    
    dec_interpol_sample_traversal = decoder.predict(sample[:7]) # old, no need
    # show the interpolation btw two vectors
    # title = "interpolation"
    # generate_new(sample, dec_interpol_sample_traversal, dir_res_model, title, temporal)

    return dec_interpol_sample_slerp, dec_interpol_sample_traversal

def latent_dim_traversal(encoded_vec, decoder, dir_res_model, dataset, temporal=False):

    print("in latent_dim_traversal")
    
    # latent vector traversal, from "Understanding disentangling in β-VAE"
    # initialise the latent representation by inferring it from a seed image
    # then traverse a single latent dimension (in [−3, 3]), 
    # whilst holding the remaining latent dimensions fixed, and plot the resulting reconstruction

    # However, they had just 20 dim latent space
    # we have 128, mavbe traverse for 10 variables at the same time?
    
    # encoded_vec is (1, 256) for seed image
    print(encoded_vec.shape)
    print("mean: ", encoded_vec[0].mean())
    print("std: ", encoded_vec[0].std())
    # encoded_vec *= 2
    # print("mean: ", encoded_vec[0].mean())
    # print("std: ", encoded_vec[0].std())

    traversed_encoded_vec = []
    traversal = encoded_vec

    for j in range(5): # check all dimensions ?
        traversed_encoded_vec = []
        for i in range(-5, 5, 1):
            #traversal[0][j] = encoded_vec[0][j] + i
            traversal[0][j*10:(j+1)*10] = encoded_vec[0][j*10:(j+1)*10] + i
            traversed_encoded_vec.append(traversal)

        traversed_encoded_vec = np.asarray(traversed_encoded_vec)
        traversed_encoded_vec.resize(traversed_encoded_vec.shape[0], encoded_vec.shape[1]) # e.g. (6, 256)
        print(traversed_encoded_vec.shape)

        dec_interpol_sample_traversal = decoder.predict(traversed_encoded_vec)
        print(dec_interpol_sample_traversal.shape)
        # show the latent_dim_traversal
        #title = "traversal"
        generate_new(encoded_vec, dec_interpol_sample_traversal, dir_res_model, dataset, temporal)

def find_interpolatable(dec_interpol_sample, data, dir_res_model, dataset, temporal=False):
    # compare decoded interpolated sample with data frames to find the closest
    from sklearn.metrics import mean_squared_error
    print(dec_interpol_sample.shape)

    # ? go through all interpolated images and accumulate min_mse ?
    # then compare vae with beta-vae?
    accumulated_min_mse = 0

    if(temporal):
        interpolatable = np.zeros((0, 3, data.shape[2], data.shape[3], 1))
    else:
        interpolatable = np.zeros((0, data.shape[1], data.shape[2], 1))

    # for comparison take the middle image for 3D conv
    for i in range(dec_interpol_sample.shape[0]):
        if(temporal):
            min_mse = mean_squared_error(dec_interpol_sample[i,1,...,0], data[0,1,...,0])
        else:
            min_mse = mean_squared_error(dec_interpol_sample[i,...,0], data[0,...,0])
        index_min_mse = 0
        for j in range(data.shape[0]):
            if(temporal):
                temp_mse = mean_squared_error(dec_interpol_sample[i,1,...,0], data[j,1,...,0])
            else:
                temp_mse = mean_squared_error(dec_interpol_sample[i,...,0], data[j,...,0])
            if (temp_mse < min_mse):
                min_mse = temp_mse
                index_min_mse = j

        print(index_min_mse, min_mse)

        # plt.figure()
        # title = "Found closest interpolatable frame with mse: "
        # title += str(min_mse)
        # plt.suptitle(title)
        # img = data[index_min_mse,:,:,0]
        # plt.imshow(img) 
        # plt.show() 

        interpolatable = np.append(interpolatable, data[index_min_mse:index_min_mse+1], axis=0)

        accumulated_min_mse += min_mse

    print("Total accumulated_min_mse:", accumulated_min_mse)
    # 0.31 for 2D vae droplet
    # 0.27 for 2D ae droplet

    print(interpolatable.shape)

    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    # for flow the data is normalized but I'm visualizing unnormalized,
    # so can't fix the range manually here

    cmap = 'viridis'
    if (dataset == "droplet"):
        cmap = 'gray'

    # draw interpolatable frames
    fig = plt.figure()
    columns = interpolatable.shape[0]
    rows = 2
    for i in range(1, columns +1):

        if(temporal):
            img = dec_interpol_sample[i-1,1,:,:,0]
        else:
            img = dec_interpol_sample[i-1,:,:,0]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        if (dataset == "droplet"):
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax) # manually set the range
        else:
            plt.imshow(img) # manually set the range
            #plt.imshow(img, cmap='viridis', vmin=0, vmax=70)

        if(temporal):
            img = interpolatable[i-1,1,:,:,0]
        else:
            img = interpolatable[i-1,:,:,0]
        fig.add_subplot(rows, columns, i+columns)
        plt.axis('off')
        if (dataset == "droplet"):
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax) # manually set the range
        else:
            plt.imshow(img) # manually set the range
            #plt.imshow(img, cmap='viridis', vmin=0, vmax=70)

        if i == interpolatable.shape[0]:
            break

    # if(temporal):
    #     plt.suptitle('3d_vae: interpolatable frames')
    # else:
    #     plt.suptitle('2d_vae: interpolatable frames')
    plt.suptitle('Interpolatable frames: top - spherical interpolation, bottom - real samples')
    plt.show()
    plt.tight_layout()
    #fig.set_size_inches(8, 6)
    fig.savefig('{}/interpolatable.png'.format(dir_res_model))
    plt.close(fig)
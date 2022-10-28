import os
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)

def generate_new(data, decoded_imgs, dir_res_model, dataset, temporal=False): # to check if the VAE is working properly

    # if (temporal):
    #     model_name="3d_beta-vae"
    # else:
    #     model_name="2d_beta-vae"

    figsize=(decoded_imgs.shape[1], decoded_imgs.shape[2])
    #print(figsize)

    if (dataset == "flow"):
        vmax = 70
    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    print('range of colormap:', vmin, vmax)

    # draw (original) data and (reconstructed) decoded_imgs
    #fig=plt.figure(figsize=figsize)
    fig = plt.figure()
    columns = decoded_imgs.shape[0]
    rows = 1
    for i in range(1, columns +1):
        """
        img = data[i,:,:,0]
        #img.reshape(84,444)
        #print(img.shape)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        """
        if(temporal):
            img = decoded_imgs[i-1,1,:,:,0] # show middle image
        else:
            img = decoded_imgs[i-1,:,:,0]
        #img.reshape(84,444)
        #print(img.shape)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        #plt.imshow(img) # manually set the range
        plt.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
        if i == decoded_imgs.shape[0]:
	        break

    # if(temporal):
    #     plt.suptitle('3d_vae: latent vector traversal')
    # else:
    #     plt.suptitle('2d_vae: latent vector traversal')
    # if (title == "interpolation"):
    #     plt.suptitle('latent vector traversal')
    # elif (title == "traversal"):
    plt.suptitle('latent vector traversal')
    #plt.show()
    plt.tight_layout()
    #fig.set_size_inches(8, 6)
    fig.savefig('{}/latent_vec_traversal.png'.format(dir_res_model))
    plt.close(fig)

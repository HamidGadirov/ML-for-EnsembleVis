import os
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)

def draw_orig_reconstr(data, decoded_imgs, title, dir_res_model, dataset, temporal=False):

    # os.makedirs(model_name, exist_ok=True)
    # filename = os.path.join(model_name, "input-decoder.png")

    vmax = max(data.max(), decoded_imgs.max())
    vmin = min(data.min(), decoded_imgs.min())
    print('range of data:', vmin, vmax)
    if (dataset == "flow"):
        vmax = 70
    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    print('range of colormap:', vmin, vmax)
        
    cmap = 'viridis'
    if (dataset == "droplet"):
        cmap='gray'

    #figsize=(60, 40)
    shift = 20 # just to show nice examples

    # draw (original) data and (reconstructed) decoded_imgs
    fig=plt.figure()
    #fig.set_size_inches(8, 6)
    
    columns = 12
    rows = 2
    for i in range(1, columns +1):
        if (temporal == True):
            img = data[i+shift,1,:,:,0]
        else:
            img = data[i+shift,:,:,0]
        #img.reshape(84,444)
        #print(img.shape)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        # plt.margins(0,0)
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

        if (temporal == True):
            img = decoded_imgs[i+shift,1,:,:,0]
        else:
            img = decoded_imgs[i+shift,:,:,0]
        #img.reshape(84,444)
        #print(img.shape)
        fig.add_subplot(rows, columns, i+columns)
        plt.axis('off')
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        # plt.margins(0,0)
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    title += 'original and reconstructed frames'
    plt.suptitle(title, fontsize=15)    
    #plt.show()
    
    plt.tight_layout()
    #fig.set_size_inches(12, 9)
    #fig.savefig('{}/orig_reconstr.png'.format(dir_res_model), bbox_inches='tight')
    fig.savefig('{}/orig_reconstr.png'.format(dir_res_model), dpi=300)
    plt.close(fig)

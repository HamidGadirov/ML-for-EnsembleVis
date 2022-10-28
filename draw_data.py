import numpy as np
from matplotlib import pyplot as plt

def draw_data(uncropped, cropped, dataset, dir_res_model):
    data = uncropped
    print(data.shape)

    vmax = data.max()
    vmin = data.min()
    print('range of data:', vmin, vmax)
    # if (dataset == "flow"):
    #     vmax = 70
    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    print('range of colormap:', vmin, vmax)
        
    cmap = 'viridis'
    if (dataset == "droplet"):
        cmap='gray'

    fig = plt.figure()
    columns = 7
    rows = 1
    for i in range(1, columns*rows+1 ):
        if (i == data.shape[0]):
            break
        img = data[i,:,:,0]
        #img.reshape(84,444)
        #print(img.shape)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    
    fig = plt.gcf()
    #plt.suptitle('Original data') 
    #plt.colorbar()
    plt.show()
    plt.tight_layout()
    fig.savefig('{}/original.png'.format(dir_res_model), dpi=300)
    plt.close()


    data = cropped
    print(data.shape)

    vmax = data.max()
    vmin = data.min()
    print('range of data:', vmin, vmax)
    # if (dataset == "flow"):
    #     vmax = 70
    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    print('range of colormap:', vmin, vmax)
    
    cmap = 'viridis'
    if (dataset == "droplet"):
        cmap='gray'

    fig = plt.figure()
    columns = 7
    rows = 1
    for i in range(1, columns*rows+1 ):
        if (i == data.shape[0]):
            break
        img = data[i,:,:,0]
        #img.reshape(84,444)
        #print(img.shape)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    
    fig = plt.gcf()
    #plt.suptitle('Original data with cropping') 
    #plt.colorbar()
    plt.tight_layout()
    plt.show()
    fig.savefig('{}/original_crop.png'.format(dir_res_model), dpi=300)
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Flow:
# dirName = '2D_AE/2d_ae_cropped_128_relu'
# dirName = '2D_AE/2d_ae_cropped_128_relu_reg'
# dirName = '2D_AE/2d_ae_cropped_64_relu_reg'

# dirName = '2D_VAE/2d_vae_cropped_128_relu'
# dirName = '2D_VAE/2d_vae_cropped_256_relu'
# dirName = '2D_VAE/2d_beta_vae_cropped_128_relu'
# dirName = '2D_VAE/2d_beta2_vae_cropped_128_relu'
# dirName = '2D_VAE/2d_beta0.5_vae_cropped_128_relu'

# dirName = '3D_AE/3d_ae_cropped_256_relu'
# dirName = '3D_AE/3d_ae_cropped_256_relu_reg'
# dirName = '3D_AE/3d_ae_cropped_512_relu'
# dirName = '3D_AE/3d_ae_cropped_512_relu_reg'

# dirName = '3D_VAE/3d_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_vae_cropped_128_relu'
# dirName = '3D_VAE/3d_beta10_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_beta100_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_beta0.5_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_beta_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_vae_cropped_512_relu'
# dirName = '3D_VAE/3d_beta0.5_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_beta2_vae_cropped_256_relu'
# dirName = '3D_VAE/3d_beta_vae_cropped_256_relu'

def getListOfFiles(dirName):
    # For the given path, get the List of all files in the directory tree 

    # create a list of file and sub directories names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles 

def load_images_from_folder(dirName):
    listOfFiles = getListOfFiles(dirName)
    listOfFiles.sort()
    print(listOfFiles)
    x = 2400
    y = 3000
    images = np.zeros((0, x, y, 4))
    dir_names = []
    for filename in listOfFiles:
        n = filename.split("/")
        key = n[2]
        #if filename.endswith(("latent_umap_scatter_labels.png", "latent_umap.png", "latent_umap.eps")):
        if filename.endswith(("latent_tsne_scatter_labels.png", "latent_tsne.png", "latent_tsne_grid.png", \
                    "latent_umap_scatter_labels.png", "latent_umap.png", "latent_umap_grid.png")):
        #if filename.endswith(("latent.png")):
            #print(filename)
            img = mpimg.imread(filename)
            print(img.shape)
            if (img.shape[0]==x and img.shape[1]==y): # img is not None and 
                img.resize(1, x, y, 4)
                images = np.append(images, img, axis=0)
                dir_names.append(key)

                # if (images.shape[0]==12):
                #     break


    return images, dir_names

def combine_images(dirName):
    combined_img, dir_names = (load_images_from_folder(dirName))
    print((combined_img.shape))

    unique_dir_names = np.unique(dir_names)
    print(unique_dir_names)

    # plt.imshow(combined_img[4])
    # plt.suptitle('latent representations')
    # plt.axis('off')
    # plt.show()

    #fig=plt.figure()

    # iter = 2
    # for k in range(iter):
    columns = 3
    rows = len(unique_dir_names)*2
    print(rows)
    fig, big_axes = plt.subplots(nrows=rows, ncols=columns, sharey=True)
    fig.set_size_inches(12*columns, 9*rows)

    #print(big_axes)

    for row, big_ax in enumerate(big_axes, start=1):
        if ((row-1)%2 == 0):
            #print(big_ax)
            #big_ax[1].set_title("Subplot row %s \n" % row, fontsize=24)
            big_ax[1].set_title(str(unique_dir_names[int((row-1)/2)]), fontsize=24)
            # Turn off axis lines and ticks of the big subplot 
            # obs alpha is 0 in RGBA string!
        #     big_ax[1].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        #     # removes the white frame
        #     big_ax[1]._frameon = False
        
        # for i in range(3):
        #     big_ax[i].tick_params(top='off', bottom='off', left='off', right='off')
        #     big_ax[i]._frameon = False

    #plt.title("New experiment name", fontsize=18)
    for i in range(1, columns*rows+1):
        # if ( (i-1)%6. == 0 ):
        #     print("h")
        #     txt = "this is an example" + str(i)
        #     plt.text(0.05,0.95, txt, transform=fig.transFigure, size=24)
        #     fig.text(txt)
        img = combined_img[i-1]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)
        if i == combined_img.shape[0]:
            break

    #plt.suptitle('latent representations')
    plt.tight_layout()
    #plt.show()
    #fig.savefig("test2.eps", format='eps')
    #fig.savefig("high_res3.png", dpi=300)
    # txt = 'this is an example'
    # plt.text(0.05,0.95, txt, transform=fig.transFigure, size=24)
    filename = dirName.split("/")
    filename = dirName + "/" + filename[1] + ".pdf"
    fig.savefig(filename)

    dir_res = "All_results_table"
    filename = dirName.split("/")
    filename = dir_res + "/" + filename[1] + ".pdf"
    print("Saved:", filename)
    fig.savefig(filename)
    #fig.savefig('{}/latent representations.png'.format(dir_res_model))
    plt.close(fig)

dirNames = {'2D_AE/2d_ae_cropped_128_relu', \
'2D_AE/2d_ae_cropped_128_relu_reg', \
'2D_AE/2d_ae_cropped_64_relu_reg', \
'2D_VAE/2d_vae_cropped_128_relu', \
'2D_VAE/2d_vae_cropped_256_relu', \
'2D_VAE/2d_beta_vae_cropped_128_relu', \
'2D_VAE/2d_beta2_vae_cropped_128_relu', \
'2D_VAE/2d_beta0.5_vae_cropped_128_relu', \
'3D_AE/3d_ae_cropped_256_relu', \
'3D_AE/3d_ae_cropped_256_relu_reg', \
'3D_AE/3d_ae_cropped_512_relu', \
'3D_AE/3d_ae_cropped_512_relu_reg', \
'3D_VAE/3d_vae_cropped_256_relu', \
'3D_VAE/3d_vae_cropped_128_relu', \
'3D_VAE/3d_beta10_vae_cropped_256_relu', \
'3D_VAE/3d_beta100_vae_cropped_256_relu', \
'3D_VAE/3d_beta0.5_vae_cropped_256_relu', \
'3D_VAE/3d_beta_vae_cropped_256_relu', \
'3D_VAE/3d_vae_cropped_512_relu', \
'3D_VAE/3d_beta0.5_vae_cropped_256_relu', \
'3D_VAE/3d_beta2_vae_cropped_256_relu', \
'3D_VAE/3d_beta_vae_cropped_256_relu'}

dirNames = {'Baseline/baseline_crop'}
dirNames = {'3D_AE/3d_ae_cropped_256_relu_reg'}

for dirName in dirNames:
    print(dirName)
    combine_images(dirName)
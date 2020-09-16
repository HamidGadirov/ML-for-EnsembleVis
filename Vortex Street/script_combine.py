import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def getListOfFiles(dirName):
    # For the given path, get the List of all files in the directory tree 

    # create a list of file and sub directories 
    # names in the given directory 
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

def load_images_from_folder():
    dirName = '2D_AE' # sampled-300
    listOfFiles = getListOfFiles(dirName)
    listOfFiles.sort()
    #print(listOfFiles)
    images = np.zeros((0, 437, 515, 4))
    for filename in listOfFiles: 
        if filename.endswith(("latent_umap_scatter_labels.png", "latent_umap.png")):
        #if filename.endswith(("latent.png")):
            #print(filename)
            img = mpimg.imread(filename)
            print(img.shape)
            img.resize(1, 437, 515, 4)
            if img is not None:
                images = np.append(images, img, axis=0)

    return images

combined_img = (load_images_from_folder())
print((combined_img.shape))

# plt.imshow(combined_img[4])
# plt.suptitle('latent representations')
# plt.axis('off')
# plt.show()

fig=plt.figure()
columns = 4
rows = 3
for i in range(1, columns*rows+1):
    img = combined_img[i-1]
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img)
    if i == combined_img.shape[0]:
        break

plt.suptitle('latent representations')
plt.show()
fig.savefig("test.png", bbox_inches='tight')
#fig.savefig('{}/latent representations.png'.format(dir_res_model))

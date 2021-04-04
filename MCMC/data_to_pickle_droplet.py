# load all data and save as a serialized pickle

# try numpy.save

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
import time
from progress.bar import Bar
import pickle

import numpy as np
from matplotlib import pyplot as plt

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

def func(path):
    if os.path.isdir(path):
        d = {}
        for name in os.listdir(path):
            d[name] = func(os.path.join(path, name))
    else:
        d = os.path.getsize(path)
    return d

def main():
    
    dirName = 'mcmc_a.raw' # 'droplet' # sampled-300
    start_time = time.time()

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    data = np.zeros((0, 160, 224, 1))
    print(data.size)
    count = 0

    # Print the files
    with Bar("Loading the data...", max=len(listOfFiles)/2 ) as bar:
        for elem in listOfFiles:
            if elem.endswith(('.bna')):
            #     if ("248_Hexa-Hexa_0,51mm_viewA" in elem): 
            #         # 248_Hexa-Hexa_0,51mm_viewA problematic
            #         print(elem)
            #         tmp_data = np.fromfile(elem, dtype='uint8')
            #         print(tmp_data.shape)
            #         frame_num = tmp_data.shape[0]/(160*224)
            #         #print(frame_num)
            #         tmp_data.resize(int(frame_num), 160, 224, 1)
            #         tmp_data = np.flip(tmp_data, 1)
            #         data = np.append(data, tmp_data, axis=0)
            #         print(data.shape)

            #         fig=plt.figure()
            #         columns = 10
            #         rows = 8
            #         for i in range(1, columns*rows+1 ):
            #             if (i == data.shape[0]):
            #                 break
            #             img = data[i,:,:,0]
            #             #img.reshape(84,444)
            #             #print(img.shape)
            #             fig.add_subplot(rows, columns, i)
            #             plt.imshow(img)
            #             plt.axis('off')

            #         fig = plt.gcf()
            #         plt.suptitle(elem) 
            #         plt.axis('off')
            #         plt.show()

            #         input("prompt")
            #     else:
            #         continue

                # input("prompt")
                tmp_data = np.fromfile(elem, dtype='uint8')
                print(tmp_data.shape)
                frame_num = tmp_data.shape[0]/(160*224)
                #print(frame_num)
                tmp_data.resize(int(frame_num), 160, 224, 1)
                tmp_data = np.flip(tmp_data, 1)
                data = np.append(data, tmp_data, axis=0)
                #print(elem)
                #print(tmp_data)

                count += 1
                #if (count==40): # ~5K frames
                #    break
                bar.next()

    #print(data)
    #print(tmp_data)
    #print(tmp_data.shape)
    #print(data.size)
    print('data shape:', data.shape)
    print(count)


    elapsed_time = time.time() - start_time
    print("All", count, "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")
    
    pkl_file = open("droplet.pkl", 'wb')
    pickle.dump(data, pkl_file, protocol=4)
    pkl_file.close

    pkl_file = open("droplet.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)

    print(data.shape)


if __name__ == '__main__':
    main()

	

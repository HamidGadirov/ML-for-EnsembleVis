# load all data and save as a serialized pickle
# numpy.save might be used as well

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
import time
from progress.bar import Bar
import pickle
import re

def getListOfFiles(dirName):
    # For the given path, get the list of all files in the directory tree
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles  

import json
with open('feature-metadata-vortex.json', 'r') as f:
    labels = json.load(f)

def main():
    
    dirName = 'sampled-300-2'
    start_time = time.time()

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    # #print(listOfFiles)
    # for k in range(30):
    #     print(listOfFiles[k])
    listOfFiles.sort()
    print("listOfFiles is sorted!")
    # input("waiting...")

    data = np.zeros((0, 441, 84, 1))
    print(data.size)
    count = 0
    cylinder_names = [] # list for labels
    maxNumImages = 2000
    with Bar("Loading the data...", max=maxNumImages) as bar:
        for elem in listOfFiles: 
            # get all .raw files while neglecting empty frames in the beginning of each member
            if re.search("(5[0-3]|4[0-9]|3[0-9]|2[0-9]|1[8-9])\.raw$", elem):
                
                tmp_data = np.fromfile(elem, dtype='uint8')
                tmp_data.resize(1, 441, 84, 1)

                """ skip labels from json file """

                n = elem.split("/")
                k = n[1] # key 
                v = n[2] # value

                num = int(v[8:10])
                print("\n", num)
                try: # check if k is in labels from json file
                    print("labels[k]:", labels[k])
                    try: # check if features is non-empty
                        print("range:", labels[k]["features"][0]["range"])
                    except IndexError:
                        print("features is empty")
                        label = labels[k]["default"] # l
                        print(label) # l by default
                        continue
                    
                    if num >= labels[k]["features"][0]["range"][0]:
                        label = labels[k]["features"][0]["name"]
                        print(label) # t for turbulent
                    else:
                        label = 'l'

                    continue # don't add labelled data into training set

                except KeyError:
                    print("not in the json file") # not in the json file, no label
                    label = " "

                """ end of labels processing part """

                data = np.append(data, tmp_data, axis=0)
                cylinder_names.append(elem)
                #cylinder_names.append(elem + " " + label)

                if(count==0): print(cylinder_names)

                count += 1
                if (count==maxNumImages): # load a subset of the dataset
                   break
                bar.next()

    print(len(cylinder_names))
    print(cylinder_names)
    #print(data)
    #print(tmp_data)
    print('data shape:', data.shape)
    #print(count)

    elapsed_time = time.time() - start_time
    print("All", count, "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    # input("waiting...")
    
    pkl_file = open("sampled-300_unlabelled_data.pkl", 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close

    pkl_file = open("sampled-300_unlabelled_names.pkl", 'wb')
    pickle.dump(cylinder_names, pkl_file)
    pkl_file.close


    pkl_file = open("sampled-300_unlabelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)

    pkl_file = open("sampled-300_unlabelled_names.pkl", 'rb')
    names = []
    names = pickle.load(pkl_file)

    for k in range(50):
        print(names[k])

    print(data.shape)
    print(len(names))
    print(names[0])


if __name__ == '__main__':
    main()

	

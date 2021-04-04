# load ONLY LABELLED data and save as a serialized pickle

# try numpy.save

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
import time
from progress.bar import Bar
import pickle

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

import json
with open('droplet-25-t10-feature-metadata-hamid.json', 'r') as f:
    labels = json.load(f)

def main():
    
    dirName = 'sampled-1000-x2-t10' # droplet
    start_time = time.time()

     # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    listOfFiles.sort()
    print("listOfFiles is sorted!")
    # for k in range(10):
    #     print(listOfFiles[k])
    # input("input: ")

    data = np.zeros((0, 160, 224, 1))
    print(data.size)
    names = []
    frame_names = []
    count = 0

    # Print the files
    with Bar("Loading the data...", max=len(listOfFiles)/2 ) as bar:
        for elem in listOfFiles:
            if elem.endswith(('.bna')):

                n = elem.split("/")
                #print("n:", n)
                key = n[0] # key 
                value = n[1] # value 
                value = value[:-4]
                print(value)

                tmp_names = []

                try: # check if k is in labels from json file
                    print("labels[value]:", labels[value])
                    frame_names.append(value)

                    len_of_features = len(labels[value]["features"])

                    for i in range (labels[value]["shape"][0]):
                        is_label = False
                        for f in range (len_of_features):
                            # null -> till the end
                            range_max = labels[value]["features"][f]["range"][1]
                            if range_max is None: # null, i.e. till the end: labels[value]["shape"][0]
                                range_max = labels[value]["shape"][0]
                            if(i >= labels[value]["features"][f]["range"][0] and i <= range_max):
                                label = labels[value]["features"][f]["name"]
                                tmp_names.append(label)
                                print(label)
                                is_label = True
                                break
                        if (is_label == False):
                            label = labels[value]["default"] # "none"
                            tmp_names.append(label)
                        print(label)
                        print(i)

                                      
                    tmp_data = np.fromfile(elem, dtype='uint8')
                    print(tmp_data.shape)
                    frame_num = tmp_data.shape[0]/(160*224)
                    print(frame_num)
                    tmp_data.resize(int(frame_num), 160, 224, 1)
                    # select dividible to 3 range of frame_num
                    a = int(tmp_data.shape[0]/3)
                    dividible_range = a*3
                    start_from = tmp_data.shape[0] - dividible_range
                    print(start_from)
                    tmp_data = tmp_data[start_from:,...] # for 3d preprocessing
                    tmp_data = np.flip(tmp_data, 1)
                    data = np.append(data, tmp_data, axis=0)

                    # names and data must be consistent
                    tmp_names = tmp_names[start_from:] # for 3d preprocessing
                    names.extend(tmp_names)

                    #print(elem)
                    #print(tmp_data)

                    print("data:", data.shape[0])
                    print("names:", len(names))

                    count += 1
                    # if (count==30): # ~5K frames
                    #    break
                    bar.next()
                except KeyError:
                    print("not in the json file") # not in the json file, no label


    print(frame_names)
    #print(data)
    #print(tmp_data)
    #print(tmp_data.shape)
    #print(data.size)
    print('data shape:', data.shape)
    print(count)

    elapsed_time = time.time() - start_time
    print("All", count, "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")
    
    pkl_file = open("droplet_labelled_data_2.pkl", 'wb')
    pickle.dump(data, pkl_file, protocol=4)
    pkl_file.close

    pkl_file = open("droplet_labelled_data_2.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)

    print(data.shape)
    
    pkl_file = open("droplet_labelled_names_2.pkl", 'wb')
    pickle.dump(names, pkl_file, protocol=4)
    pkl_file.close

    pkl_file = open("droplet_labelled_names_2.pkl", 'rb')
    names = []
    names = pickle.load(pkl_file)

    print(len(names))
    print(names)


if __name__ == '__main__':
    main()


	

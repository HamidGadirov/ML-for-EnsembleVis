# load all data and save as a serialized pickle

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
with open('feature-metadata.json', 'r') as f:
    labels = json.load(f)

def main():
    
    dirName = 'sampled-300-2' # sampled-300
    start_time = time.time()

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    # #print(listOfFiles)
    # for k in range(30):
    #     print(listOfFiles[k])
    listOfFiles.sort()
    print("listOfFiles is sorted!")
    # for k in range(300):
    #     print(listOfFiles[k])
    # input("zoom: ")

    data = np.zeros((0, 441, 84, 1))
    print(data.size)
    count = 0
    cylinder_names = [] # list for labels
    # Print the files
    with Bar("Loading the data...", max=len(listOfFiles)*(54-18)/54) as bar:
        for elem in listOfFiles: 
            if elem.endswith(('53.raw','52.raw','51.raw','50.raw','49.raw','48.raw','47.raw','46.raw','45.raw', \
            '44.raw','43.raw','42.raw','41.raw','40.raw','39.raw','38.raw','37.raw','36.raw','35.raw','34.raw','33.raw', \
            '32.raw','31.raw','30.raw','29.raw','28.raw','27.raw','26.raw','25.raw','24.raw','23.raw','22.raw','21.raw', \
            '20.raw','19.raw','18.raw',)):
                
                tmp_data = np.fromfile(elem, dtype='uint8')
                #print(tmp_data.shape)
                tmp_data.resize(1, 441, 84, 1)
                #data = np.append(data, tmp_data, axis=0)
                #print(elem)
                #print(tmp_data)

                ############ add labels from json file

                n = elem.split("/")
                k = n[1] # key 
                v = n[2] # value

                #try:
                num = int(v[8:10])
                print("\n", num)
                try: # check if k is in labels from json file
                    print("labels[k]:", labels[k])
                    try: # check if features is non-empty
                        print("range:", labels[k]["features"][0]["range"])
                    except IndexError:
                        print("features is empty")
                        #labelled_names.append(k + v + labels[k]["default"])
                        #count += 1
                        label = labels[k]["default"] # l
                        #cylinder_names.append(elem + " " + label)
                        print(label) # l by default ??
                        continue
                    
                    if num >= labels[k]["features"][0]["range"][0]:
                        #labelled_names.append(k + v + labels[k]["features"][0]["name"])
                        #count += 1
                        label = labels[k]["features"][0]["name"]
                        print(label) # t for turbulent
                    else:
                        label = 'l'

                    continue # don't add labelled data into training set

                except KeyError:
                    print("not in the json file") # not in the json file, no label
                    label = " "
                #except ValueError:
                #    print("invalid literal for int()")

                #############

                data = np.append(data, tmp_data, axis=0)
                cylinder_names.append(elem)
                #cylinder_names.append(elem + " " + label)

                if(count==0): print(cylinder_names)

                count += 1
                #if (count==6000):
                #    break
                bar.next()

    print(len(cylinder_names))
    print(cylinder_names)
    #print(data)
    #print(tmp_data)
    #print(tmp_data.shape)
    #print(data.size)
    print('data shape:', data.shape)
    #print(count)

    #input("zoom: ")

    elapsed_time = time.time() - start_time
    print("All", count, "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")
    
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
    """
    d = func(dirName)
    
    pkl_file = open("sampled-300_dict.pkl", 'wb')
    pickle.dump(d, pkl_file)
    pkl_file.close

    pkl_file = open("sampled-300_dict.pkl", 'rb')
    d = {}
    d = pickle.load(pkl_file)
    """
    """
    pkl_file = open("data.pkl", 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close

    np.save('data.npy', data)
    print("saved")
    """

    for k in range(50):
        print(names[k])

    print(data.shape)
    print(len(names))
    print(names[0])
    #print(d.values)


if __name__ == '__main__':
    main()

	

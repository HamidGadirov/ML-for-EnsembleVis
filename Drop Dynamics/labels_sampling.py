# load ONLY LABELLED data and save as a serialized pickle

# try numpy.save

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
import time
from progress.bar import Bar
import pickle
import random

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

    load_save_rand = False
    display = False
    save_names = False
    show_names_data = True

    if (load_save_rand):
        dirName = 'sampled-1000-x2-t10' # droplet
        start_time = time.time()

        # Get the list of all files in directory tree at given path
        listOfFiles = getListOfFiles(dirName)
        listOfFiles.sort()
        print("listOfFiles is sorted!")
        print(len(listOfFiles))
        # for k in range(10):
        #     print(listOfFiles[k])

        allMembers = []
        for elem in listOfFiles:
            if elem.endswith(('.bna')):
                allMembers.append(elem)
        n = len(allMembers)
        print(n)

        # f = open("labels_file.txt", "a")
        all_labels = []

        data = np.zeros((0, 160, 224, 1))

        for _ in range(1000):
            # randomly select n
            select_m = random.randint(0, n-1)
            # print(select_m)
            # print(allMembers[select_m])

            # get time steps of the member
            tmp_data = np.fromfile(allMembers[select_m], dtype='uint8')
            # print(tmp_data.shape)
            frame_num = tmp_data.shape[0]/(160*224)
            # print(frame_num)
            tmp_data.resize(int(frame_num), 160, 224, 1)
            tmp_data = np.flip(tmp_data, 1)

            # select random num of time step
            select_t_s = random.randint(2, frame_num-2)
            # print(select_t_s)
            # fig=plt.figure()
            # img = tmp_data[select_t_s,:,:,0]

            # plt.imshow(img)
            # plt.axis('off')

            # fig = plt.gcf()
            # name = allMembers[select].split("/")
            # title = str(name[1]) + ", frame " + str(select_t_s)
            # plt.suptitle(title) 
            # plt.axis('off')
            # plt.show()

            all_labels.append(str(allMembers[select_m]) + ", frame " + str(select_t_s) + ", ")

            # print(tmp_data.shape)
            data = np.append(data, tmp_data[select_t_s-1:select_t_s+2,...], axis=0)


        print(all_labels)
        print(data.shape)

        with open('labels_file.txt', 'w') as f:
            for item in all_labels:
                f.write("%s\n" % item)

        pkl_file = open("droplet_sampled_labelled_data.pkl", 'wb')
        pickle.dump(data, pkl_file, protocol=4)
        pkl_file.close

    if (display):
        pkl_file = open("droplet_sampled_labelled_data.pkl", 'rb')
        data = []
        data = pickle.load(pkl_file)

        print(data.shape)

        # fig=plt.figure()
        # img = data[1,:,:,0]
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

        fig=plt.figure()
        columns = 9
        rows = 8
        for i in range(1, columns*rows+1 ):
            if (i == data.shape[0]):
                break
            order = (i+560)*3+1 # 1 4 7 ... 217
            img = data[order,:,:,0]
            #img.reshape(84,444)
            #print(img.shape)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.axis('off')

        fig = plt.gcf()
        plt.suptitle(order) 
        plt.axis('off')
        plt.show()

        # f = open("labels_file.txt", "a")
        # f.write(all_labels)
        # f.close()

    if (save_names):
        names = [] # names of sampled labels
        count = 0
        with open('labels_file.txt', 'r') as f:
            for line in f:
                fields = line.split(", ")
                name = fields[2][:-1]
                # print(name)
                for _ in range(3):
                    names.append(name)
                count += 1
                if (count == 600):
                    break

        # print(names[590:])

        pkl_file = open("droplet_sampled_labelled_names.pkl", 'wb')
        pickle.dump(names, pkl_file, protocol=4)
        pkl_file.close

    if (show_names_data):
        pkl_file = open("droplet_sampled_labelled_data.pkl", 'rb')
        data = []
        data = pickle.load(pkl_file)
        print(data.shape)

        pkl_file = open("droplet_sampled_labelled_names.pkl", 'rb')
        names = []
        names = pickle.load(pkl_file)
        print(len(names))

        # for i in range(30):
        print(names[400:472])

        fig=plt.figure()
        columns = 9
        rows = 8
        for i in range(1, columns*rows+1 ):
            if (i == data.shape[0]):
                break
            # order = (i+560)*3+1 # 1 4 7 ... 217
            img = data[i+400,:,:,0]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.axis('off')

        fig = plt.gcf()
        plt.suptitle("400+") 
        plt.axis('off')
        plt.show()





    input("input: ")

    data = np.zeros((0, 160, 224, 1))
    print(data.size)
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

                try: # check if k is in labels from json file
                    print("labels[value]:", labels[value])

                    # len_of_features = len(labels[value]["features"])

                    # for i in range (labels[value]["shape"][0]):
                    #     is_label = False
                    #     for f in range (len_of_features):
                    #         # null -> till the end
                    #         range_max = labels[value]["features"][f]["range"][1]
                    #         if range_max is None: # null, i.e. labels[value]["shape"][0]
                    #             range_max = labels[value]["shape"][0]
                    #         if(i >= labels[value]["features"][f]["range"][0] and i <= range_max):
                    #             label = labels[value]["features"][f]["name"]
                    #             names.append(label)
                    #             print(label)
                    #             is_label = True
                    #             break
                    #     if (is_label == False):
                    #         label = labels[value]["default"] # "none"
                    #         names.append(label)
                    #     print(label)
                    #     print(i)

                                      

                except KeyError:
                    print("not in the json file") # not in the json file, no label

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
                    #print(elem)
                    #print(tmp_data)

                    count += 1
                    if (count==160): # ~22K frames
                       break
                    bar.next()



    #print(data)
    #print(tmp_data)
    #print(tmp_data.shape)
    #print(data.size)
    print('data shape:', data.shape)
    print(count)


    elapsed_time = time.time() - start_time
    print("All", count, "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")
    
    pkl_file = open("droplet_unlabelled_data.pkl", 'wb')
    pickle.dump(data, pkl_file, protocol=4)
    pkl_file.close

    pkl_file = open("droplet_unlabelled_data.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)

    print(data.shape)


if __name__ == '__main__':
    main()


	
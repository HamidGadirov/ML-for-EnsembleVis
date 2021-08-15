import numpy as np
from matplotlib import pyplot as plt
import pickle

def preprocess_2d(data, visualize):
    # reshape, visualize, normalize, scale

    # for i in range(10):
    #     plt.figure()
    #     img = data[i,...,0]
    #     plt.imshow(img) 
    #     plt.show() 

    figsize=(40, 60)

    if (visualize == True):
        # visualize all data
        fig=plt.figure(figsize=figsize)
        columns = 15
        rows = 6
        for i in range(1, columns*rows+1 ):
            if (i == data.shape[0]):
                break
            img = data[i,:,:,0]
            #img -= 80
            #img.reshape(84,444)
            #print(img.shape)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='viridis')
            
        fig = plt.gcf()
        plt.suptitle('Original data') 
        plt.show()
    
    #normalize data - subtract mean and std dev - that is standart procedure
    x_train = data
    #x_train = data[6600:,]
    #print("unnormalized:", x_train)

    print(x_train.max())
    print(x_train.min())

    # normalize to zero mean and unit variance
    data_mean = x_train.mean()
    data_std = x_train.std()
    x_train = (x_train - x_train.mean()) / x_train.std()
    #print("normalized:", x_train)
    print(x_train.max())
    print(x_train.min())
    print(x_train.mean())
    print(x_train.std())
    """
    # scale to range [0, 1] -- this slows training? bad result
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    #print(x_train)
    print(x_train.max())
    print(x_train.min())
    print(x_train.mean())
    print(x_train.std())
    """
    if (visualize == True):
        # visualize all normalized and scaled data
        fig=plt.figure(figsize=figsize)
        columns = 15
        rows = 6
        for i in range(1, columns*rows+1):
            if (i == x_train.shape[0]):
                break
            img = x_train[i,:,:,0]
            #img.reshape(84,444)
            #print(img.shape)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='viridis', vmin=x_train.min(), vmax=x_train.max())
        
        fig = plt.gcf()
        plt.suptitle('Normalized data') 
        plt.show()
    
    return x_train, data_mean, data_std

def preprocess_3d(dataset, visualize, data, names=""):
    print('Preprocessing for 3D convs')
    # reshape, visualize, normalize, scale

    # plt.figure()
    # img = data[43,:,:,0]
    # plt.imshow(img) 
    # plt.show() 

    #if (dataset == "flow"):

    # for k in range(10):
    #     print(names[k])

    # proper 3D labeling is done only for test set here!

    # get starting position of diff members
    pkl_file = open("start_of_diff_member.pkl", 'rb')
    start_of_diff_members = np.asarray(pickle.load(pkl_file))
    print("Starting position of diff members were loaded")
    # start_of_diff_members = np.insert(start_of_diff_members, 0, 1) # add 1
    # print(start_of_diff_members)
    # input("X")

    # Order: 5400 - series, sliding window style; 1800 - sampled

    count = data.shape[0]
    print("total num:", count)

    timestep = 0
    i = 0
    names_timesteps = []
    # data_timesteps = np.zeros(( int(count/3), 3, data.shape[1], data.shape[2], 1 ))
    data_timesteps = np.zeros(( count, 3, data.shape[1], data.shape[2], 1 ))
    test_size = 7200 # 7200

    # train set:
    for k in range(count-test_size):
        data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
        timestep += 1
        if (timestep%3 == 0):
            i += 1
            timestep = 0
            #print(i)
            names_timesteps.append(names[k]) # these are just empty
            #print(len(names_timesteps))
            # print(k)

    print(i) # 5000
    print('data with timesteps: ', data_timesteps.shape)
    print('names with timesteps: ', len(names_timesteps))

    # pad the first and the last timestep
    # 5400 test set:
    timestep = 0
    start_of_diff_members_ind = 0
    sampled_size = 1800 # 0 or 1800
    for k in range(count-test_size, count-sampled_size):
        # print("k:", k)
        # if k-(count-test_size) == 0: # begnning
        #     data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
        #     timestep += 1
        #     print("beginning")

        # if k-(count-test_size) == start_of_diff_members[start_of_diff_members_ind-1]: # start_of_diff_members
        #     data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
        #     timestep += 1
        #     print("start_of_diff_members")
        #     print(timestep)

        if k-(count-test_size) == 0 or k-(count-test_size) == start_of_diff_members[start_of_diff_members_ind-1]:
            # begin_of_diff_members
            data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
            timestep += 1
        else:
            data_timesteps[i,timestep,:,:,:] = data[k-1,:,:,:]
            timestep += 1

        data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
        timestep += 1

        if k-(count-test_size) == start_of_diff_members[start_of_diff_members_ind]-1: # end_of_diff_members
            data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
            timestep += 1
            start_of_diff_members_ind += 1
            # print("end_of_diff_members")
            # print(timestep) # always 3
        else: 
            data_timesteps[i,timestep,:,:,:] = data[k+1,:,:,:]
            timestep += 1

        # print(timestep)
        if (timestep%3 == 0):
            i += 1
            timestep = 0
            #print(i)
            names_timesteps.append(names[k]) # middle label
            #print(len(names_timesteps))

    # input("X")
    print(i) # 5000+5400
    print('data with timesteps: ', data_timesteps.shape)
    print('names with timesteps: ', len(names_timesteps))

    # sampled labels must be used, otherwise no column classes
    # 1800 sampled test set:
    step = 3
    for k in range(count-sampled_size+1, count+1, step): # k starts in middle always
        # for _ in range(3): # repeat labels 3 times
            for timestep in range(3):
                data_timesteps[i,timestep,:,:,:] = data[k-1+timestep,:,:,:] # data[-1,0,1]
                # print(names[k-1+timestep])
                # plt.figure()
                # img = data_timesteps[i,timestep,:,:,:]
                # plt.imshow(img) 
                # plt.show() 
            # print("End of loop")
            i += 1
            names_timesteps.append(names[k]) # middle label
            # print(names[k])

    data_timesteps = data_timesteps[:i]

    print("i:", i)
    print('data with timesteps: ', data_timesteps.shape)
    #print(i, timestep)
    print('names with timesteps: ', len(names_timesteps))

    if (visualize == True):
        # visualize all data
        fig=plt.figure(figsize=(20, 100))
        columns = 15
        rows = 4
        for i in range(1, columns*rows+1 ):
            if (i == data_timesteps.shape[0]):
                break
            for _ in range(3):
                img = data_timesteps[i+5000+5400+1739,_,:,:,0]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img)
        
        fig = plt.gcf()
        plt.suptitle('Sampled data') 
        plt.show()

    # input("X")
    
    #normalize data - subtract mean and std dev - that is standart procedure
    x_train = data_timesteps

    print(x_train.max())
    print(x_train.min())

    # normalize to zero mean and unit variance, this is best
    data_mean = x_train.mean()
    data_std = x_train.std()
    
    x_train = (x_train - x_train.mean()) / x_train.std()
    #print(x_train)
    print(x_train.max())
    print(x_train.min())
    print(x_train.mean())
    print(x_train.std())
    """
    # scale to range [0, 1], not necessary
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    #print(x_train)
    print(x_train.max())
    print(x_train.min())
    print(x_train.mean())
    print(x_train.std())
    """
    if (visualize == True):
        # visualize all normalized and scaled data
        fig=plt.figure(figsize=(20, 100))
        columns = 25
        rows = 4
        for i in range(1, columns*rows+1 ):
            if (i == x_train.shape[0]):
                break
            img = x_train[i,1,:,:,0]
            #img.reshape(84,444)
            #print(img.shape)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        
        fig = plt.gcf()
        plt.suptitle('Normalized data') 
        plt.show()

    if (visualize == True):
        # visualize all normalized and scaled data
        fig=plt.figure(figsize=(20, 100))
        columns = 25
        rows = 4
        for i in range(1, columns*rows+1 ):
            if (i == x_train.shape[0]):
                break
            img = x_train[i,2,:,:,0]
            #img.reshape(84,444)
            #print(img.shape)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        
        fig = plt.gcf()
        plt.suptitle('Normalized data') 
        plt.show()
    
    # if (dataset == "flow"):
    #     return x_train, data_mean, data_std, names_timesteps
    # else:
    #     return x_train, data_mean, data_std

    return x_train, data_mean, data_std, names_timesteps


def preprocess(dataset, visualize, data, names="", temporal=False):

    if (temporal == False):
        x_train, data_mean, data_std = preprocess_2d(data, visualize)
    else:
        # if (dataset == "droplet"):
        #     x_train, data_mean, data_std = preprocess_3d(dataset, visualize, data)
        # else: # flow, 3D
        x_train, data_mean, data_std, names_timesteps = preprocess_3d(dataset, visualize, data, names)
        

    # if (dataset == "flow" and temporal == True):
    #     print("3D flow")
    #     return x_train, data_mean, data_std, names_timesteps
    if (temporal == True):
        return x_train, data_mean, data_std, names_timesteps
    else:
        return x_train, data_mean, data_std


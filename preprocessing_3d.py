import numpy as np
from matplotlib import pyplot as plt

def preprocess(data, names, visualize):
    # reshape, visualize, normalize, scale

    for k in range(10):
        print(names[k])

    count = data.shape[0]
    timestep = 0
    i = 0
    names_timesteps = []
    data_timesteps = np.zeros(( int(count/3), 3, data.shape[1], data.shape[2], 1 ))
    for k in range(count):
        data_timesteps[i,timestep,:,:,:] = data[k,:,:,:]
        #print(names[k])
        timestep += 1
        if (timestep%3 == 0):
            i += 1
            timestep = 0
            #print(i)
            names_timesteps.append(names[k])
            #print(len(names_timesteps))

    print('data with timesteps: ', data_timesteps.shape)
    #print(i, timestep)
    print('names with timesteps: ', len(names_timesteps))

    if (visualize == True):
        # visualize all data
        fig=plt.figure(figsize=(20, 100))
        columns = 25
        rows = 4
        for i in range(1, columns*rows+1 ):
            if (i == data_timesteps.shape[0]):
                break
            img = data_timesteps[i,0,:,:,0]
            #img.reshape(84,444)
            #print(img.shape)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            
        fig = plt.gcf()
        plt.suptitle('Original data') 
        plt.show()
    
    #normalize data - subtract mean and std dev - that is standart procedure
    x_train = data_timesteps

    print(x_train.max())
    print(x_train.min())

    # normalize to zero mean and unit variance
    data_mean = x_train.mean()
    data_std = x_train.std()
    x_train = (x_train - x_train.mean()) / x_train.std()
    #print(x_train)
    print(x_train.max())
    print(x_train.min())
    print(x_train.mean())
    print(x_train.std())
    """
    # scale to range [0, 1]
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
    
    return x_train, data_mean, data_std, names_timesteps

from matplotlib import pyplot as plt
import numpy as np
import time
import pickle

data = np.fromfile('mcmc_a.raw', dtype='f8')
print(data.shape)
data.resize(95000, 50, 50)
print(data.shape)

examples = []
examples.append(data[5050])
examples.append(data[22050])
examples.append(data[13050])
examples.append(data[40050])
examples.append(data[7050])
examples.append(data[36050])
examples.append(data[18050])
examples.append(data[75050])
examples.append(data[32050])

fig=plt.figure()
columns = 9
rows = 1
for i in range(1, columns*rows+1 ):
    if (i == data.shape[0]):
        break
    img = examples[i-1]
    img = np.flip(img, 1)
    fig.add_subplot(rows, columns, i)
    # fig.tight_layout(pad=2.0)
    plt.imshow(img)
    plt.axis('off')

fig = plt.gcf()

plt.axis('off')
plt.show()
fig.savefig('examples.png', dpi=300)

# input("waiting...")

labelled = False # True
if labelled:
    # save only labelled data
    labelled_data = np.zeros((0, 50, 50))
    names = []
    f = open("labels_mcmc.txt", 'r')
    for line in f:
        l = line.split(" ")
        from_to = l[0].split("-")
        print(from_to[0], from_to[1])
        labelled_data = np.append(labelled_data, data[int(from_to[0]):int(from_to[1]),...], axis=0)
        for i in range(int(from_to[1])-int(from_to[0])):
            names.append(l[1][0])

    print(len(names))
    print(labelled_data.shape)

    # input("waiting...")

    pkl_file = open("mcmc_labelled_data.pkl", 'wb')
    pickle.dump(labelled_data, pkl_file, protocol=4)
    pkl_file.close

    pkl_file = open("mcmc_labelled_names.pkl", 'wb')
    pickle.dump(names, pkl_file, protocol=4)
    pkl_file.close
else: 
    # save all data

    pkl_file = open("mcmc.pkl", 'wb')
    pickle.dump(data, pkl_file, protocol=4)
    pkl_file.close

    pkl_file = open("mcmc.pkl", 'rb')
    data = []
    data = pickle.load(pkl_file)

    print(data.shape)
from matplotlib import pyplot as plt
import numpy as np
import time
import pickle

# image = np.empty((50,50,95000))

# image.data[:] = open('mcmc_a.raw').read()
# start_time = time.time()

image = np.fromfile('mcmc_a.raw', dtype='f8')
print(image.shape)
image.resize(95000,50,50)
# image.resize(50,50,95000)
print(image.shape)

# fig=plt.figure()
# columns = 10
# rows = 10
# for i in range(1, columns*rows+1 ):
#     if (i == image.shape[0]):
#         break
#     shift = 9100
#     img = image[i+shift,:,:]
#     # img = image[:,:,i+10000]
#     img = np.flip(img, 1)
#     #img.reshape(84,444)
#     #print(img.shape)
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
#     plt.axis('off')

# fig = plt.gcf()
# plt.suptitle(shift)
# plt.axis('off')
# plt.show()

examples = []
examples.append(image[5050])
examples.append(image[22050])
examples.append(image[13050])
examples.append(image[40050])
examples.append(image[7050])
examples.append(image[36050])
examples.append(image[18050])
examples.append(image[75050])
examples.append(image[32050])

fig=plt.figure()
columns = 9
rows = 1
for i in range(1, columns*rows+1 ):
    if (i == image.shape[0]):
        break
    img = examples[i-1]
    img = np.flip(img, 1)
    fig.add_subplot(rows, columns, i)
    # fig.tight_layout(pad=2.0)
    plt.imshow(img)
    plt.axis('off')

fig = plt.gcf()

# plt.suptitle(shift)
plt.axis('off')
plt.show()
fig.savefig('examples.png', dpi=300)

input("input: ")

data = image
print('data shape:', data.shape)

# save labelled data
labelled_data = np.zeros((0,50,50))
names = []
f = open("labels.txt", 'r')
for line in f:
    l = line.split(" ")
    # names.append(l[1][0])
    from_to = l[0].split("-")
    print(from_to[0], from_to[1])
    labelled_data = np.append(labelled_data, data[int(from_to[0]):int(from_to[1]),...], axis=0)
    for i in range(int(from_to[1])-int(from_to[0])):
        names.append(l[1][0])

print(len(names))
print(labelled_data.shape)

pkl_file = open("mcmc_labelled_data.pkl", 'wb')
pickle.dump(labelled_data, pkl_file, protocol=4)
pkl_file.close

pkl_file = open("mcmc_labelled_names.pkl", 'wb')
pickle.dump(names, pkl_file, protocol=4)
pkl_file.close


# input("input: ")

# elapsed_time = time.time() - start_time
# print("All", count, "frames were loaded successfully in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

# pkl_file = open("mcmc.pkl", 'wb')
# pickle.dump(data, pkl_file, protocol=4)
# pkl_file.close

# pkl_file = open("mcmc.pkl", 'rb')
# data = []
# data = pickle.load(pkl_file)

print(data.shape)


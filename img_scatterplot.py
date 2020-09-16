import numpy as np
from matplotlib import pyplot as plt

def im_scatter(x, y, data, dataset, ax=None, zoom=1, temporal=False):

    #zoom = input("zoom: ")
    vmax = data.max()
    vmin = data.min()
    print('range of data:', vmin, vmax)
    if (dataset == "flow"):
        vmax = 70
    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    print('range of colormap:', vmin, vmax)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    # if ax is None:
    #     ax = plt.gca()
    # try:
    #     image = plt.imread(image)
    # except TypeError:
    #     # Likely already an array...
    #     pass
    #im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artist = []
    i = 0
    for x0, y0 in zip(x, y):
        if (temporal):
            img = data[i,1,:,:,0] # middle image
        else:
            img = data[i,:,:,0]
        i += 1
        im = OffsetImage(img, zoom=zoom)
        #img = im.get_children()[0]
        #img.set_clim(vmin, vmax)
        im.get_children()[0].set_clim(vmin, vmax)
        if (dataset == "droplet"):
            im.get_children()[0].set_cmap('gray')
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artist.append(ax.add_artist(ab))
        # plt.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    #plt.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
    #plt.colorbar()

    return im

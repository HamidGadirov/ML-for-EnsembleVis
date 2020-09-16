import os
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from tsne_projection import tsne_projection
from umap_projection import umap_projection
from img_scatterplot import im_scatter
from classification import kNN_classification_flow, kNN_classification_droplet

class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def get_cluster_centers(encoded_vec_2d, names):

    turb_vec, lam_vec = [], [] # turbilent and laminar vectors

    for i in range(encoded_vec_2d.shape[0]):
        if names[i].find(" l") == -1:
            turb_vec.append(encoded_vec_2d[i]) # wrong!
            #print(names[i])
        else:
            lam_vec.append(encoded_vec_2d[i])
            #print(names[i])
            
    print("turb: ", len(turb_vec))
    print("lam: ", len(lam_vec))

    lam_center = np.mean(lam_vec, axis=0)
    turb_center = np.mean(turb_vec, axis=0)

    center = []
    center.append(lam_center)
    center.append(turb_center)
    #print(center)

    return center

def kmeans_clustering(encoded_vec_2d):

    from sklearn.cluster import KMeans
    import numpy as np

    kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_vec_2d)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(encoded_vec_2d)
    plt.subplot(222)
    plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=y_pred)
    plt.title("Two clusters")

    y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(encoded_vec_2d)
    plt.subplot(223)
    plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=y_pred)
    plt.title("Three clusters")

    y_pred = KMeans(n_clusters=4, random_state=0).fit_predict(encoded_vec_2d)
    plt.subplot(224)
    plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=y_pred)
    plt.title("Four clusters")

    plt.show()

def pca_projection(encoded_vec, test_data, latent_vector, title, dataset="", names="", temporal=False):

    # normalize to zero mean and unit variance
    encoded_vec = (encoded_vec - encoded_vec.mean()) / encoded_vec.std()
    #print('pca encoded_vec.mean:', encoded_vec.mean())
    #print('pca encoded_vec.std:', encoded_vec.std())

    pca = PCA(n_components=50, whiten=True) # try >2

    if (latent_vector == False):
        encoded_vec_resized = encoded_vec
        encoded_vec_resized = np.reshape( encoded_vec_resized, (encoded_vec.shape[0], encoded_vec.shape[1]*encoded_vec.shape[2]*encoded_vec.shape[3]) )
        print(encoded_vec_resized.shape)
        encoded_vec = encoded_vec_resized

    start_time = time.time()


    # for i in range(encoded_vec.shape[0]):
    #     if (names[i].find("sampled-300/cylinder_100_20_70/frame_0049.raw t") != -1):
    #         plt.figure()
    #         img = test_data[i,:,:,0]
    #         plt.imshow(img) 
    #         plt.show()
    #         break

    #test_data_tmp = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
    encoded_vec_2d = pca.fit_transform(encoded_vec) # encoded_vec

    elapsed_time = time.time() - start_time
    print("PCA was computed in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

    print('PCA output shape:', encoded_vec_2d.shape) # (batch_size, 2)

    # kNN for separability flow dataset
    if (dataset == "flow"):
        knn_title = "PCA"
        # labels = kNN_classification(encoded_vec_2d, names, knn_title)

    title_tsne = ""
    title_tsne += title
    title_tsne += 'Latent -> PCA -> t-SNE scatterplot, perp='

    title_umap = ""
    title_umap += title
    title_umap += 'Latent -> PCA -> UMAP scatterplot '

    # use t-sne & umap after the pca projection (in that case n_comp >2)
    print("t-SNE after PCA")
    if (temporal):
        #tsne_projection(encoded_vec_2d, test_data, latent_vector, names, title, perp=20, temporal=True)
        tsne_projection(encoded_vec_2d, test_data, latent_vector, title_tsne, dataset, names, temporal=True, perp=30)
        #tsne_projection(encoded_vec_2d, test_data, latent_vector, names, title, perp=40, temporal=True)

        umap_projection(encoded_vec, test_data, latent_vector, title_umap, dataset, names, temporal=True)
    else:
        #tsne_projection(encoded_vec_2d, test_data, latent_vector, names, title, perp=20)
        tsne_projection(encoded_vec_2d, test_data, latent_vector, title_tsne, dataset, names, perp=30)
        #tsne_projection(encoded_vec_2d, test_data, latent_vector, names, title, perp=40)

        umap_projection(encoded_vec, test_data, latent_vector, title_umap, dataset, names)

    """

    x = encoded_vec_2d[:, 0]
    y = encoded_vec_2d[:, 1]

    if (temporal):
        model_name="3d_beta-vae"
    else:
        model_name="2d_beta-vae"

    # draw a scatterplot with annotations
    fig, ax = plt.subplots()
    sc = plt.scatter(x, y)

    if (dataset == "mnist"):
        zoom = 0.6
    if (dataset == "droplet"):
        zoom = 0.1
    if (dataset == "flow"):
        zoom = 0.15

        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        #names = names[21:30]

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            try:
                text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
                annot.set_text(text)
            except IndexError:
                print(ind["ind"])
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            #annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

    
   
    #plt.scatter(x, y)
    #fig = plt.gcf()
    title_pca = ""
    title_pca += title
    title_pca += 'Latent -> PCA scatterplot'
    plt.suptitle(title_pca)
    plt.show()
    #filename = os.path.join(model_name, "pca_scatter.png")
    #fig.savefig(filename)

    # draw a scatterplot with images and annotations
    if (temporal == True):
        print("3D data")
        image_path = test_data[0,0,:,:,0]
        fig, ax = plt.subplots()
        im_scatter(x, y, test_data, image_path, ax=ax, zoom=zoom, temporal=True)
    else:
        image_path = test_data[0,:,:,0]
        fig, ax = plt.subplots()
        im_scatter(x, y, test_data, image_path, ax=ax, zoom=zoom)

    #ax.plot(x, y)

    sc = plt.scatter(x, y)
    if (dataset == "flow"):
        annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords='figure points',
                        bbox=dict(boxstyle="round", fc="w"))
        annot.set_visible(False)

        fig.canvas.mpl_connect("motion_notify_event", hover)

    fig = plt.gcf()
    title_pca += ', frames'
    plt.suptitle(title_pca)

    # ZoomPan scrollig
    scale = 1.1
    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale = scale)
    figPan = zp.pan_factory(ax)

    plt.show()
    plt.close()
    
    filename = os.path.join(model_name, "pca_scatter_images.png")
    fig.savefig(filename)
    """

import os
import time
import numpy as np
import pickle
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
params = {'legend.fontsize': 15, 'legend.loc': "upper right"}
# plt.rcParams['legend.loc'] = "upper right"
#         #   'legend.handlelength': 2}
plt.rcParams.update(params)
from sklearn.manifold import TSNE
import statistics

from img_scatterplot import im_scatter
from metrics import kNN_classification_flow, kNN_classification_droplet, kNN_classification_mnist
from metrics import kNN_fraction_flow, kNN_fraction_droplet, kNN_fraction_mnist
from metrics import variance_flow, variance_droplet, variance_mnist
from grid_projection import grid_projection

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

def draw_movement(encoded_vec_2d, names, labels, title, dir_res_model):

    fig, ax = plt.subplots()
    scatter = plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=labels)
    handles = scatter.legend_elements()[0]
    label_names = ("laminar", "turbulent")
    legend1 = ax.legend(handles, label_names)
    ax.add_artist(legend1)
    
    # # draw movement of the temporal data in the projection, one ensemble
    # encoded_vec_2d_ensemble = []
    # names_ensemble = []
    # #print(names)
    # # e.g. we have name: sampled-300-2/cylinder_80_25_30/frame_0051.raw t
    # cylinders = names[1].split("/")[1] # cylinder_80_25_30
    # print(cylinders)
    # for idx, name in enumerate(names):
    #     cylinder = name.split("/")[1]
    #     if (cylinder == cylinders):
    #         encoded_vec_2d_ensemble.append(encoded_vec_2d[idx])
    #         names_ensemble.append(name)
    # print(names_ensemble) # why same ??
    # encoded_vec_2d_ensemble = np.asarray(encoded_vec_2d_ensemble)
    # plt.plot(encoded_vec_2d_ensemble[:, 0], encoded_vec_2d_ensemble[:, 1])


    # draw movement of the temporal data in the projection, all ensembles
    cylinders = []
    for i, name in enumerate(names):
        cylinders.append(name.split("/")[1])

    unique_cylinders = np.unique(cylinders)
    print(unique_cylinders)

    #print(names)
    # e.g. we have name: sampled-300-2/cylinder_80_25_30/frame_0051.raw t
    # cylinders = names[1].split("/")[1] # cylinder_80_25_30
    # print(cylinders)
    i = 0
    for cylinders in unique_cylinders:
        i += 1
        encoded_vec_2d_ensemble = []
        names_ensemble = []
        print(cylinders)
        for idx, name in enumerate(names):
            cylinder = name.split("/")[1]
            if (cylinder == cylinders):
                encoded_vec_2d_ensemble.append(encoded_vec_2d[idx])
                names_ensemble.append(name)
        #print(names_ensemble) # why same ??
        encoded_vec_2d_ensemble = np.asarray(encoded_vec_2d_ensemble)
        plt.plot(encoded_vec_2d_ensemble[:, 0], encoded_vec_2d_ensemble[:, 1])
        if (i==5): # draw only for 5 cylinders
            break


    knn_title = title
    # knn_title += ", separability="
    # knn_title += str("%.3f" % acc_mean)
    plt.suptitle(knn_title, fontsize=17)
    plt.axis('off')
    #plt.show()
    plt.tight_layout()
    #fig.set_size_inches(10, 8)
    fig.savefig('{}/latent_tsne_scatter_labels_movement.png'.format(dir_res_model), dpi=300)
    plt.close(fig)

def tsne_projection(encoded_vec, test_data, latent_vector, title, dir_res_model, dataset="", names="", temporal=False, perp=30):

    # if encoded_vec is True - not loading from pickle
    print(dir_res_model)
    fn = os.path.join(dir_res_model, "encoded_tsne_2d.pkl")
    #print(encoded_vec)

    # pkl_file = open("sampled-300_labelled_names.pkl", 'wb')
    # pickle.dump(cylinder_names, pkl_file)
    # pkl_file.close

    print(type(encoded_vec))

    if isinstance(encoded_vec, np.ndarray):
    #if encoded_vec.any(): # project from encoded_vec to 2d
        # normalize to zero mean and unit variance
        encoded_vec = (encoded_vec - encoded_vec.mean()) / encoded_vec.std()
        # print('tsne encoded_vec.mean:', encoded_vec.mean())
        # print('tsne encoded_vec.std:', encoded_vec.std())
        # print(encoded_vec)

        if (latent_vector == False):
            encoded_vec_resized = encoded_vec
            encoded_vec_resized = np.reshape( encoded_vec_resized, (encoded_vec.shape[0], encoded_vec.shape[1]*encoded_vec.shape[2]*encoded_vec.shape[3]) )
            print(encoded_vec_resized.shape)
            encoded_vec = encoded_vec_resized

        start_time = time.time()

        tsne = TSNE(n_components=2, random_state=0, perplexity=perp)
        encoded_vec_2d = tsne.fit_transform(encoded_vec) # encoded_vec

        elapsed_time = time.time() - start_time
        print("t-SNE accuracy was computed in", "{0:.2f}".format(round(elapsed_time, 2)), "seconds.")

        #encoded_vec_2d = encoded_vec # 2D dim latent space

        # Save the encoded data 
        #print(type(encoded_vec_2d))
        pkl_file = open(fn, 'wb')
        pickle.dump(encoded_vec_2d, pkl_file)
        print("Encoding vecs were saved to pickle")
        pkl_file.close

    else: # load directly from pickle
        if not ("Raw data" in title):
            pkl_file = open(fn, 'rb')
            encoded_vec_2d = pickle.load(pkl_file)
            print("Encoding vecs were loaded from pickle")
            encoded_vec_2d = np.asarray(encoded_vec_2d)
            print(encoded_vec_2d.shape)

    filename = os.path.join(dir_res_model, "metrics.txt")

    title += str(perp)

    # kNN for separability flow dataset
    if (dataset == "flow"):
        knn_title = "t-SNE"

        if("Raw data" in title): # baseline
            test_data_tmp = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
            tsne = TSNE(n_components=2, random_state=0, perplexity=perp)
            encoded_vec_2d = tsne.fit_transform(test_data_tmp) # encoded_vec

        # # measure the uncertainty
        # accuracy = []
        # for _ in range(2):
        #     if("Raw data" in title): # baseline
        #         test_data_tmp = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
        #         encoded_vec_2d = tsne.fit_transform(test_data_tmp) # encoded_vec
        #     else:
        #         #encoded_vec_2d = encoded_vec # apply kNN to latent vectors
        #         if encoded_vec:
        #             encoded_vec_2d = tsne.fit_transform(encoded_vec) # encoded_vec
        #         else: # load directly from pickle
        #             pkl_file = open(fn, 'rb')
        #             encoded_vec_2d = pickle.load(pkl_file)
        #             encoded_vec_2d = np.asarray(encoded_vec_2d)
        #             print(encoded_vec_2d.shape)

        #     # for k in range(5, 37, 4): similar
        #     acc, labels = kNN_classification_flow(encoded_vec_2d, names, knn_title)
        #     accuracy.append(acc)

        # print(len(labels), "labels considered")
        # acc_mean, acc_std = statistics.mean(accuracy), statistics.stdev(accuracy)
        # print("Accuracy mean and std:", acc_mean, acc_std)

        separability, labels = kNN_classification_flow(encoded_vec_2d, names, knn_title)
        print("Separability:", separability)
        print(len(labels), "labels considered")

        with open(filename, "w") as text_file:
            text_file.write("Separability ")
            #print(f"Accuracy of clustering: ", file=text_file)
            #print("t-SNE mean and std: %f %f" % (acc_mean, acc_std), file=text_file)
            text_file.write("t-SNE %f \n" % separability)
            # print("UMAP mean and std: %f %f" % (acc_mean, acc_std), file=text_file)

        # Neighborhood hit metric: measure the fraction of the k-nearest neighbours 
        # of a projected point that has the same class label
        fraction = kNN_fraction_flow(encoded_vec_2d, names, knn_title)
        print("Neighborhood hit (fraction):", fraction)
        with open(filename, "a") as text_file:
            text_file.write("Neighborhood_hit ")
            text_file.write("t-SNE %f \n" % fraction)

        # Distance from cluster centers metric
        #dist_to_laminar, dist_to_turbulent = get_cluster_centers_flow(encoded_vec_2d, names) 
        dist_to_centers_mean = variance_flow(encoded_vec_2d, names) 
        # print("Distances:", dist_to_laminar, dist_to_turbulent)
        # with open(filename, "a") as text_file:
        #     text_file.write("Distances: \n")
        #     text_file.write("t-SNE dist_to_laminar and dist_to_turbulent: %f %f \n" % (dist_to_laminar, dist_to_turbulent))
        print("Mean of distances for all classes:", dist_to_centers_mean)
        with open(filename, "a") as text_file:
            text_file.write("Variance ")
            text_file.write("t-SNE %f \n" % dist_to_centers_mean)

        print("Labels:", len(labels))
        # draw a scatterplot with colored labels
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots()
        scatter = plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=labels)
        handles = scatter.legend_elements()[0]
        label_names = ("laminar", "turbulent")
        legend1 = ax.legend(handles, label_names)
        ax.add_artist(legend1)

        knn_title = title
        # knn_title += ", separability="
        # knn_title += str("%.3f" % separability)
        knn_title += ", neigh hit="
        knn_title += str("%.3f" % fraction)
        knn_title += ", spread="
        knn_title += str("%.3f" % dist_to_centers_mean)
        #plt.suptitle(knn_title, fontsize=15)
        ax.set_title(knn_title, fontsize=17)
        plt.axis('off')
        #plt.show()
        plt.tight_layout()
        #fig.set_size_inches(10, 8)
        fig.savefig('{}/latent_tsne_scatter_labels.png'.format(dir_res_model), dpi=300)
        plt.close(fig)

        draw_movement(encoded_vec_2d, names, labels, title, dir_res_model)

    elif (dataset == "droplet"):

        knn_title = "t-SNE"

        if("Raw data" in title): # baseline
            test_data_tmp = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
            tsne = TSNE(n_components=2, random_state=0, perplexity=perp)
            encoded_vec_2d = tsne.fit_transform(test_data_tmp) # encoded_vec

        # # measure the uncertainty
        # accuracy = []
        # for _ in range(2):
        #     if("Raw data" in title): # baseline
        #         test_data_tmp = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
        #         encoded_vec_2d = tsne.fit_transform(test_data_tmp) # encoded_vec
        #     else:
        #         #encoded_vec_2d = encoded_vec # apply kNN to latent vectors
        #         encoded_vec_2d = tsne.fit_transform(encoded_vec) # encoded_vec
            
        #     # for k in range(5, 37, 4): similar
        #     acc = kNN_classification_droplet(encoded_vec_2d, names, knn_title)
        #     accuracy.append(acc)

        # print(len(names), "labels considered")
        # acc_mean, acc_std = statistics.mean(accuracy), statistics.stdev(accuracy)
        # print("Accuracy mean and std:", acc_mean, acc_std)

        separability = kNN_classification_droplet(encoded_vec_2d, names, knn_title)
        print("Separability:", separability)
        print(len(names), "labels considered")

        with open(filename, "w") as text_file:
            text_file.write("Separability ")
            #print(f"Accuracy of clustering: ", file=text_file)
            #print("t-SNE mean and std: %f %f" % (acc_mean, acc_std), file=text_file)
            text_file.write("t-SNE %f \n" % separability)
            # print("UMAP mean and std: %f %f" % (acc_mean, acc_std), file=text_file)

        # Neighborhood hit metric: measure the fraction of the k-nearest neighbours 
        # of a projected point that has the same class label
        fraction = kNN_fraction_droplet(encoded_vec_2d, names, knn_title)
        print("Neighborhood hit (fraction):", fraction)
        with open(filename, "a") as text_file:
            text_file.write("Neighborhood_hit ")
            text_file.write("t-SNE %f \n" % fraction)

        # Distance from cluster centers metric
        dist_to_centers_mean = variance_droplet(encoded_vec_2d, names) 
        print("Mean of distances for all classes:", dist_to_centers_mean)
        with open(filename, "a") as text_file:
            text_file.write("Variance ")
            text_file.write("t-SNE %f \n" % dist_to_centers_mean)

        print("Labels:", len(names))
        print("unique:", np.unique(names))
        unique_names, indexed_names = np.unique(names, return_inverse=True)
        #unique_names = ['bubble', 'bubble-splash', 'crown', 'crown-splash', 'splash', 'drop', 'none'] # scatter colors
        #print(unique_names, indexed_names)
        # draw a scatterplot with colored labels
        fig, ax = plt.subplots()
        scatter = plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=indexed_names)
        #plt.legend(unique_names)
        # produce a legend with the unique colors from the scatter
        handles = scatter.legend_elements()[0]
        labels = unique_names
        legend1 = ax.legend(handles, labels)
        ax.add_artist(legend1)
        #ax.legend(prop={'size': 6})
        #cbar= plt.colorbar()
        knn_title = title
        # knn_title += ", separability="
        # knn_title += str("%.3f" % separability)
        knn_title += ", neigh hit="
        knn_title += str("%.3f" % fraction)
        knn_title += ", spread="
        knn_title += str("%.3f" % dist_to_centers_mean)
        #plt.suptitle(knn_title, fontsize=15)
        ax.set_title(knn_title, fontsize=17)
        plt.axis('off')
        #plt.show()
        plt.tight_layout()
        #fig.set_size_inches(12, 9)
        fig.savefig('{}/latent_tsne_scatter_labels.png'.format(dir_res_model), dpi=300)
        plt.close(fig)

    if (dataset == "mnist"):
        # if encoded_vec.any():
        #     encoded_vec_2d = tsne.fit_transform(encoded_vec) # encoded_vec
        # else: # load directly from pickle
        #     pkl_file = open(fn, 'rb')
        #     encoded_vec_2d = pickle.load(pkl_file)
        #     encoded_vec_2d = np.asarray(encoded_vec_2d)
        #     print(encoded_vec_2d.shape)

        if("Raw data" in title): # baseline
            test_data_tmp = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
            tsne = TSNE(n_components=2, random_state=0, perplexity=perp)
            encoded_vec_2d = tsne.fit_transform(test_data_tmp) # encoded_vec

        knn_title = "t-SNE"

        separability = kNN_classification_mnist(encoded_vec_2d, names, knn_title)
        print("Separability:", separability)
        print(len(names), "labels considered")

        with open(filename, "w") as text_file:
            text_file.write("Separability ")
            text_file.write("t-SNE %f \n" % separability)
            # print("UMAP mean and std: %f %f" % (acc_mean, acc_std), file=text_file)

        # Neighborhood hit metric: measure the fraction of the k-nearest neighbours 
        # of a projected point that has the same class label
        fraction = kNN_fraction_mnist(encoded_vec_2d, names, knn_title)
        print("Neighborhood hit (fraction):", fraction)
        with open(filename, "a") as text_file:
            text_file.write("Neighborhood_hit ")
            text_file.write("t-SNE %f \n" % fraction)

        # Distance from cluster centers metric
        dist_to_centers_mean = variance_mnist(encoded_vec_2d, names) 
        print("Mean of distances for all classes:", dist_to_centers_mean)
        with open(filename, "a") as text_file:
            text_file.write("Variance ")
            text_file.write("t-SNE %f \n" % dist_to_centers_mean)

        print("unique:", np.unique(names))
        unique_names, indexed_names = np.unique(names, return_inverse=True)
        #print(unique_names, indexed_names)
        # draw a scatterplot with colored labels
        fig, ax = plt.subplots()
        scatter = plt.scatter(encoded_vec_2d[:, 0], encoded_vec_2d[:, 1], c=indexed_names)
        #plt.legend(unique_names)
        # produce a legend with the unique colors from the scatter
        handles = scatter.legend_elements()[0]
        labels = unique_names
        legend1 = ax.legend(handles, labels)
        ax.add_artist(legend1)
        #cbar= plt.colorbar()
        knn_title = title
        # knn_title += ", separability="
        # knn_title += str("%.3f" % separability)
        #plt.suptitle(knn_title, fontsize=15)
        ax.set_title(knn_title, fontsize=17)
        plt.axis('off')
        #plt.show()
        plt.tight_layout()
        #fig.set_size_inches(12, 9)
        fig.savefig('{}/latent_tsne_scatter_labels.png'.format(dir_res_model), dpi=300)
        plt.close(fig)


    ### Kernelized sorting
    proj = "tsne"
    grid_projection(encoded_vec_2d, test_data, dataset, dir_res_model, title, proj, temporal)

    x = encoded_vec_2d[:, 0]
    y = encoded_vec_2d[:, 1]

    # if (temporal):
    #     model_name="3d_beta-vae"
    # else:
    #     model_name="2d_beta-vae"

    # draw a scatterplot with annotations
    #fig, ax = plt.subplots()
    #sc = plt.scatter(x, y)

    if (dataset == "mnist"):
        zoom = 0.6
    if (dataset == "droplet"):
        zoom = 0.15
    if (dataset == "flow"):
        zoom = 0.15

        #fig=plt.figure()

        # annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
        #             bbox=dict(boxstyle="round", fc="w"),
        #             arrowprops=dict(arrowstyle="->"))
        # annot.set_visible(False)

        # #names = names[21:30]

        # def update_annot(ind):
        #     pos = sc.get_offsets()[ind["ind"][0]]
        #     annot.xy = pos
        #     try:
        #         text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        #         annot.set_text(text)
        #     except IndexError:
        #         print(ind["ind"])
        #     #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        #     #annot.get_bbox_patch().set_alpha(0.4)

        # def hover(event):
        #     vis = annot.get_visible()
        #     if event.inaxes == ax:
        #         cont, ind = sc.contains(event)
        #         if cont:
        #             update_annot(ind)
        #             annot.set_visible(True)
        #             fig.canvas.draw_idle()
        #         else:
        #             if vis:
        #                 annot.set_visible(False)
        #                 fig.canvas.draw_idle()

        #fig.canvas.mpl_connect("motion_notify_event", hover)


    #fig = plt.gcf()
    # title += str(perp)
    #plt.suptitle(title)
    #plt.show()
    #filename = os.path.join(model_name, "tsne_scatter.png")
    #fig.savefig(filename)

    # from getkey import getkey, keys
    # while True:  # making a loop
    #     key = getkey()
    #     if key == keys.UP:
    #         zoom += 0.1
    #     if key == keys.DOWN:
    #         zoom -= 0.1
    #     if key == 'q':
    #         break

    vmin = test_data.min()
    vmax = test_data.max()
    # plt.clim(vmin, vmax)
    # plt.set_cmap('viridis')
    # plt.clim(vmin, vmax)

    # draw a scatterplot with images and annotations
    fig, ax = plt.subplots()
    if (temporal == True):
        print("3D data")
        # image_path = test_data[0,0,:,:,0]
        # fig, ax = plt.subplots()
        im = im_scatter(x, y, test_data, dataset, ax=ax, zoom=zoom, temporal=True)
    else:
        # image_path = test_data[0,:,:,0]
        # fig, ax = plt.subplots()
        im = im_scatter(x, y, test_data, dataset, ax=ax, zoom=zoom)
    #ax.plot(x, y)

    # sc = plt.scatter(x, y)
    # if (dataset == "flow"):
    #     annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords='figure points',
    #                     bbox=dict(boxstyle="round", fc="w"))
    #     annot.set_visible(False)

    #     #fig.canvas.mpl_connect("motion_notify_event", hover)

    title += ', frames'
    ax.set_title(title, fontsize=17)
    #plt.suptitle(title, fontsize=15)
    #plt.subplots_adjust(bottom=0.5, top=1.5)

    # ZoomPan scrollig
    scale = 1.1
    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale = scale)
    figPan = zp.pan_factory(ax)

    #fig.colorbar(im.get_children()[0])

    plt.axis('off')
    #plt.show()
    #plt.close()
    plt.tight_layout() # (pad=2)
    #fig.set_size_inches(10, 8)
    #fig.savefig('{}/latent_tsne.png'.format(dir_res_model), bbox_inches='tight')
    fig.savefig('{}/latent_tsne.png'.format(dir_res_model), dpi=300)
    # filename = os.path.join(model_name, "tsne_scatter_images.png")
    # fig.savefig(filename)
    plt.close(fig)
import os, re
from matplotlib import artist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

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

# dirName = '3D_VAE' # this should be current dir
dirName = './'

def getListOfFiles(dirName):
    # For the given path, get the List of all files in the directory tree 

    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    #print(listOfFile)
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

def load_metrics_from_folder():
    listOfFiles = getListOfFiles(dirName)
    listOfFiles.sort()
    #print(listOfFiles)
    print("####")
    #dir_names = []
    # metrics_pareto = {}
    # metrics_pareto["Neighborhood_hit"] = []
    # metrics_pareto["Distances"] = []

    #res = {[], metrics_pareto}
    res = {}

    models = []
    for f in listOfFiles:
        try:
            #print(f)
            model = f.split("/")[2] #that is the folder where to avg
            models.append(model)
        except:
            print(f)
    
    keys = np.unique(models)
    # print(keys)
    # values = metrics_pareto
    # for key in keys:
    #     print(key)
    #     break
    #     res.update(zip(key, values))

    d = {} # main dict
    temp_d = {} # temp dict for avg
    for key in keys:
        d[key] = {}
        temp_d[key] = {}
        d[key]['Neighborhood_hit'] = []
        d[key]['Silhouette'] = []
        # d[key]['Variance'] = []
        # d[key]['Separability'] = []
        temp_d[key]['Neighborhood_hit'] = []
        temp_d[key]['Silhouette'] = []
        # temp_d[key]['Variance'] = []
        # temp_d[key]['Separability'] = []

    # res[0].extend(np.unique(models))
    # print(res[0])

    print("------")
    print(d)
    print("------")

    for key, value in d.items():
        print(key) # model
        print(d[key]) # values
        #temp_d[key] = d[key]
        # key = "3d_vae_cropped_256_relu" # test

        for filename in listOfFiles:
            #print(filename)
            try:
                experiment = filename.split("/")[3]
            except:
                print(filename)
            #print("key:",key)
            #print("experiment:",experiment)
            if re.search("\d{3}$", experiment):
                continue
                experiment = experiment[:-4]
            elif re.search("\d{2}$", experiment):
                if int(re.search("\d{2}$", experiment).group(0)) > 20 : # only first 20 runs
                    continue
                experiment = experiment[:-3]
            elif re.search("\d{1}$", experiment):
                experiment = experiment[:-2]

            if key == experiment:
                #print("in fn:",filename)
                #print("values:",d[key])

                if not d[key]["Neighborhood_hit"]:
                    #print("empty")
                    # accumulate metrics

                    if filename.endswith(("metrics.txt")):
                        #print(filename)
                        with open(filename) as f:
                            lines = f.readlines()
                        print(lines)

                        lines = np.asarray(lines)
                        print(lines.shape[0])

                        for line in lines:
                            words = line.split(" ")
                            # words = words[:-2]
                            # if words[0] == "Separability" or words[0] == "\n":
                            if words[0] == "\n":
                                continue

                            if words[0] == "Neighborhood_hit" or words[0] == "Silhouette":
                                if words[1] == "UMAP":
                                    print(words[2])
                                    temp_d[key][words[0]].append(words[2])
                                    print("empty",d[key])
                                    print("not empty",temp_d[key])
                # else:
                #     print(d[key])

        print("end loop")
        #print(type(temp_d[key]['Neighborhood_hit'][0]))
        print(temp_d[key])
        # average temp_d[key]
        if not temp_d[key]["Neighborhood_hit"]: # test
            continue

        import statistics
        # d[key]['Neighborhood_hit'] = statistics.mean(list(map(float, temp_d[key]['Neighborhood_hit'])))
        # d[key]['Variance'] = statistics.mean(list(map(float, temp_d[key]['Variance'])))
        # #d[key] = temp_d[key]
        # print(d[key])

        d[key]['Neighborhood_hit'].append((list(map(float, temp_d[key]['Neighborhood_hit']))))
        d[key]['Neighborhood_hit'].append((list(map(float, temp_d[key]['Neighborhood_hit']))))
        d[key]['Silhouette'].append((list(map(float, temp_d[key]['Silhouette']))))
        d[key]['Silhouette'].append((list(map(float, temp_d[key]['Silhouette']))))
        # d[key]['Variance'].append((list(map(float, temp_d[key]['Variance']))))
        # d[key]['Separability'].append(statistics.mean(list(map(float, temp_d[key]['Separability']))))
        # d[key]['Separability'].append(statistics.stdev(list(map(float, temp_d[key]['Separability']))))
        #d[key] = temp_d[key]
        print(d[key])
        print("++++++++++++++++++++++++")

        #break
    
    # # return a tuple: dir and avg metrics_pareto
    # # accumulate all dirs and metrics_pareto, then pareto and plot
    return d


d = load_metrics_from_folder()
print("#####")
print(d)
print(len(d))

# metrics_pareto = np.zeros((0,2)) #
metrics_pareto = []
metrics_pareto_mean_std = []

print("#####")
i = 0
pareto_list = []
for key, value in d.items():
    if d[key]["Neighborhood_hit"]: # test
        print(key)
        # if("100" in key) or ("20" in key):
        #     continue
        # if("beta2" in key) and ("32" in key):
        #     continue
        # if("beta6" in key) and ("128" in key):
        #     continue

        print(d[key])
        pareto_list.append(key)
        #print(type(d[key]["Neighborhood_hit"]))
        metrics_pareto.append([])
        metrics_pareto[i].append(d[key]["Neighborhood_hit"][0])
        # metrics_pareto[i].append(d[key]["Variance"][0])
        # metrics_pareto[i].append(d[key]["Separability"][0])
        metrics_pareto[i].append(d[key]["Silhouette"][0])

        metrics_pareto_mean_std.append([]) # add variance
        metrics_pareto_mean_std[i].append(d[key]["Neighborhood_hit"][1])
        # metrics_pareto_mean_std[i].append(d[key]["Variance"][1])
        # metrics_pareto_mean_std[i].append(d[key]["Separability"][1])
        metrics_pareto_mean_std[i].append(d[key]["Silhouette"][1])

        i += 1

print(pareto_list)

print("Mean and std:", metrics_pareto_mean_std)

#print(metrics_pareto)
metrics_pareto = np.asarray(metrics_pareto)
#print(metrics_pareto)
print(metrics_pareto.shape)

x = metrics_pareto[:,0]
y = metrics_pareto[:,1]

metrics_pareto_mean_std = np.asarray(metrics_pareto_mean_std)
xerr = metrics_pareto_mean_std[:,0]
yerr = metrics_pareto_mean_std[:,1]
# zerr = metrics_pareto_mean_std[:,2]

# print(x,y)

#     metrics_pareto["Neighborhood_hit"] = []
#     metrics_pareto["Distances"] = []

# print(metrics_pareto.shape)
# print(metrics_pareto[0])
# print(metrics_pareto[1])

# x = metrics_pareto["Neighborhood_hit"]
# y = metrics_pareto["Distances"]

# fig, ax = plt.subplots()
# plt.scatter(x, y)
# ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
# #plt.errorbar(x, y, e, linestyle='None', marker='^')
# plt.xlabel('Neighborhood_hit')
# plt.ylabel('Variance')
# #plt.show()
# # plt.close()

# print(next(iter(metrics_pareto)))
# print(len(metrics_pareto[next(iter(metrics_pareto))]))

# metrics_vals = metrics_pareto
# print(metrics_vals)

# metrics_vals = np.vstack((x,y))
# print(metrics_vals)

# # we want max Neighborhood_hit but min Distances
# # # let's inverse the dist for simplicity - No!
# # #scores[:, 1] = 1/scores[:, 1]
# # #print(scores)

x_all = metrics_pareto[:, 0]
y_all = metrics_pareto[:, 1]

# z_all = metrics_pareto[:, 2]

filename = "metrics.txt"

#uncertainty equal to the standard deviation

# # draw a scatterplot with annotations
# fig, ax = plt.subplots() # just a figure and only one subplot
# # ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o') # for std
# sc = ax.scatter(x_all, y_all)

names = pareto_list # all unique models
unordered_labels = []
for name in names:
    if ('baseline' in name):
        temp = re.findall(r'\d+', name)
        res = list(map(int, temp))
        unordered_labels.append(res)

print(unordered_labels)
unordered_labels = [item for sublist in unordered_labels for item in sublist]
print("!", unordered_labels)

def draw_boxplot(x_all, y_title, fig_name):
    x_all = np.array(x_all)

    beta2 = []
    beta8 = []
    baseline = []
    ae = []
    wae = []
    for i in range(len(x_all)):
        if ('beta2' in names[i]):
            beta2.append(np.vstack(x_all[i]))
        if ('beta8' in names[i]):
            beta8.append(np.vstack(x_all[i]))
        if ('baseline' in names[i]):
            baseline.append(np.vstack(x_all[i]))
        if ('_ae' in names[i]):
            print(x_all[i])
            ae.append(np.vstack(x_all[i]))
        if ('wae' in names[i]):
            wae.append(np.vstack(x_all[i]))
    
    step = 250
    labels = np.arange(250, 2500+step, step)
    labels = np.insert(labels, 0, 100)

    dataset_size = 100000
    # in percentage
    labels = [round(i/dataset_size*100, 2) for i in labels]
    # add % sign
    labels = [str(i)+'%' for i in labels]
    print("Labels:", labels)

        # for idx, model in enumerate(models_list):
    #     models_list[idx] = np.array(model).reshape(len(labels), len(model[0]))

    beta2 = np.array(beta2).reshape(len(labels), len(beta2[0]))
    beta8 = np.array(beta8).reshape(len(labels), len(beta8[0]))
    baseline = np.array(baseline).reshape(len(labels), len(baseline[0]))
    ae = np.array(ae).reshape(len(labels), len(ae[0]))
    wae = np.array(wae).reshape(len(labels), len(wae[0]))
    # print("VAE len:", len(beta2[1]))
    # print("VAE base:", len(baseline[0]))
    # print(beta2.shape)
    # print(beta2)

    # sort according to labels
    # models_list = [beta2, beta8, ae, wae]
    # for model in models_list:
    #     zipped_lists = zip(unordered_labels, model)
    #     sorted_pairs = sorted(zipped_lists)
    #     tuples = zip(*sorted_pairs)
    #     ordered_labels, model = [ list(tuple) for tuple in  tuples]
    #     model = np.asarray(model)

    zipped_lists = zip(unordered_labels, beta2, baseline, beta8, ae, wae)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    ordered_labels, beta2, baseline, beta8, ae, wae = [ list(tuple) for tuple in  tuples]
    beta2 = np.asarray(beta2)
    beta8 = np.asarray(beta8)
    baseline = np.asarray(baseline)
    ae = np.asarray(ae)
    wae = np.asarray(wae)
    print("ordered_labels", ordered_labels)
    print("labels", labels)
    print("WAE", wae)

    def draw_plot(data, offset, edge_color, fill_color, lab):
        pos = np.arange(data.shape[0])+offset 
        data = data.transpose()
        # print("Box", data)
        gray_diamond = dict(markerfacecolor='gray', marker='d')
        bp = ax.boxplot(data, positions=pos, widths=0.1, patch_artist=True, manage_xticks=False, flierprops=gray_diamond)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color, linewidth=1.5)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        median_x = [] 
        median_y = []
        for medline in bp['medians']:
            median_x.append((medline.get_xdata()[0]+medline.get_xdata()[1])/2.)
            median_y.append((medline.get_ydata()[0]+medline.get_ydata()[1])/2.)
        # ax.legend([bp["boxes"][0], bp["boxes"][0], bp["boxes"][0], bp["boxes"][0], bp["boxes"][0]], ["beta2-VAE", "beta8-VAE", "Baseline", "Sparse-AE", "WAE"], loc='upper right')
        return bp, median_x, median_y

    col = []
    for index in range(5):
        col.append(list(plt.cm.tab10(index)))
    # print(col)

    fig, ax = plt.subplots()
    linewidth = 2
    bp1, median_x1, median_y1 = draw_plot(beta8, 0,"black", col[0], "beta8-VAE")
    # artist_bp.append(bp1["boxes"][0])
    # print("medians:", median_x1, median_y1)
    # median_x.append(median_x1)
    # median_y.append(median_y1)
    plt.plot(median_x1, median_y1, color=col[0], linewidth=linewidth)
    bp2, median_x1, median_y1 = draw_plot(beta2, -0.15, "black", col[2], "beta2-VAE")
    plt.plot(median_x1, median_y1, color=col[2], linewidth=linewidth)
    bp3, median_x1, median_y1 = draw_plot(baseline, -0.05,"black", col[3], "Baseline")
    plt.plot(median_x1, median_y1, color=col[3], linewidth=linewidth)
    bp4, median_x1, median_y1 = draw_plot(ae, +0.05,"black", col[4], "Sparse-AE")
    plt.plot(median_x1, median_y1, color=col[4], linewidth=linewidth)
    bp5, median_x1, median_y1 = draw_plot(wae, +0.15,"black", col[1], "WAE")
    plt.plot(median_x1, median_y1, color=col[1], linewidth=linewidth)

    fs = 20
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0]], 
    ["beta8-VAE", "beta2-VAE", "Baseline", "Sparse-AE", "SWAE"], fontsize=fs, loc='lower right')
    # plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp4["boxes"][0], bp5["boxes"][0]], 
    #     ["beta2-VAE", "beta8-VAE", "Sparse-AE", "SWAE"], fontsize=fs, loc='lower right')
    
    plt.xticks(range(len(labels)), labels, fontsize=fs)
    plt.yticks(fontsize=fs)
    # plt.xlabel('Num of Labels', fontsize=20)
    # plt.ylabel(y_title, fontsize=20)
    # plt.suptitle("MCMC, metrics stabilty", fontsize=22)
    # fig.subplots_adjust(wspace=0.1)
    plt.grid(axis='y')
    fig.set_size_inches(10, 6)
    plt.tight_layout()
    # plt.savefig(__file__+'.png', bbox_inches='tight')
    # fig.set_size_inches(16, 9)
    plt.savefig(fig_name, dpi=300)
    plt.show()
    plt.close()


y_title = "Neighborhood hit"
fig_name = "mcmc_stability_neigh_hit_box.pdf"
draw_boxplot(x_all, y_title, fig_name)

y_title = "Silhouette"
fig_name = "mcmc_stability_silhouette_box.pdf"
draw_boxplot(y_all, y_title, fig_name)



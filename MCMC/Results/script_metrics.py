import os
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
        d[key]['Variance'] = []
        d[key]['Separability'] = []
        temp_d[key]['Neighborhood_hit'] = []
        temp_d[key]['Variance'] = []
        temp_d[key]['Separability'] = []

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

                            if words[1] == "UMAP":
                                print(words[2])
                                temp_d[key][words[0]].append(words[2][:-2])
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

        d[key]['Neighborhood_hit'].append(statistics.mean(list(map(float, temp_d[key]['Neighborhood_hit']))))
        d[key]['Neighborhood_hit'].append(statistics.stdev(list(map(float, temp_d[key]['Neighborhood_hit']))))
        d[key]['Variance'].append(statistics.mean(list(map(float, temp_d[key]['Variance']))))
        d[key]['Variance'].append(statistics.stdev(list(map(float, temp_d[key]['Variance']))))
        d[key]['Separability'].append(statistics.mean(list(map(float, temp_d[key]['Separability']))))
        d[key]['Separability'].append(statistics.stdev(list(map(float, temp_d[key]['Separability']))))
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
        if("100" in key) or ("20" in key):
            continue
        if("beta2" in key) and ("32" in key):
            continue
        if("beta6" in key) and ("128" in key):
            continue
        if("beta10" in key) and ("256" in key):
            continue

        print(d[key])
        pareto_list.append(key)
        #print(type(d[key]["Neighborhood_hit"]))
        metrics_pareto.append([])
        metrics_pareto[i].append(d[key]["Neighborhood_hit"][0])
        metrics_pareto[i].append(d[key]["Variance"][0])
        metrics_pareto[i].append(d[key]["Separability"][0])

        metrics_pareto_mean_std.append([]) # add variance
        metrics_pareto_mean_std[i].append(d[key]["Neighborhood_hit"][1])
        metrics_pareto_mean_std[i].append(d[key]["Variance"][1])
        metrics_pareto_mean_std[i].append(d[key]["Separability"][1])

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
zerr = metrics_pareto_mean_std[:,2]

print(x,y)

#     metrics_pareto["Neighborhood_hit"] = []
#     metrics_pareto["Distances"] = []

# print(metrics_pareto.shape)
# print(metrics_pareto[0])
# print(metrics_pareto[1])

# x = metrics_pareto["Neighborhood_hit"]
# y = metrics_pareto["Distances"]

fig, ax = plt.subplots()
plt.scatter(x, y)
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
#plt.errorbar(x, y, e, linestyle='None', marker='^')
plt.xlabel('Neighborhood_hit')
plt.ylabel('Variance')
#plt.show()
plt.close()

# print(next(iter(metrics_pareto)))
# print(len(metrics_pareto[next(iter(metrics_pareto))]))

metrics_vals = metrics_pareto
print(metrics_vals)

# metrics_vals = np.vstack((x,y))
# print(metrics_vals)

# # we want max Neighborhood_hit but min Distances
# # # let's inverse the dist for simplicity - No!
# # #scores[:, 1] = 1/scores[:, 1]
# # #print(scores)

def identify_pareto(metrics_vals):
    # Count number of items
    #population_size = len(metrics_pareto[next(iter(metrics_pareto))])
    population_size = metrics_vals.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by our 'j' point
            if (metrics_vals[j][0] > metrics_vals[i][0]) and (metrics_vals[j][1] < metrics_vals[i][1]): # don't assume equal
                # Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]
    
pareto = identify_pareto(metrics_vals)
print('Pareto front index vales')
print('Points on Pareto front: \n', pareto)
print(type(pareto))
for i in pareto:
    print(pareto_list[i])

pareto_front = metrics_vals[pareto]
print('\nPareto front scores')
print(pareto_front)

pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values

x_all = metrics_pareto[:, 0]
y_all = metrics_pareto[:, 1]
x_pareto = pareto_front[:, 0]
y_pareto = pareto_front[:, 1]

z_all = metrics_pareto[:, 2]

# import csv
# filename = "mcmc_metrics.csv"

# with open(filename, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Method", "Separability", "Neighborhood hit", "Spread"])
#     for i in range (len(x_all)):
#         writer.writerow([str(pareto_list[i]), str(round(z_all[i],4))+u"\u00B1"+str(round(zerr[i],3)), \
#         str(round(x_all[i],4)) + "±" + str(round(xerr[i],3)), \
#         str(round(y_all[i],4)) + "±" + str(round(yerr[i],3)) ])

filename = "metrics.txt"

with open(filename, "w") as text_file:
    text_file.write("Method  Separability  Neighborhood hit  Spread  (± for uncertainty (std dev)) \n")
    for i in range (len(x_all)):
        text_file.write(str(pareto_list[i]) + "  ")
        text_file.write(str(round(z_all[i],4)) + "±" + str(round(zerr[i],3)) + "  ")
        text_file.write(str(round(x_all[i],4)) + "±" + str(round(xerr[i],3)) + "  ")
        text_file.write(str(round(y_all[i],4)) + "±" + str(round(yerr[i],3)) + "\n")
        
# input("stop")

#uncertainty equal to the standard deviation

# # draw a scatterplot with annotations
# fig, ax = plt.subplots() # just a figure and only one subplot
# # ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o') # for std
# sc = ax.scatter(x_all, y_all)

names = pareto_list # all unique models

col = []
for index in range(10):
    col.append(list(plt.cm.tab10(index)))

marker_size = 100

# draw a scatterplot with annotations
fig, ax = plt.subplots() # just a figure and only one subplot
plt.grid()

for i in range(len(x_all)):
    if ('_ae' in names[i]):
        sc = ax.scatter(x_all[i], y_all[i], c=col[4], marker='v', s=marker_size)
    if ('vae' in names[i]):
        sc = ax.scatter(x_all[i], y_all[i], c=col[0], marker='o', s=marker_size) # ~blue
    if ('wae' in names[i]):
        sc = ax.scatter(x_all[i], y_all[i], c=col[1], marker='s', s=marker_size)

models = []
for name in names: # replace some characters
    name = name.replace('beta_', 'beta4 ')
    name = name.replace('_', ' ')
    name = name.replace('cropped', 'crop')
    name = name.replace('norm crop', 'crop norm')
    name = name.replace(' crop', '')
    name = name.replace(' lrelu', '')
    name = name.replace(' relu', '')
    name = name.replace(' norm', '')
    name = name.replace('beta', r'$\beta$')
    #print(name)
    name = name.replace('2d', '2D')
    name = name.replace('3d', '3D')
    name = name.replace('vae', '') # VAE
    name = name.replace('wae', '') # SWAE
    name = name.replace('ae', 'AE')
    name = name.replace('2D', '')
    if 'WAE' in name:
        name = name.replace('reg', '')
    if 'reg' in name:
        name = name.replace('AE', '') # Sparse AE
        name = name.replace('reg', '')
    if 'baseline' in name:
        name = name.replace('baseline', '') # Baseline
    models.append(name)

    name = name.replace('  ', ' ')
# print("after repl:", names)
# print("after repl:", models)

# for i, txt in enumerate(models):
#     print(txt)
#     annot_all = ax.annotate(txt, xy=(x_all[i], y_all[i]), 
#         xytext=(0,5), 
#         ha='center',
#         textcoords="offset points", fontsize=10)

annot = ax.annotate("", xy=(0,0), xytext=(0,5), ha='center',
            bbox=dict(boxstyle="round", fc="w")) # arrowprops=dict(arrowstyle="->"))
annot.set_visible(True)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    try:
        text = "{}".format(" ".join([models[n] for n in ind["ind"]]))
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
            #annot_all.set_visible(False)
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                #annot_all.set_visible(False)
                annot.set_visible(True) #
                fig.canvas.draw_idle()

# fig.canvas.mpl_connect("motion_notify_event", hover)

ax.plot(x_pareto, y_pareto, color=col[2], linewidth=3) # connecting green line for pareto front
# ax.scatter(x_pareto, y_pareto, c=col[2], marker='D', s=marker_size+25)
plt.scatter(x_pareto, y_pareto, facecolors='none', edgecolors=col[2], s=marker_size+100)

# ax.plot(x_pareto, y_pareto, marker='o', c=col[2], s=marker_size+25,
#          linestyle='-', 
#          markevery=[-1], 
#          markerfacecolor='white', 
#          markeredgewidth=1.5)

for i in range(len(pareto_list)):
    if ("baseline" in pareto_list[i]):
        #print(pareto_list[i])
        #print(x_all[i], y_all[i])
        ax.scatter(x_all[i], y_all[i], c='r', s=marker_size) # baseline

# ZoomPan scrollig
scale = 1.1
zp = ZoomPan()
figZoom = zp.zoom_factory(ax, base_scale = scale)
figPan = zp.pan_factory(ax)

# ax.set_xlim([0.24,0.82])
#ax.set_ylim([ymin,ymax])
fs = 16
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.xlabel('Neighborhood hit', fontsize=fs)
plt.ylabel('Spread', fontsize=fs)
# plt.suptitle("Drop Dynamics, Pareto frontier", fontsize=22)

texts = [ax.text(x_all[i], y_all[i], models[i], fontsize=13) for i in range(len(x_all))]
for i, text in enumerate(texts):
    print(i, text)

adjust_text(texts, lim=1, precision=0.001)
# adjust_text(texts, lim=1, arrowprops=dict(arrowstyle="->", color='b', lw=0.5))
# adjust_text(texts, x, y, arrowprops=dict(arrowstyle="-", color='black', lw=0.5), 
#             autoalign='', only_move={'points':'y', 'text':'y'})

# texts[4].set_y(0.1415) # beta2 VAE 128
# texts[12].set_y(0.138) # beta4 VAE 128
# texts[18].set_y(0.153) # beta4 VAE 128
texts[4].set_x(0.9915) # beta2 VAE 128
texts[4].set_y(0.144) # beta2 VAE 128
texts[12].set_x(0.986) # b4 128
texts[14].set_x(0.994) # 128
texts[18].set_x(0.997) # SWAE 2
texts[18].set_y(0.15) # SWAE 2

import matplotlib.lines as mlines
purple_triangle = mlines.Line2D([], [], color=col[4], marker='v', linestyle='None',
                          markersize=10, label='Sparse AE')
orange_square = mlines.Line2D([], [], color=col[1], marker='s', linestyle='None',
                          markersize=10, label='SWAE')
blue_circle = mlines.Line2D([], [], color=col[0], marker='o', linestyle='None',
                          markersize=10, label='VAE')
red_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='Baseline')
green_ = mlines.Line2D([], [], color="white", marker='o', markeredgecolor=col[2], linestyle='None',
                          markersize=10, label='Pareto efficient')

plt.legend(handles=[purple_triangle, orange_square, blue_circle, red_circle, green_], prop={'size': 13})

# fig.set_size_inches(16, 9)
fig.set_size_inches(8, 8)
plt.tight_layout()
# plt.savefig('mcmc_ae_vae_wae_pareto.png', dpi=300)
plt.savefig('mcmc_ae_vae_wae_pareto.pdf')  
plt.show()


# Show the baseline: one point (red)
#dirName = 'Baseline'


# in the end to have a visualization with point for each dir (our analysis)
# and correstonding folder (shown as a list or in the vis interactively)

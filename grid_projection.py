import sys
import matplotlib.pylab as mp
import time
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
#from vector import CGaussKernel,CLinearKernel,CRBFKernel
from numpy.random import randn,rand
#import lap # LAP solvers
import pdb

def KS(X_1, X_2): # KS(imgdata,griddata.T)

    init_type = 'eig' # random, eig, ... 
    omegas = 1.0
    llambda = 1.0 # step size
    n_iter = 30 # number of LAP iterations
    n_obs = X_1.shape[0] # number of observations
    bases = np.eye(n_obs) 
    
    # compute the kernel matrix
    starttime = time.time()
    # dk = CRBFKernel();
    # dl = CRBFKernel();
    # dK = dk.Dot(X_1, X_1)
    # dK = np.dot(X_1, X_1)
    # dL = np.dot(X_2, X_2)
    # omega_K = 1.0*omegas / np.median(dK.flatten())
    # omega_L = 1.0 / np.median(dL.flatten())	
    
    # #kernel_K = CGaussKernel(omega_K)
    # kernel_K = omega_K
    # #kernel_L = CGaussKernel(omega_L)
    # kernel_L = omega_L
    
    # K = kernel_K.Dot(X_1,X_1) # should use incomplete cholesky instead ...
    # L = kernel_L.Dot(X_2,X_2) # should use incomplete cholesky instead ...
    
    K = np.dot(X_1, X_1.T)
    L = np.dot(X_2, X_2.T)

    # print("X_1:", X_1)
    # print("K:", K)
    # print("X_2:", X_2)
    # print("L:", L)
    
    stoptime = time.time()
    #print 'computing kernel matrices (K and L) takes %f seconds '% (stoptime-starttime)   
    
    #print( 'original objective function : ')
    H = np.eye(n_obs) - np.ones(n_obs)/n_obs
    #print( np.trace(np.dot(np.dot(np.dot(H,K),H),L)) ) 
    
    # initializing the permutation matrix
    if (cmp(init_type,'random') == 0):
        #print( 'random initialization is being used...' )
        PI_0 = init_random(n_obs)
    elif (cmp(init_type,'eig') == 0):
        #print( 'sorted eigenvector initialization is being used... ')
        PI_0 = init_eig(K,L,n_obs)
    #else:
        #print( 'wrong initialization type... ' )

    # centering of kernel matrices    
    H = np.eye(n_obs) - np.ones(n_obs)/n_obs
    K = np.dot(H,np.dot(K,H))
    L = np.dot(H,np.dot(L,H))

    #print( 'initial objective: ' )
    #print( np.trace(np.dot(np.dot(np.dot(PI_0,K),PI_0.T),L)) )  

    # iterative linear assignment solution
    PI_t = np.zeros((n_obs,n_obs))
    for i in range(n_iter):
        #print( 'iteration : ',i )
        #starttime = time.clock()        
        grad = compute_gradient(K,L,PI_0) # L * P_0 * K
        #print('grad:', grad)
        #stoptime = time.clock()
        #print 'computing gradient takes %f seconds '% (stoptime-starttime)
        
        # convert grad (profit matrix) to cost matrix
        # assuming it is a finite cost problem, thus 
        # all the elements are substracted from the max value of the whole matrix
        cost_matrix = grad
        #starttime = time.clock()    
        from scipy.optimize import linear_sum_assignment   
        indexes = linear_sum_assignment(cost_matrix)[1] 
        #print(indexes)
        #indexes = lap.lapjv(cost_matrix)[1]
        #indexes = hungarian.hungarian(cost_matrix)[0]
        indexes = np.array(indexes)
        #print(indexes)
        
        #stoptime = time.clock()
        #print 'lap solver takes %f seconds '% (stoptime-starttime)
        
        PI_t = np.eye(n_obs)
        PI_t = PI_t[indexes,]
        
        # convex combination
        PI = (1-llambda)*PI_0 + (llambda)*PI_t
        # gradient ascent
        #PI = PI_0 + llambda*compute_gradient(K,L,PI_t)

        #print(PI_0.shape)
        #print(PI_t.shape, K.shape, L.shape)

        # computing the objective function
        obj_funct = np.trace(np.dot(np.dot(np.dot(PI_t,K),PI_t.T),L))
        #print( 'objective function value : ',obj_funct )
        #print( '\n' )

        # another termination criteria
        if (np.trace(np.dot(np.dot(np.dot(PI,K),PI.T),L)) - np.trace(np.dot(np.dot(np.dot(PI_0,K),PI_0.T),L)) <= 1e-5):
                PI_final = PI_t
                break
      
        PI_0 = PI
        if (i == n_iter-1):
            PI_final = PI_t
            
    print("iterative linear assignment was completed in", i, "iterations")
    return PI_final

def compute_gradient(K,L,PI_0):
    grad = np.dot(L,np.dot(PI_0,K))
    grad = 2*grad
    return grad
 
def init_eig(K,L,n_obs):
    # with sorted eigenvectors
    [U_K,V_K] = np.linalg.eig(K)
    [U_L,V_L] = np.linalg.eig(L)
    i_VK = np.argsort(-V_K[:,0])
    i_VL = np.argsort(-V_L[:,0])
    PI_0 = np.zeros((n_obs,n_obs))
    PI_0[np.array(i_VL),np.array(i_VK)] = 1
    return PI_0

def init_random(n_obs):
    # random initialization
    bases = np.eye(n_obs)    
    init = np.random.permutation(n_obs)
    PI_0 = bases[init,:]
    return PI_0

def cmp(a, b):
    return (a > b) - (a < b)


def grid_projection(encoded_vec_2d, test_data, names, dataset, dir_res_model, title, proj, temporal):

    if (dataset == "mnist"):
        return
    
    ### Projection to regular grid (Kernelized sorting)
    print("Grid Projection...")
    print("data:", test_data.shape) # (300, 3, 160, 224, 1)

    if(dataset == 'flow'):
        psize_x = 309 # 441 309 denosing or baseline
        if("Raw data" in title): # baseline
            psize_x = 309
        psize_y = 84
        ino = 6
        jno = 50
    elif(dataset == 'droplet'):
        # before - 160 168
        psize_x = 132
        psize_y = 146 # 224 168 denosing or baseline
        # if("Raw data" in title): # baseline
            # psize_y = 168
        ino = 30 # 30
        jno = 40 # 40
    elif(dataset == 'mcmc'):
        psize_x = 50
        psize_y = 50
        ino = 50 #
        jno = 50 #

    griddata = np.zeros((2,ino*jno))
    griddata[0,] = np.kron(range(1,ino+1),np.ones((1,jno)))
    griddata[1,] = np.tile(range(1,jno+1),(1,ino))
    #print(griddata.shape)

    encoded_vec_2d = encoded_vec_2d[:ino*jno,]
    test_data = test_data[:ino*jno,]

    #print(test_data)
    vmin = test_data.min()
    vmax = test_data.max()
    print("vmin, vmax:", vmin, vmax)

    # do kernelized sorting procedure
    PI = KS(encoded_vec_2d, griddata.T)
    #print("PI:",PI)
    i_sorting = PI.argmax(axis=1)
    #print(i_sorting)
    #print(i_sorting.shape)

    #encoded_vec_2d_sorted = encoded_vec_2d[i_sorting,]

    # now create a regular plot and put images with sorted indices:
    encoded_vec_2d_sorted = encoded_vec_2d[i_sorting,]

    imgdata_sorted = test_data[i_sorting,]


    vmin = imgdata_sorted.min()
    vmax = imgdata_sorted.max()
    print('range of data:', vmin, vmax)
    if (dataset == "flow"):
        vmax = 70 # uncropped - 80
    if (dataset == "droplet"):
        vmin = -1.5
        vmax = 1.5
    print('range of colormap:', vmin, vmax)
    #imgdata_sorted = test_data # without sorting
    #print(imgdata_sorted.shape)
    #print(imgdata_sorted)

    # we have only 1 channel for all data.
    print("data to visualize:", imgdata_sorted.shape) # (300, 3, 160, 224, 1)

    # border_width = 5
    # w = imgdata_sorted.shape[3]
    # h = imgdata_sorted.shape[2]

    # from PIL import Image, ImageOps

    # # Add border and save
    # imgdata_sorted = ImageOps.expand(imgdata_sorted, border=5, fill=(0,0,0))

    # for x in range(w):
    #     for y in range(h):
    #         if (x<border_width
    #             or y<border_width 
    #             or x>w-border_width-1 
    #             or y>h-border_width-1):
    #                 imgdata_sorted[:,:,y,x,0] = (0.4,0.4,0.4) # color

    # (1.5,1.5,1.5) yellow
    # (0.8,0.8,0.8) green
    # (0.4,0.4,0.4) blue
    # (-0.6,-0.6,-0.6)
    # (-0.8,-0.8,-0.8)
    # (-1.1,-1.1,-1.1) 
    # (-1.3,-1.3,-1.3) 
    
    irange = range(0,psize_x*ino,psize_x)
    jrange = range(0,psize_y*jno,psize_y)
    #patching = np.zeros((ino*psize_x, jno*psize_y, channels))
    patching = np.zeros((ino*psize_x, jno*psize_y))
    for i in range(ino):
        for j in range(jno):
            if (temporal):
                patching[irange[i]:irange[i]+psize_x, jrange[j]:jrange[j]+psize_y] = \
                    np.reshape(imgdata_sorted[(i)*jno+j,1,], [psize_x, psize_y]) # middle image
            else:
                patching[irange[i]:irange[i]+psize_x, jrange[j]:jrange[j]+psize_y] = \
                    np.reshape(imgdata_sorted[(i)*jno+j,], [psize_x, psize_y])

                    # try fig.add_subplot(rows, columns, i+columns)

    #plt.show()
    # print(patching.shape)
    # patching.resize(patching.shape[0], patching.shape[1])
    # print(patching.shape)
    # from PIL import Image, ImageDraw
    # im = Image.fromarray(patching.astype(np.uint8))
    # im.show()

    # vmin = patching.min()
    # vmax = patching.max()
    # print("vmin, vmax:", vmin, vmax)

    fig=plt.figure()
    title += ", frames on grid"
    plt.suptitle(title, fontsize=15)
    plt.axis('off')
    cmap = 'viridis'
    # if (dataset == "droplet"):
    #     cmap='gray'

    plt.imshow(patching, cmap=cmap, vmin=vmin, vmax=vmax)
    # if(dataset == 'droplet'):
    #     #plt.imshow(patching.astype(np.uint8)) #, cmap='viridis', vmin=vmin, vmax=vmax)
    #     plt.imshow(patching, cmap='viridis', vmin=vmin, vmax=vmax)
    # if(dataset == 'flow'):
    #     plt.imshow(patching.astype(np.uint8), cmap='viridis', vmin=vmin, vmax=80)
    #plt.show()
    #plt.colorbar()
    plt.tight_layout()
    # fig.set_size_inches(10, 8)
    if (proj == "tsne"):
        fig.savefig('{}/latent_tsne_grid.png'.format(dir_res_model), dpi=300)
    if (proj == "umap"):
        fig.savefig('{}/latent_umap_grid.png'.format(dir_res_model), dpi=300)
    plt.close(fig)

    # if(dataset == 'droplet'):
    #     names = [names[i] for i in i_sorting]
    # print(names)
    #     # 132 146
    #     import matplotlib.patches as patches
    #     fig, ax = plt.subplots()
    #     ax.imshow(patching, cmap=cmap, vmin=vmin, vmax=vmax)
    #     rect = patches.Rectangle((146+1, 132+1), 146-5, 132-5, linewidth=1, edgecolor='yellow', facecolor='none')
    #     ax.add_patch(rect)
    #     plt.show()
    #     fig.savefig('{}/latent_umap_grid_frame.png'.format(dir_res_model), dpi=300)

    #     fig, ax = plt.subplots()
    #     ax.imshow(patching, cmap=cmap, vmin=vmin, vmax=vmax)
    #     k = 0
    #     for i in range(ino):
    #         for j in range(jno):
    #             # print(names[k])
    #             if names[k]=='bubble': c='indigo' 
    #             if names[k]=='bubble-splash': c='purple'
    #             if names[k]=='column': c='orange'
    #             if names[k]=='crown': c='darkblue'
    #             if names[k]=='crown-splash': c='mediumblue'
    #             if names[k]=='drop': c='limegreen'
    #             if names[k]=='none': c='yellow'
    #             if names[k]=='splash': c='dodgerblue'
    #             rect = patches.Rectangle((j*(146), i*(132)), 146-5, 132-5, linewidth=0.5, edgecolor=c, facecolor='none')
    #             ax.add_patch(rect)
    #             k += 1
    #     print(k)
    #     plt.show()
    #     fig.savefig('{}/latent_umap_grid_frames.png'.format(dir_res_model), dpi=300)

    if(dataset == 'mcmc'):
        names = [names[i] for i in i_sorting]
        # print(names)
        # 50 50
        import matplotlib.patches as patches
        fig, ax = plt.subplots()
        ax.imshow(patching, cmap=cmap, vmin=vmin, vmax=vmax)
        k = 0
        for i in range(ino):
            for j in range(jno):
                # print(names[k])
                if names[k]=='2': c='purple' 
                if names[k]=='3': c='mediumblue'
                if names[k]=='4': c='orange'
                if names[k]=='5': c='limegreen'
                if names[k]=='1': c='yellow'
                rect = patches.Rectangle((j*(50), i*(50)), 50-2, 50-2, linewidth=0.5, edgecolor=c, facecolor='none')
                ax.add_patch(rect)
                k += 1
        print(k)
        plt.show()
        fig.savefig('{}/latent_umap_grid_frames.png'.format(dir_res_model), dpi=300)

    if(dataset == "droplet"):
        names = [names[i] for i in i_sorting]
        print(names)
        # 30 40
        # 132 146
        import matplotlib.patches as patches
        fig, ax = plt.subplots()
        ax.imshow(patching, cmap=cmap, vmin=vmin, vmax=vmax)
        k = 0
        for i in range(ino):
            for j in range(jno):
                # print(names[k])
                if names[k]=='bubble': c='darkviolet' 
                if names[k]=='bubble-splash': c='magenta'
                if names[k]=='column': c='orange'
                if names[k]=='crown': c='darkblue'
                if names[k]=='crown-splash': c='royalblue'
                if names[k]=='splash': c='mediumturquoise'
                if names[k]=='drop': c='limegreen'
                if names[k]=='none': c='yellow'
                rect = patches.Rectangle((j*(146), i*(132)), 146-2, 132-2, linewidth=0.5, edgecolor=c, facecolor='none')
                ax.add_patch(rect)
                k += 1
        print(k)
        plt.show()
        fig.savefig('{}/latent_umap_grid_frames.png'.format(dir_res_model), dpi=300)

import os, re

def model_directories(dir_res, model_name):
    # create directory to save results

    if re.search("\d{3}.h5", model_name):
        model = model_name[:-7]
    elif re.search("\d{2}.h5", model_name):
        model = model_name[:-6]
    else:
        model = model_name[:-5]

    dir_res = os.path.join(dir_res, model)
    print("Saved here:", dir_res)
    model = model_name[:-3]
    dir_res_model = os.path.join(dir_res, model)

    return dir_res_model

def models_metrics_stability(mod_nam, dataset):
    # for metrics stability evaluation
    model_names_all = []
    num_of_runs = 20
    if (dataset == "droplet"):
        start = 20 # 400   60 180 300
        end = 100 # 2400
        step = 40 # 400
    if (dataset == "mcmc"):
        start = 20 # 500
        end = 50 # 2500
        step = 30 # 250
    for lab in range(start,end+step, step): # labels to consider
        #print(i)

        for m_n in mod_nam:
            for i in range(num_of_runs):    
                m_n_index = m_n + "_" + str(lab) + "_" + str(i+1) + ".h5"
                model_names_all.append(m_n_index)

    model_names = model_names_all
    print(model_names)

    return model_names

def model_name_metrics_stability(model_name, x_test, names, dataset):
    # metrics stability, remove numbers
    if (dataset == "droplet"):
        start = 20 # 400   60 180 300
        end = 100 # 2400
        step = 40 # 400
    if (dataset == "mcmc"):
        start = 20 # 500
        end = 50 # 2500
        step = 30 # 250
    for lab in reversed(range(start,end+step, step)):
        # print(lab)
        to_remove = "_" + str(lab)
        if to_remove in model_name[:-5]:
            x_test_ = x_test[:lab,...] # #labels to consider
            names_ = names[:lab] # #labels to consider
            print("Labels considered:", x_test_.shape[0])
            model_name = model_name[:-5].replace(to_remove, '') + model_name[-5:]
    print(model_name)


    if re.search("\d{3}.h5", model_name):
        x = re.search("\d{3}.h5", model_name).group(0)[:-3]
        dir_model_name = os.path.join("weights", model_name[:-6])
        model_name = model_name[:-7]
    elif re.search("\d{2}.h5", model_name):
        x = re.search("\d{2}.h5", model_name).group(0)[:-3]
        dir_model_name = os.path.join("weights", model_name[:-5])
        model_name = model_name[:-6]
    elif (re.search("\d{1}.h5", model_name)):
        x = re.search("\d{1}.h5", model_name).group(0)[:-3]
        dir_model_name = os.path.join("weights", model_name[:-4])
        model_name = model_name[:-5]

    weight_num = int(x)%5+1
    print(weight_num)

    dir_model_name += str(weight_num) + ".h5"
    print("Weights: ", dir_model_name)

    model_name += ".h5"

    return model_name, dir_model_name, x_test_, names_


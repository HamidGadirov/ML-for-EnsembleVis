
# Keract visualizations
def visualize_keract(x_train, encoder, decoder):

    from keract import get_activations, display_activations, display_heatmaps
    from keract import get_gradients_of_trainable_weights, get_gradients_of_activations

    keract_inputs = x_train[:1]
    keract_targets = x_train[:1]

    print('encoder activations')
    layer_name='conv2d_3'

    activations = get_activations(encoder, keract_inputs, layer_name)
    print(activations[layer_name].shape)

    #dec_input = activations[layer_name]
    #plt.show(dec_input[:,:,:,0])

    display_activations(activations, cmap="binary", save=False) # for all filters
    display_heatmaps(activations, keract_inputs, save=False) # for all filters overlayed on input_img

    # gradients_of_trainable_weights
    #weights_grads = get_gradients_of_trainable_weights(encoder, keract_inputs, keract_targets)
    #print(weights_grads[layer_name].shape)

    layer_name='latent_vector'

    activations = get_activations(encoder, keract_inputs, layer_name)
    dec_input = activations[layer_name]
    print('activations')
    display_activations(activations, cmap="binary", save=False)
    print('heatmaps')
    display_heatmaps(activations, keract_inputs, save=False)


    print('decoder activations')
    layer_name='conv2d_transpose_3'

    activations = get_activations(decoder, dec_input, layer_name)
    print(activations[layer_name].shape)
    print('activations')
    display_activations(activations, cmap="binary", save=False)
    print('heatmaps')
    display_heatmaps(activations, keract_inputs, save=False)


    ### An exploration of convnet filters with Keras

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in encoder.layers])
    
    from keras import backend as K

    layer_name = 'conv2d_3'
    filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    input_img = x_train[:1]

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    print(grads)

    # normalization trick: we normalize the gradient
    #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    #iterate = K.function([input_img], [loss, grads])


# Keras-vis
def visualize_keras(x_train, encoder, decoder):
    
    from vis.utils import utils
    from keras import activations

    layer_name='conv2d_3'

    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(encoder, layer_name)

    # Swap softmax with linear
    #model.layers[layer_idx].activation = activations.linear
    #model = utils.apply_modifications(model)

    from vis.visualization import visualize_activation

    from matplotlib import pyplot as plt
    #plt.rcParams['figure.figsize'] = (18, 6)

    # 20 is the imagenet category for 'ouzel'
    #img = visualize_activation(encoder, layer_idx, filter_indices=0)
    #plt.imshow(img)


    from vis.visualization import visualize_saliency, overlay

    img = x_train[:1]
    grads = visualize_saliency(encoder, layer_idx, filter_indices=0, seed_input=img)
    
    # visualize grads as heatmap
    plt.imshow(grads, cmap='jet')


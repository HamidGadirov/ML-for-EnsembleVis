# ML-for-EnsembleVis
Autoencoder-based Feature Extraction for Ensemble Visualization

In this master’s thesis, we investigate machine learning methods to support the visu-
alization of ensemble data. Our goal is to develop methods that allow us to efficiently
explore the projections of various ensemble datasets and investigate the ability of
autoencoder-based techniques to extract high-level data features. This enables clus-
tering of data samples on the projections according to their behavior modes. First,
we apply unsupervised feature learning techniques, such as autoencoders or varia-
tional autoencoders, to ensemble members for high-level feature extraction. Then,
we perform a projection from the extracted feature space for scatterplot visualiza-
tion. In order to obtain quantitative results, in addition to qualitative, we develop
metrics for evaluation of the resulting projections. After that, we use the quantita-
tive results to obtain a set of Pareto efficient models. We evaluate various feature
learning methods and projection techniques, and compare their ability of extracting
expressive high-level data features. Our results indicate that the learned unsuper-
vised features improve the clustering on the final projections. Autoencoders and
(beta-)variational autoencoders with properly selected parameters are capable of
extracting high-level features from ensembles. The combination of metrics allow
us to evaluate the resulting projections. We summarize our findings by offering
practical suggestions for applying autoencoder-based techniques to ensemble data.

# How to run code to reproduce results:

1. Go to the corresponding directory of the dataset (Drop Dynamics/Vortex Street/MNIST).

2. Open the corresponding file (e.g. 3d_vae.py, where "3d" stands for 3D convolutional model and "vae" for Variational Autoencoder).
It is possible to select the model(s) for testing via "model_names" variable.

3. Before executing, it is necessary to unzip the test data files (reconstruction, interpolation, all visualizations with time steps).

Results will be saved e.g. in the Results/3D_VAE/"model_name" folder.

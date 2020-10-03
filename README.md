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

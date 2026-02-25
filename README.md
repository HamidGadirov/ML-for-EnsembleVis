## [ISVC 2021] Evaluation and Selection of Autoencoders for Expressive Dimensionality Reduction of Spatial Ensembles

[Hamid Gadirov](https://hamidgadirov.github.io), [Gleb Tkachev](https://gleb-t.com/), [Thomas Ertl](https://scholar.google.com/citations?user=qFQ9jHkAAAAJ&hl=en), [Steffen Frey](https://freysn.github.io/)  
*Springer, International Symposium on Visual Computing (ISVC), 2021*  
[Paper (PDF)](https://gleb-t.com/publication/ae-ensemble/ae-ensemble.pdf)

Official implementation of the ISVC 2021 paper *"Evaluation and Selection of Autoencoders for Expressive Dimensionality Reduction of Spatial Ensembles"*.

### BibTeX
```bibtex
@inproceedings{gadirov2021evaluation,
  title={Evaluation and selection of autoencoders for expressive dimensionality reduction of spatial ensembles},
  author={Gadirov, Hamid and Tkachev, Gleb and Ertl, Thomas and Frey, Steffen},
  booktitle={International Symposium on Visual Computing},
  pages={222--234},
  year={2021},
  organization={Springer}
}
```

This paper evaluates how autoencoder variants with different architectures and parameter settings affect the quality of 2D projections for spatial ensembles, and proposes a guided selection approach based on partially labeled data. Extracting features with autoencoders prior to applying techniques like UMAP substantially enhances the projec- tion results and better conveys spatial structures and spatio-temporal behavior. Our comprehensive study demonstrates substantial impact of different variants, and shows that it is highly data-dependent which ones yield the best possible projection results. We propose to guide the se- lection of an autoencoder configuration for a specific ensemble based on projection metrics. These metrics are based on labels, which are however prohibitively time-consuming to obtain for the full ensemble. Address- ing this, we demonstrate that a small subset of labeled members suffices for choosing an autoencoder configuration. We discuss results featuring various types of autoencoders applied to two fundamentally different en- sembles featuring thousands of members: channel structures in soil from Markov chain Monte Carlo and time-dependent experimental data on droplet-film interaction.



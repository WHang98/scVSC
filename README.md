##  scVSC: Deep variable subspace clustering based on single-cell transcriptome data
Single-cell RNA sequencing (scRNA-seq) strives to capture cellular diversity with higher resolution images than bulk RNA sequencing. Clustering analysis is a crucial step as it provides an opportunity to further identify and uncover undiscovered cell types. Most existing clustering methods support unsupervised clustering but cannot integrate prior information. When faced with the high dimensionality of scRNA-seq data and common dropout events, purely unsupervised clustering methods may fail to produce biologically interpretable clusters, which complicates cell type assignment. Here, we propose scVSC, a semi-supervised clustering model for scRNA sequence analysis using deep generative neural networks. Specifically, scVSC carefully designs a ZINB loss-based autoencoder architecture that inherently integrates adversarial training and semi-supervised modules in the latent space. In a series of experiments on scRNA-seq datasets spanning thousands to tens of thousands of cells, scVSC can significantly improve clustering performance compared to dozens of unsupervised and semi-supervised algorithms, promoting clustering and interpretability of downstream analyses. 

## Requirements:
Python --- 3.8.13
pytorch -- 1.11.0
Scanpy --- 1.0.4

Nvidia Tesla P40

### Usage

Step 1: Prepare pytorch environment. See [Pytorch](https://pytorch.org/get-started/locally/).

Step 2: Prepare data. Download all data from https://github.com/WHang98/scVSC.
The public datasets in the paper are stored in the `Data` folder as .h5 or .csv  files.

Step 3: Data preprocess.
```
process.py #Load and process the data
```
Step 4: Run on the scRNA-seq datasets
```
scVSC.py #implementation of scVSC algorithm
```



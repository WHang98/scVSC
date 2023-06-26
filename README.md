##  scVSC: Deep variable subspace clustering based on single-cell transcriptome data
Single-cell RNA sequencing (scRNA-seq) is a potent advancement for analyzing gene expression at the individual cell level, allowing for the identification of cellular heterogeneity and subpopulations. However, it suffers from technical limitations that result in sparse and heterogeneous data. Here, we propose scVSC, an unsupervised clustering algorithm built on deep representation neural networks. The method incorporates the variational inference into the subspace model, which imposes regularization constraints on the latent space and further prevents overfitting.In a series of experiments across multiple datasets, scVSC outperforms existing state-of-the-art unsupervised and semi-supervised clustering tools regarding clustering accuracy and running efficiency. Moreover, the study indicates that scVSC could visually reveal the state of trajectory differentiation, accurately identify differentially expressed genes, and further discover biologically critical pathways. 

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



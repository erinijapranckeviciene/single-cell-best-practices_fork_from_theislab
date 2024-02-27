#!/usr/bin/env python
# coding: utf-8

# (pre-processing:dimensionality-reduction)=
# # Dimensionality Reduction
# 
# As previously mentioned, scRNA-seq is a high-throughput sequencing technology that produces datasets with high dimensions in the number of cells and genes. This immediately points to the fact that scRNA-seq data suffers from the 'curse of dimensionality'. 
# 
# ```{admonition} Curse of dimensionality
# The Curse of dimensionality was first brought up by R. Bellman {cite}`bellman1957dynamic` and descibes the problem that in theory high-dimensional data contains more information, but in practice this is not the case. Higher dimensional data often contains more noise and redundancy and therefore adding more information does not provide benefits for downstream analysis steps. 
# ```
# 
# Not all genes are informative and are important for the task of cell type clustering based on their expression profiles. We already aimed to reduce the dimensionality of the data with feature selection, as a next step one can further reduce the dimensions of single-cell RNA-seq data with dimensionality reduction algorithms. These algorithms are an important step during preprocessing to reduce the data complexity and for visualization. Several dimensionality reduction techniques have been developed and used for single-cell data analysis.
# 
# :::{figure-md} Dimensionality reduction
# 
# <img src="../_static/images/preprocessing_visualization/dimensionality_reduction.jpeg" alt="Dimensionality reduction" class="bg-primary mb-1" width="800px">
# 
# Dimensionality reduction embeds the high-dimensional data into a lower dimensional space. The low-dimensional representation still captures the underlying structure of the data while having as few as possible dimensions. Here we visualize a three dimensional object projected into two dimensions. 
# 
# :::
# 
# Xing et al. compared in an independent comparison the stability, accuracy and computing cost of 10 different dimensionality reduction methods {cite}`Xiang2021`. They propose to use t-distributed stochastic neighbor embedding (t-SNE) as it yielded the best overall performance. Uniform manifold approximation and projection (UMAP) showed the highest stability and separates best the original cell populations. An additional dimensionality reduction worth mentioning in this context is principal component analysis (PCA) which is still widely used.
# 
# Generally, t-SNE and UMAP are very robust and mostly equivalent if specific choices for the initialization are selected {cite}`Kobak2019`.
# 
# All aforementioned methods are implemented in scanpy.
# 

# In[1]:


import scanpy as sc

sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
    frameon=False,
)


# In[2]:


adata = sc.read(
    filename="s4d8_feature_selection.h5ad",
    backup_url="https://figshare.com/ndownloader/files/40016014",
)


# We will use a normalized representation of the dataset for dimensionality reduction and visualization, specifically the shifted logarithm. 

# In[3]:


adata.X = adata.layers["log1p_norm"]


# We start with:
# 
# ## PCA
# 
# In our dataset each cell is a vector of a `n_var`-dimensional vector space spanned by some orthonormal basis. As scRNA-seq suffers from the 'curse of dimensionality', we know that not all features are important to understand the underlying dynamics of the dataset and that there is an inherent redundancy{cite}`grun2014validation`. PCA creates a new set of uncorrelated variables, so called principle components (PCs), via an orthogonal transformation of the original dataset. The PCs are linear combinations of features in the original dataset and are ranked with decreasing order of variance to define the transformation. Through the ranking usually the first PC amounts to the largest possible variance. PCs with the lowest variance are discarded to effectively reduce the dimensionality of the data without losing information.
# 
# PCA offers the advantage that it is highly interpretable and computationally efficient. However, as scRNA-seq datasets are rather sparse due to dropout events and therefore highly non-linear, visualization with the linear dimensionality reduction technique PCA is not very appropriate. PCA is typically used to select the top 10-50 PCs which are used for downstream analysis tasks.
# 

# In[4]:


# setting highly variable as highly deviant to use scanpy 'use_highly_variable' argument in sc.pp.pca
adata.var["highly_variable"] = adata.var["highly_deviant"]
sc.pp.pca(adata, svd_solver="arpack", use_highly_variable=True)


# In[5]:


sc.pl.pca_scatter(adata, color="total_counts")


# 
# ## t-SNE
# 
# t-SNE is a graph based, non-linear dimensionality reduction technique which projects the high dimensional data onto 2D or 3D components. The method defines a Gaussian probability distribution based on the high-dimensional Euclidean distances between data points. Subsequently, a Student t-distribution is used to recreate the probability distribution in a low dimensional space where the embeddings are optimized using gradient descent.

# In[6]:


sc.tl.tsne(adata, use_rep="X_pca")


# In[7]:


sc.pl.tsne(adata, color="total_counts")


# 
# ## UMAP
# 
# UMAP is a graph based, non-linear dimensionality reduction technique and principally similar to t-SNE. It constructs a high dimensional graph representation of the dataset and optimizes the low-dimensional graph representation to be structurally as similar as possible to the original graph.
# 
# 
# We first calculate PCA and subsequently a neighborhood graph on our data.

# In[8]:


sc.pp.neighbors(adata)
sc.tl.umap(adata)


# In[9]:


sc.pl.umap(adata, color="total_counts")


# ## Inspecting quality control metrics 
# 
# We can now also inspect the quality control metrics we calculated previously in our PCA, TSNE or UMAP plot and potentially identify low-quality cells.

# In[10]:


sc.pl.umap(
    adata,
    color=["total_counts", "pct_counts_mt", "scDblFinder_score", "scDblFinder_class"],
)


# As we can observe, cells with a high doublet score are projected to the same region in the UMAP. We will keep them in the dataset for now but might re-visit our quality control strategy later.

# In[11]:


adata.write("s4d8_dimensionality_reduction.h5ad")


# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```

# ## Contributors
# 
# We gratefully acknowledge the contributions of:
# 
# ### Authors
# 
# * Anna Schaar
# 
# ### Reviewers
# 
# * Lukas Heumos
# 

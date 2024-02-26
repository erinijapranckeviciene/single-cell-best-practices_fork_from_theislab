#!/usr/bin/env python
# coding: utf-8

# # Data integration

# ## Motivation
# 
# A central challenge in most scRNA-seq data analyses is presented by batch effects. Batch effects are changes in measured expression levels that are the result of handling cells in distinct groups or “batches”. For example, a batch effect can arise if two labs have taken samples from the same cohort, but these samples are dissociated differently. If Lab A optimizes its dissociation protocol to dissociate cells in the sample while minimizing the stress on them, and Lab B does not, then it is likely that the cells in the data from the group B will express more stress-linked genes (JUN, JUNB, FOS, etc. see {cite}`Van_den_Brink2017-si`) even if the cells had the same profile in the original tissue. In general, the origins of batch effects are diverse and difficult to pin down. Some batch effect sources might be technical such as differences in sample handling, experimental protocols, or sequencing depths, but biological effects such as donor variation, tissue, or sampling location are also often interpreted as a batch effect {cite}`Luecken2021-jo`. Whether or not biological factors should be considered batch effects can depend on the experimental design and the question being asked. Removing batch effects is crucial to enable joint analysis that can focus on finding common structure in the data across batches and enable us to perform queries across datasets. Often it is only after removing these effects that rare cell populations can be identified that were previously obscured by differences between batches. Enabling queries across datasets allows us to ask questions that could not be answered by analysing individual datasets, such as _Which cell types express SARS-CoV-2 entry factors and how does this expression differ between individuals?_ {cite}`Muus2021-ti`.

# When removing batch effects from omics data, one must make two central choices: (1) the method and parameterization, and (2) the batch covariate. As batch effects can arise between groupings of cells at different levels (i.e., samples, donors, datasets etc.), the choice of batch covariate indicates which level of variation should be retained and which level removed. The finer the batch resolution, the more effects will be removed. However, fine batch variation is also more likely to be confounded with biologically meaningful signals. For example, samples typically come from different individuals or different locations in the tissue. These effects may be meaningful to inspect. Thus, the choice of batch covariate will depend on the goal of your integration task. Do you want to see differences between individuals, or are you focused on common cell type variation? An approach to batch covariate selection based on quantitative analyses was pioneered in a recent effort to build an integrated atlas of the human lung, where the variance attributable to different technical covariates was used to make this choice {cite}`Sikkema2022-tk`.

# ### Types of integration models
# 
# Methods that remove batch effects in scRNA-seq are typically composed of (up to) three steps:
# 1. Dimensionality reduction
# 2. Modeling and removing the batch effect
# 3. Projection back into a high-dimensional space
# 
# While modeling and removing the batch effect (Step 2) is the central part of any batch removal method, many methods first project the data to a lower dimensional space (Step 1) to improve the signal-to-noise ratio (see the {ref}`dimensionality reduction chapter <pre-processing:dimensionality-reduction>`) and perform batch correction in that space to achieve better performance (see {cite}`Luecken2021-jo`). In the third step, a method may project the data back into the original high-dimensional feature space after removing the fitted batch effect to output a batch-corrected gene expression matrix.

# Batch-effect removal methods can vary in each of these three steps. They may use various linear or non-linear dimensionality reduction approaches, linear or non-linear batch effect models, and they may output different formats of batch-corrected data. Overall, we can divide methods for batch effect removal into 4 categories. In their order of development, these are global models, linear embedding models, graph-based methods, and deep learning approaches (Fig I1).
# 
# **Global models** originate from bulk transcriptomics and model the batch effect as a consistent (additive and/or multiplicative) effect across all cells. A common example is ComBat {cite}`Johnson2007-sl`.
# 
# **Linear embedding models** were the first single-cell-specific batch removal methods. These approaches often use a variant of singular value decomposition (SVD) to embed the data, then look for local neighborhoods of similar cells across batches in the embedding, which they use to correct the batch effect in a locally adaptive (non-linear) manner. Methods often project the data back into gene expression space using the SVD loadings, but may also only output a corrected embedding. This is the most common group of methods and prominent examples include the pioneering mutual nearest neighbors (MNN) method {cite}`Haghverdi2018-bd` (which does not perform any dimensionality reduction), Seurat integration {cite}`Butler2018-js,Stuart2019-lq`, Scanorama {cite}`Hie2019-er`, FastMNN {cite}`Haghverdi2018-bd`, and Harmony {cite}`Korsunsky2019-ex`.
# 
# **Graph-based methods** are typically the fastest methods to run. These approaches use a nearest-neighbor graph to represent the data from each batch. Batch effects are corrected by forcing connections between cells from different batches and then allowing for differences in cell type compositions by pruning the forced edges. The most prominent example of these approaches is the Batch-Balanced _k_-Nearest Neighbor (BBKNN) method {cite}`Polanski2019-zy`.
# 
# **Deep learning (DL) approaches** are the most recent, and most complex methods for batch effect removal that typically require the most data for good performance. Most deep learning integration methods are based on autoencoder networks, and either condition the dimensionality reduction on the batch covariate in a conditional variational autoencoder (CVAE) or fit a locally linear correction in the embedded space. Prominent examples of DL methods are **scVI** {cite}`Lopez2018-au`, **scANVI** {cite}`Xu2021-dh`, and **scGen** {cite}`Lotfollahi2019-cy`.
# 
# Some methods can use cell identity labels to provide the method with a reference for what biological variation should not be removed as a batch effect. As batch-effect removal is typically a preprocessing task, such approaches may not be applicable to many integration scenarios as labels are generally not yet available at this stage.
# 
# More detailed overviews of batch-effect removal methods can be found in {cite}`Argelaguet2021-pb` and {cite}`Luecken2021-jo`.
# 
# ![Overview_fig](figures/integration_overview_figure.jpeg)
# Fig. I1: Overview of different types of integration methods with examples.

# ### Batch removal complexity
# 
# The removal of batch effects in scRNA-seq data has previously been divided into two subtasks: batch correction and data integration {cite}`Luecken2019-og`. These subtasks differ in the complexity of the batch effect that must be removed. Batch correction methods deal with batch effects between samples in the same experiment where cell identity compositions are consistent, and the effect is often quasi-linear. In contrast, data integration methods deal with complex, often nested, batch effects between datasets that may be generated with different protocols and where cell identities may not be shared across batches. While we use this distinction here we should note that these terms are often used interchangeably in general use. Given the differences in complexity, it is not surprising that different methods have been benchmarked as being optimal for these two subtasks.

# ### Comparisons of data integration methods
# 
# Several benchmarks have previously evaluated the performance of methods for batch correction and data integration. When removing batch effects, methods may overcorrect and remove meaningful biological variation in addition to the batch effect. For this reason, it is important that integration performance is evaluated by considering both batch effect removal and the conservation of biological variation.
# 
# The _k_-nearest-neighbor Batch-Effect Test (kBET) was the first metric for quantifying batch correction of scRNA-seq data  {cite}`Buttner2019-yl`. Using kBET, the authors found that **ComBat** outperformed other approaches for batch correction while comparing predominantly global models. Building on this, two recent benchmarks {cite}`Tran2020-ia` and {cite}`Chazarra-Gil2021-ri` also benchmarked linear-embedding and deep-learning models on batch correction tasks with few batches or low biological complexity. These studies found that the linear-embedding models **Seurat** {cite}`Butler2018-js,Stuart2019-lq` and **Harmony** {cite}`Korsunsky2019-ex` performed well for simple batch correction tasks.
# 
# Benchmarking complex integration tasks poses additional challenges due to both the size and number of datasets as well as the diversity of scenarios. Recently, a large study used 14 metrics to benchmark 16 methods across integration method classes on five RNA tasks and two simulations {cite}`Luecken2021-jo`. While top-performing methods per task differed, approaches that use cell type labels performed better across tasks. Furthermore, deep learning approaches **scANVI** (with labels), **scVI**, and **scGen** (with labels), as well as the linear embedding model **Scanorama,** performed best, particularly on complex tasks, while **Harmony** performed well on less complex tasks. A similar benchmark performed for the specific purpose of integrating retina datasets to build an _ocular mega-atlas_ also found that **scVI** outperformed other methods {cite}`Swamy2021-uy`.
# 

# ### Choosing an integration method
# 
# While integration methods have now been extensively benchmarked, an optimal method for all scenarios does not exist. Packages of integration performance metrics and evaluation pipelines like [**scIB**](https://github.com/theislab/scib) and [**batchbench**](https://github.com/cellgeni/batchbench) can be used to evaluate integration performance on your own data. However many metrics (particularly those that measure the conservation of biological variation) require ground-truth cell identity labels. Parameter optimization may tune many methods to work for particular tasks, yet in general, one can say that **Harmony** and **Seurat** consistently perform well for simple batch correction tasks, and **scVI**, **scGen**, **scANVI**, and **Scanorama** perform well for more complex data integration tasks. When choosing a method, we would recommend looking into these options first. Additionally, integration method choice may be guided by the required output data format  (i.e., do you need corrected gene expression data or does an integrated embedding suffice?). It would be prudent to test multiple methods and evaluate the outputs on the basis of quantitative definitions of success before selecting one. Extensive guidelines for data integration method choice can be found in {cite}`Luecken2021-jo`.

# In the rest of this chapter, we demonstrate some of the best-performing methods and quickly demonstrate how integration performance can be evaluated.

# In[1]:


# Python packages
import scanpy as sc
import scvi
import bbknn
import scib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# R interface
from rpy2.robjects import pandas2ri
from rpy2.robjects import r
import rpy2.rinterface_lib.callbacks
import anndata2ri

pandas2ri.activate()
anndata2ri.activate()

get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[2]:


get_ipython().run_cell_magic('R', '', '# R packages\nlibrary(Seurat)\n')


# ## Dataset
# 
# The dataset we will use to demonstrate data integration contains several samples of bone marrow mononuclear cells. These samples were originally created for the Open Problems in Single-Cell Analysis [NeurIPS Competition 2021](https://openproblems.bio/neurips_2021/) {cite}`Luecken2022-kv,Lance2022-yy`. The [10x Multiome](https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression) protocol was used which measures both RNA expression (scRNA-seq) and chromatin accessibility (scATAC-seq) in the same cells. The version of the data we use here was already pre-processed to remove low-quality cells.
# 
# Let's read in the dataset using **scanpy** to get an `AnnData` object.

# In[3]:


adata_raw = sc.read_h5ad(
    "../../datasets/openproblems_bmmc_multiome_genes_filtered.h5ad"
)
adata_raw.layers["logcounts"] = adata_raw.X
adata_raw


# The full dataset contains 69,249 cells and measurements for 129,921 features. There are two versions of the expression matrix, `counts` which contains the raw count values and `logcounts` which contains normalised log counts (these values are also stored in `adata.X`).
# 
# The `obs` slot contains several variables, some of which were calculated during pre-processing (for quality control) and others that contain metadata about the samples. The ones we are interested in here are:
# 
# * `cell_type` - The annotated label for each cell
# * `batch` - The sequencing batch for each cell
# 
# For a real analysis it would be important to consider more variables but to keep it simple here we will only look at these.
# 
# We define variables to hold these names so that it is clear how we are using them in the code. This also helps with reproducibility because if we decided to change one of them for whatever reason we can be sure it has changed in the whole notebook.

# In[4]:


label_key = "cell_type"
batch_key = "batch"


# ```{admonition} What to use as the batch label?
# 
# As mentioned above, deciding what to use as a "batch" for data integration is one of the central choices when integrating your data. The most common approach is to define each sample as a batch (as we have here) which generally produces the strongest batch correction. However, samples are usually confounded with biological factors that you may want to preserve. For example, imagine an experiment that took samples from two locations in a tissue. If samples are considered as batches then data integration methods will attempt to remove differences between them and therefore differences between the locations. In this case, it may be more appropriate to use the donor as the batch to remove differences between individuals but not between locations. The planned analysis should also be considered. In cases where you are integrating many datasets and want to capture differences between individuals, the dataset may be a useful batch covariate. In our example, it may be better to have consistent cell type labels for the two locations and then test for differences between them than to have separate clusters for each location which need to be annotated separately and then matched.
# 
# The issue of confounding between samples and biological factors can be mitigated through careful experimental design that minimizes the batch effect overall by using multiplexing techniques that allow biological samples to be combined into a single sequencing sample. However, this is not always possible and requires both extra processing in the lab and as well as extra computational steps.
# ```
# 
# Let's have a look at the different batches and how many cells we have for each.

# In[5]:


adata_raw.obs[batch_key].value_counts()


# There are 13 different batches in the dataset. During this experiment, multiple samples were taken from a set of donors and sequenced at different facilities so the names here are a combination of the sample number (eg. "s1") and the donor (eg. "d2"). For simplicity, and to reduce computational time, we will select three samples to use.

# In[6]:


keep_batches = ["s1d3", "s2d1", "s3d7"]
adata = adata_raw[adata_raw.obs[batch_key].isin(keep_batches)].copy()
adata


# After subsetting to select these batches we are left with 10,270 cells.
# 
# We have two annotations for the features stored in `var`:
# 
# * `feature_types` - The type of each feature (RNA or ATAC)
# * `gene_id` - The gene associated with each feature
# 
# Let's have a look at the feature types.

# In[7]:


adata.var["feature_types"].value_counts()


# We can see that there are over 100,000 ATAC features but only around 13,000 gene expression ("GEX") features. Integration of multiple modalities is a complex problem that will be described in the {ref}`multimodal integration chapter <multimodal-integration:advanced-integration>`, so for now we will subset to only the gene expression features. We also perform simple filtering to make sure we have no features with zero counts (this is necessary because by selecting a subset of samples we may have removed all the cells which expressed a particular feature).

# In[8]:


adata = adata[:, adata.var["feature_types"] == "GEX"].copy()
sc.pp.filter_genes(adata, min_cells=1)
adata


# Because of the subsetting we also need to re-normalise the data. Here we just normalise using global scaling by the total counts per cell.

# In[9]:


adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()


# We will use this dataset to demonstrate integration.
# 
# Most integration methods require a single object containing all the samples and a batch variable (like we have here). If instead, you have separate objects for each of your samples you can join them using the **anndata** `concat()` function. See the [concatenation tutorial](https://anndata.readthedocs.io/en/stable/concatenation.html) for more details. Similar functionality exists in other ecosystems.
# 
# ```{admonition} Integrating UMI and full-length data
# 
# Integrating samples from UMI and full-length protocols can present additional challenges. This is because full-length protocols are affected by gene-length bias (longer genes will be more highly expressed) while UMI data is not {cite}`Phipson2017-qt`. Because of this, it is generally recommended to transform counts for full-length samples into a unit that corrects for gene-length (such as transcripts per million (TPM) {cite}`Wagner2012-qf`) before attempting integration. This isn't necessary however if all the samples being integrated used a full-length protocol.
# ```

# ## Unintegrated data
# 
# It is always recommended to look at the raw data before performing any integration. This can give some indication of how big any batch effects are and what might be causing them (and therefore which variables to consider as the batch label). For some experiments, it might even suggest that integration is not required if samples already overlap. This is not uncommon for mouse or cell line studies from a single lab for example, where most of the variables which contribute to batch effects can be controlled (i.e. the batch correction setting).
# 
# We will perform highly variable gene (HVG) selection, PCA and UMAP dimensionality reduction as we have seen in previous chapters.

# In[10]:


sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata


# This adds several new items to our AnnData object. The `var` slot now includes means, dispersions and the selected variable genes. In the `obsp` slot we have distances and connectivities for our KNN graph and in `obsm` are the PCA and UMAP embeddings.
# 
# Let's plot the UMAP, colouring the points by cell identity and batch labels. If the dataset had not already been labelled (which is often the case) we would only be able to consider the batch labels.

# In[11]:


adata.uns[batch_key + "_colors"] = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
]  # Set custom colours for batches
sc.pl.umap(adata, color=[label_key, batch_key], wspace=1)


# Often when looking at these plots you will see a clear separation between batches. In this case, what we see is more subtle and while cells from the same label are generally near each other there is a shift between batches. If we were to perform a clustering analysis using this raw data we would probably end up with some clusters containing a single batch which would be difficult to interpret at the annotation stage. We are also likely to overlook rare cell types which are not common enough in any single sample to produce their own cluster. While UMAPs can often display batch effects, as always when considering these 2D representations it is important not to overinterpret them. For a real analysis, you should confirm the integration in other ways such as by inspecting the distribution of marker genes. In the “Benchmarking your own integration” section below we discuss metrics for quantifying the quality of an integration.
# 
# Now that we have confirmed there are batch effects to correct we can move on to the different integration methods. If the batches perfectly overlaid each other or we could discover meaningful cell clusters without correction then there would be no need to perform integration.

# ## Batch-aware feature selection

# As shown in {ref}`previous chapters <pre-processing:feature-selection>` we often select a subset of genes to use for our analysis in order to reduce noise and processing time. We do the same thing when we have multiple samples, however, it is important that gene selection is performed in a batch-aware way. This is because genes that are variable across the whole dataset could be capturing batch effects rather than the biological signals we are interested in. It also helps to select genes relevant to rare cell identities, for example, if an identity is only present in one sample then markers for it may not be variable across all the samples but should be in that one sample.
# 
# We can perform batch-aware highly variable gene selection by setting the `batch_key` argument in the **scanpy** `highly_variable_genes()` function. **scanpy** will then calculate HVGs for each batch separately and combine the results by selecting those genes that are highly variable in the highest number of batches. We use the **scanpy** function here because it has this batch awareness built in. For other methods, we would have to run them on each batch individually and then manually combine the results.

# In[12]:


sc.pp.highly_variable_genes(
    adata, n_top_genes=2000, flavor="cell_ranger", batch_key=batch_key
)
adata
adata.var


# We can see there are now some additional columns in `var`:
# 
# * `highly_variable_nbatches` - The number of batches where each gene was found to be highly variable
# * `highly_variable_intersection` - Whether each gene was highly variable in every batch
# * `highly_variable` - Whether each gene was selected as highly variable after combining the results from each batch
# 
# Let's check how many batches each gene was variable in:

# In[13]:


n_batches = adata.var["highly_variable_nbatches"].value_counts()
ax = n_batches.plot(kind="bar")
n_batches


# The first thing we notice is that most genes are not highly variable. This is typically the case but it can depend on how different the samples we are trying to integrate are. The overlap then decreases as we add more samples, with relatively few genes being highly variable in all three batches. By selecting the top 2000 genes we have selected all HVGs that are present in two or three batches and most of those that are present in one batch.
# 
# ```{admonition} How many genes to use?
# 
# This is a question that doesn't have a clear answer. The authors of the **scvi-tools** package which we use below recommend between 1000 and 10000 genes but how many depends on the context including the complexity of the dataset and the number of batches. A survey from a previous best practices paper {cite}`Luecken2019-og` indicated people typically use between 1000 and 6000 HVGs in a standard analysis. While selecting fewer genes can aid in the removal of batch effects {cite}`Luecken2021-jo` (the most highly-variable genes often describe only dominant biological variation), we recommend selecting slightly too many genes rather than selecting too few and risk removing genes that are important for a rare cell type or a pathway of interest. It should however be noted that more genes will also increase the time required to run the integration methods.
# ```
# 
# We will create an object with just the selected genes to use for integration.

# In[14]:


adata_hvg = adata[:, adata.var["highly_variable"]].copy()
adata_hvg


# ## Variational autoencoder (VAE) based integration

# The first integration method we will use is **scVI** (single-cell Variational Inference), a method based on a conditional variational autoencoder {cite}`Lopez2018-au` available in the **scvi-tools** package {cite}`Gayoso2022-ar`. A [variational autoencoder](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) is a type of artificial neural network which attempts to reduce the dimensionality of a dataset. The conditional part refers to conditioning this dimensionality reduction process on a particular covariate (in this case batches) such that the covariate does not affect the low-dimensional representation. In benchmarking studies **scVI** has been shown to perform well across a range of datasets with a good balance of batch correction while conserving biological variability {cite}`Luecken2021-jo`. **scVI** models raw counts directly, so it is important that we provide it with a count matrix rather than a normalized expression matrix.
# 
# First, let's make a copy of our dataset to use for this integration. Normally it is not necessary to do this but as we will demonstrate multiple integration methods making a copy makes it easier to show what has been added by each method.

# In[15]:


adata_scvi = adata_hvg.copy()


# ### Data preparation

# The first step in using **scVI** is to prepare our AnnData object. This step stores some information required by **scVI** such as which expression matrix to use and what the batch key is.

# In[16]:


scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key)
adata_scvi


# The fields created by **scVI** are prefixed with `_scvi`. These are designed for internal use and should not be manually modified. The general advice from the **scvi-tools** authors is to not make any changes to our object until after the model is trained. On other datasets, you may see a warning about the input expression matrix containing unnormalised count data. This usually means you should check that the layer provided to the setup function does actually contain count values but it can also happen if you have values from performing gene length correction on data from a full-length protocol or from another quantification method that does not produce integer counts.

# ### Building the model

# We can now construct an **scVI** model object. As well as the **scVI** model we use here, the **scvi-tools** package contains various other models (we will use the **scANVI** model below).

# In[17]:


model_scvi = scvi.model.SCVI(adata_scvi)
model_scvi


# The **scVI** model object contains the provided AnnData object as well as the neural network for the model itself. You can see that currently the model is not trained. If we wanted to modify the structure of the network we could provide additional arguments to the model construction function but here we just use the defaults.
# 
# We can also print a more detailed description of the model that shows us where things are stored in the associated AnnData object.

# In[18]:


model_scvi.view_anndata_setup()


# Here we can see exactly what information has been assigned by **scVI**, including details like how each different batch is encoded in the model.

# ### Training the model

# The model will be trained for a given number of _epochs_, a training iteration where every cell is passed through the network. By default **scVI** uses the following heuristic to set the number of epochs. For datasets with fewer than 20,000 cells, 400 epochs will be used and as the number of cells grows above 20,000 the number of epochs is continuously reduced. The reasoning behind this is that as the network sees more cells during each epoch it can learn the same amount of information as it would from more epochs with fewer cells.

# In[19]:


max_epochs_scvi = np.min([round((20000 / adata.n_obs) * 400), 400])
max_epochs_scvi


# We now train the model for the selected number of epochs (this will take ~20-40 minutes depending on the computer you are using).

# In[20]:


model_scvi.train()


# ```{admonition} Early stopping
# 
# Additionally to setting a target number of epochs, it is possible to also set `early_stopping=True` in the training function. This will let **scVI** decide to stop training early depending on the convergence of the model. The exact conditions for stopping can be controlled by other parameters.
# ```

# ### Extracting the embedding 

# The main result we want to extract from the trained model is the latent representation for each cell. This is a multi-dimensional embedding where the batch effects have been removed that can be used in a similar way to how we use PCA dimensions when analysing a single dataset. We store this in `obsm` with the key `X_scvi`.

# In[21]:


adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()


# ### Calculate a batch-corrected UMAP

# We will now visualise the data as we did before integration. We calculate a new UMAP embedding but instead of finding nearest neighbors in PCA space, we start with the corrected representation from **scVI**.

# In[22]:


sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
sc.tl.umap(adata_scvi)
adata_scvi


# Once we have the new UMAP representation we can plot it coloured by batch and identity labels as before.

# In[23]:


sc.pl.umap(adata_scvi, color=[label_key, batch_key], wspace=1)


# This looks better! Before, the various batches were shifted apart from each other. Now, the batches overlap more and we have a single blob for each cell identity label.
# 
# In many cases, we would not already have identity labels so from this stage we would continue with clustering, annotation and further analysis as described in other chapters.

# ## VAE integration using cell labels

# When performing integration with **scVI** we pretended that we didn't already have any cell labels (although we showed them in plots). While this scenario is common there are some cases where we do know something about cell identity in advance. Most often this is when we want to combine one or more publicly available datasets with data from a new study. When we have labels for at least some of the cells we can use **scANVI** (single-cell ANnotation using Variational Inference) {cite}`Xu2021-dh`. This is an extension of the **scVI** model that can incorporate cell identity label information as well as batch information. Because it has this extra information it can try to keep the differences between cell labels while removing batch effects. Benchmarking suggests that **scANVI** tends to better preserve biological signals compared to **scVI** but sometimes it is not as effective at removing batch effects {cite}`Luecken2021-jo`. While we have labels for all cells here it is also possible to use **scANVI** in a semi-supervised manner where labels are only provided for some cells.
# 
# ```{admonition} Label harmonization
# 
# If you are using **scANVI** to integrate multiple datasets for which you already have labels it is important to first perform _label harmonization_. This refers to a process of checking that labels are consistent across the datasets that are being integrated. For example, a cell may be annotated as a "T cell" in one dataset, but a cell of the same type could have been given the label "CD8+ T cell" in another dataset. How best to harmonize labels is an open question but often requires input from subject-matter experts.
# ```
# 
# We start by creating a **scANVI** model object. Note that because **scANVI** refines an already trained **scVI** model, we provide the scVI model rather than an AnnData object. If we had not already trained an **scVI** model we would need to do that first. We also provide a key for the column of `adata.obs` which contains our cell labels as well as the label which corresponds to unlabelled cells. In this case all of our cells are labelled so we just provide a dummy value. In most cases, it is important to check that this is set correctly so that **scANVI** knows which label to ignore during training.
# 

# In[24]:


# Normally we would need to run scVI first but we have already done that here
# model_scvi = scvi.model.SCVI(adata_scvi) etc.
model_scanvi = scvi.model.SCANVI.from_scvi_model(
    model_scvi, labels_key=label_key, unlabeled_category="unlabelled"
)
print(model_scanvi)
model_scanvi.view_anndata_setup()


# This **scANVI** model object is very similar to what we saw before for **scVI**. As mentioned previously, we could modify the structure of the model network but here we just use the default parameters.
# 
# Again, we have a heuristic for selecting the number of training epochs. Note that this is much fewer than before as we are just refining the **scVI** model, rather than training a whole network from scratch.

# In[25]:


max_epochs_scanvi = int(np.min([10, np.max([2, round(max_epochs_scvi / 3.0)])]))
model_scanvi.train(max_epochs=max_epochs_scanvi)


# We can extract the new latent representation from the model and create a new UMAP embedding as we did for **scVI**.

# In[26]:


adata_scanvi = adata_scvi.copy()
adata_scanvi.obsm["X_scANVI"] = model_scanvi.get_latent_representation()
sc.pp.neighbors(adata_scanvi, use_rep="X_scANVI")
sc.tl.umap(adata_scanvi)
sc.pl.umap(adata_scanvi, color=[label_key, batch_key], wspace=1)


# By looking at the UMAP representation it is difficult to tell the difference between **scANVI** and **scVI** but as we will see below there are differences in metric scores when the quality of the integrations is quantified. This is a reminder that we shouldn't overinterpret these two-dimensional representations, especially when it comes to comparing methods.

# ## Graph-based integration

# The next method we will look at is **BBKNN** or "Batch Balanced KNN" {cite}`Polanski2019-zy`. This is a very different approach to **scVI**, which rather than using a neural network to embed cells in a batch corrected space, instead modifies how the _k_-nearest neighbor (KNN) graph used for clustering and embedding is constructed. As we have seen in {ref}`previous chapters <cellular-structure:clustering>` the normal KNN procedure connects cells to the most similar cells across the whole dataset. The change that **BBKNN** makes is to enforce that cells are connected to cells from other batches. While this is a simple modification it can be quite effective, particularly when there are very strong batch effects. However, as the output is an integrated graph it can have limited downstream uses as few packages will accept this as an input.
# 
# An important parameter for **BBKNN** is the number of neighbors per batch. A suggested heuristic for this is to use 25 if there are more than 100,000 cells or the default of 3 if there are fewer than 100,000.

# In[27]:


neighbors_within_batch = 25 if adata_hvg.n_obs > 100000 else 3
neighbors_within_batch


# Before using **BBKNN** we first perform a PCA as we would before building a normal KNN graph. Unlike **scVI** which models raw counts here, we start with the log-normalised expression matrix.

# In[28]:


adata_bbknn = adata_hvg.copy()
adata_bbknn.X = adata_bbknn.layers["logcounts"].copy()
sc.pp.pca(adata_bbknn)


# We can now run **BBKNN**, replacing the call to the **scanpy** `neighbors()` function in a standard workflow. An important difference is to make sure the `batch_key` argument is set which specifies a column in `adata_hvg.obs` that contains batch labels.

# In[29]:


bbknn.bbknn(
    adata_bbknn, batch_key=batch_key, neighbors_within_batch=neighbors_within_batch
)
adata_bbknn


# Unlike the default **scanpy** function, **BBKNN** does not allow specifying a key for storing results so they are always stored under the default "neighbors" key.
# 
# We can use this new integrated graph just like we would a normal KNN graph to construct a UMAP embedding.
# 

# In[30]:


sc.tl.umap(adata_bbknn)
sc.pl.umap(adata_bbknn, color=[label_key, batch_key], wspace=1)


# This integration is also improved compared to the unintegrated data with cell identities grouped together but we sill see some shifts between batches.

# ## Linear embedding integration using Mutual Nearest Neighbors (MNN)

# Some downstream applications cannot accept an integrated embedding or neighborhood graph and require a corrected expression matrix. One approach that can produce this output is the integration method in **Seurat** {cite}`Satija2015-or,Butler2018-js,Stuart2019-lq`. The **Seurat** integration method belongs to a class of _linear embedding models_ that make use of the idea of _mutual nearest neighbors_ (which **Seurat** calls _anchors_) to correct batch effects {cite}`Haghverdi2018-bd`. Mutual nearest neighbors are pairs of cells from two different datasets which are in the neighborhood of each other when the datasets are placed in the same (latent) space. After finding these cells they can be used to align the two datasets and correct the differences between them. **Seurat** has also been found to be one of the top mixing methods in some evaluations {cite}`Tran2020-ia`.
# 
# As **Seurat** is an R package we must transfer our data from Python to R. Here we prepare the AnnData to convert so that it can be handled by **rpy2** and **anndata2ri**.

# In[31]:


adata_seurat = adata_hvg.copy()
# Convert categorical columns to strings
adata_seurat.obs[batch_key] = adata_seurat.obs[batch_key].astype(str)
adata_seurat.obs[label_key] = adata_seurat.obs[label_key].astype(str)
# Delete uns as this can contain arbitrary objects which are difficult to convert
del adata_seurat.uns
adata_seurat


# The prepared AnnData is now available in R as a SingleCellExperiment object thanks to **anndata2ri**. Note that this is transposed compared to an AnnData object so our observations (cells) are now the columns and our variables (genes) are now the rows.

# In[32]:


get_ipython().run_cell_magic('R', '-i adata_seurat', 'adata_seurat\n')


# **Seurat** uses its own object to store data. Helpfully the authors provide a function to convert from SingleCellExperiment. We just provide the SingleCellExperiment object and tell **Seurat** which assays (layers in our AnnData object) contain raw counts and normalised expression (which **Seurat** stores in a slot called "data").

# In[33]:


get_ipython().run_cell_magic('R', '-i adata_seurat', 'seurat <- as.Seurat(adata_seurat, counts = "counts", data = "logcounts")\nseurat\n')


# Unlike some of the other methods, we have seen which take a single object and a batch key, the **Seurat** integration functions require a list of objects. We create this using the `SplitObject()` function.

# In[34]:


get_ipython().run_cell_magic('R', '-i batch_key', 'batch_list <- SplitObject(seurat, split.by = batch_key)\nbatch_list\n')


# We can now use this list to find anchors for each pair of datasets. Usually, you would identify batch-aware highly variable genes first (using the `FindVariableFeatures()` and `SelectIntegrationFeatures()` functions) but as we have done that already we tell **Seurat** to use all the features in the object.

# In[35]:


get_ipython().run_cell_magic('R', '', 'anchors <- FindIntegrationAnchors(batch_list, anchor.features = rownames(seurat))\nanchors\n')


# **Seurat** can then use the anchors to compute a transformation that maps one dataset onto another. This is done in a pairwise way until all the datasets are merged. By default **Seurat** will determine a merge order so that more similar datasets are merged together first but it is also possible to define this order.

# In[36]:


get_ipython().run_cell_magic('R', '', 'integrated <- IntegrateData(anchors)\nintegrated\n')


# The result is another Seurat object, but notice now that the active assay is called "integrated". This contains the corrected expression matrix which is the final output of the integration.
# 
# Here we extract that matrix and prepare it for transfer back to Python.

# In[37]:


get_ipython().run_cell_magic('R', '-o integrated_expr', '# Extract the integrated expression matrix\nintegrated_expr <- GetAssayData(integrated)\n# Make sure the rows and columns are in the same order as the original object\nintegrated_expr <- integrated_expr[rownames(seurat), colnames(seurat)]\n# Transpose the matrix to AnnData format\nintegrated_expr <- t(integrated_expr)\nprint(integrated_expr[1:10, 1:10])\n')


# We will now store the corrected expression matrix as a layer in our AnnData object. We also set `adata.X` to use this matrix.

# In[38]:


adata_seurat.X = integrated_expr
adata_seurat.layers["seurat"] = integrated_expr
print(adata_seurat)
adata.X


# Now that we have the results of our integration we can calculate a UMAP and plot it as we have for the other methods (we could also have done this in R).

# In[39]:


# Reset the batch colours because we deleted them earlier
adata_seurat.uns[batch_key + "_colors"] = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
]
sc.tl.pca(adata_seurat)
sc.pp.neighbors(adata_seurat)
sc.tl.umap(adata_seurat)
sc.pl.umap(adata_seurat, color=[label_key, batch_key], wspace=1)


# As we have previously seen, the batches are mixed while the labels are separated. It is tempting to select an integration based on the UMAPs but this does not fully represent the quality of an integration. In the next section, we present some approaches to more rigorously evaluate integration methods.
# 
# ```{admonition} A note on scalability
# 
# As you ran the different integration methods you may have noticed that **scVI** took the most time. While this is true for small datasets like the example shown here, [benchmarks have shown](https://www.nature.com/articles/s41592-021-01336-8/figures/13) that **scVI** scales well for larger datasets. This is largely because the number of training epochs is adjusted for larger dataset sizes. MNN methods typically don't scale as well, partly because they perform several pairwise integrations, so if you have 20 batches you are performing 20 integrations while other methods can consider all batches at once.
# ```

# ## Benchmarking your own integration

# The methods demonstrated here are selected based on results from benchmarking experiments including the [single-cell integration benchmarking project](https://theislab.github.io/scib-reproducibility/) {cite}`Luecken2021-jo`. This project also produced a software package called [**scib**](https://www.github.com/theislab/scib) that can be used to run a range of integration methods as well as the metrics that were used for evaluation. In this section, we show how to use this package to evaluate the quality of an integration.
# 
# ```{admonition} What is the ground truth?
# 
# Some of these metrics, particularly those that evaluate the conservation of biological variation, require a known ground truth to compare to. Usually, this is a cell identity label but can sometimes be other information such as known trajectories. Because of this requirement, it is difficult to evaluate integration for a completely new dataset where it is unclear what biological signal should be preserved.
# ```
# 
# The **scib** metrics can be run individually but there are also wrappers for running multiple metrics at once. Here we run a subset of the metrics which are quick to compute using the `metrics_fast()` function. This function takes a few arguments: the original unintegrated dataset, the integrated dataset, a batch key and a label key. Depending on the output of the integration method we might also need to supply additional arguments, for example here we specify the embedding to use for **scVI** and **scANVI** with the `embed` argument. You can also control how some metrics are run with additional arguments. Also note that you may need to check that objects are formatted properly so that **scIB** can find the required information.
# 
# Let's run the metrics for each of the integrations we have performed above, as well as the unintegrated data (after highly variable gene selection).

# In[40]:


metrics_scvi = scib.metrics.metrics_fast(
    adata, adata_scvi, batch_key, label_key, embed="X_scVI"
)
metrics_scanvi = scib.metrics.metrics_fast(
    adata, adata_scanvi, batch_key, label_key, embed="X_scANVI"
)
metrics_bbknn = scib.metrics.metrics_fast(adata, adata_bbknn, batch_key, label_key)
metrics_seurat = scib.metrics.metrics_fast(adata, adata_seurat, batch_key, label_key)
metrics_hvg = scib.metrics.metrics_fast(adata, adata_hvg, batch_key, label_key)


# Here is an example of what one of the metrics results looks like for a single integration:

# In[41]:


metrics_hvg


# Each row is a different metric and the values show the score for that metric. Scores are between 0 and 1, where 1 is a good performance and 0 is a poor performance (**scib** can also return unscaled scores for some metrics if required). Because we have only run the fast metrics here, some of the metrics have `NaN` scores. Also, note that some metrics cannot be used with some output formats which can also be a reason for `NaN` values being returned.
# 
# To compare the methods it is useful to have all the metrics results in one table. This code combines them and tidies them into a more convenient format.

# In[42]:


# Concatenate metrics results
metrics = pd.concat(
    [metrics_scvi, metrics_scanvi, metrics_bbknn, metrics_seurat, metrics_hvg],
    axis="columns",
)
# Set methods as column names
metrics = metrics.set_axis(
    ["scVI", "scANVI", "BBKNN", "Seurat", "Unintegrated"], axis="columns"
)
# Select only the fast metrics
metrics = metrics.loc[
    [
        "ASW_label",
        "ASW_label/batch",
        "PCR_batch",
        "isolated_label_silhouette",
        "graph_conn",
        "hvg_overlap",
    ],
    :,
]
# Transpose so that metrics are columns and methods are rows
metrics = metrics.T
# Remove the HVG overlap metric because it's not relevant to embedding outputs
metrics = metrics.drop(columns=["hvg_overlap"])
metrics


# We now have all the scores in one table with metrics as columns and methods as rows. Styling the table with a gradient can make it easier to see the differences between scores.

# In[43]:


metrics.style.background_gradient(cmap="Blues")


# For some metrics, the scores tend to be in a relatively small range. To emphasise the differences between methods and place each metric on the same scale, we scale them so that the worst performer gets a score of 0, the best performer gets a score of 1 and the others are somewhere in between.

# In[44]:


metrics_scaled = (metrics - metrics.min()) / (metrics.max() - metrics.min())
metrics_scaled.style.background_gradient(cmap="Blues")


# The values now better represent the differences between methods (and better match the colour scale). However, it is important to note that the scaled scores can only be used to compare the relative performance of this specific set of integrations. If we wanted to add another method we would need to perform the scaling again. We also can't say that an integration is definitively "good", only that it is better than the other methods we have tried. This scaling emphasises differences between methods. For example, if we had metric scores of 0.92, 0.94 and 0.96 these would be scaled to 0, 0.5 and 1.0. This makes the first method appear to score much worse, even though it is only slightly lower than the other two and still got a very high score. This effect is bigger when comparing a few methods and when they get similar raw scores. Whether you look at raw or scaled scores depends on whether you want to focus on absolute performance or the difference in performance between methods.
# 
# The evaluation metrics can be grouped into two categories, those that measure the removal of batch effects and those that measure the conservation of biological variation. We can calculate summary scores for each of these categories by taking the mean of the scaled values for each group. This kind of summary score wouldn't make sense with raw values as some metrics consistently produce higher scores than others (and therefore have a greater effect on the mean).

# In[45]:


metrics_scaled["Batch"] = metrics_scaled[
    ["ASW_label/batch", "PCR_batch", "graph_conn"]
].mean(axis=1)
metrics_scaled["Bio"] = metrics_scaled[["ASW_label", "isolated_label_silhouette"]].mean(
    axis=1
)
metrics_scaled.style.background_gradient(cmap="Blues")


# Plotting the two summary scores against each other gives an indication of the priorities of each method. Some will be biased towards batch correction while others will favour retaining biological variation.

# In[46]:


fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
metrics_scaled.plot.scatter(
    x="Batch",
    y="Bio",
    c=range(len(metrics_scaled)),
    ax=ax,
)

for k, v in metrics_scaled[["Batch", "Bio"]].iterrows():
    ax.annotate(
        k,
        v,
        xytext=(6, -3),
        textcoords="offset points",
        family="sans-serif",
        fontsize=12,
    )


# In our small example scenario **BBKNN** is clearly the worst performer, getting the lowest scores for both batch removal and biological conservation. The other three methods have similar batch correction scores with **scANVI** scoring highest for biological conservation followed by **scVI** and **Seurat**.
# 
# To get an overall score for each method we can combine the two summary scores. The **scIB** paper suggests a weighting of 40% batch correction and 60% biological conservation but you may prefer to weight things differently depending on the priorities for your dataset.

# In[47]:


metrics_scaled["Overall"] = 0.4 * metrics_scaled["Batch"] + 0.6 * metrics_scaled["Bio"]
metrics_scaled.style.background_gradient(cmap="Blues")


# Let's make a quick bar chart to visualise the overall performance.

# In[48]:


metrics_scaled.plot.bar(y="Overall")


# As we have already seen **scVI** and **scANVI** are the best performers with **scANVI** scoring slightly higher. It is important to note that this is just an example of how to run these metrics for this specific dataset, not a proper evaluation of these methods. For that, you should refer to existing benchmarking publications. In particular, we have only run a small selection of high-performing methods and a subset of metrics. Also, remember that scores are relative to the methods used so even if the methods perform almost equally well and small differences will be exaggerated.
# 
# Existing benchmarks have suggested methods that generally perform well, but performance can also be quite variable across scenarios. For some analyses, it may be worthwhile performing your own evaluation of integration. The **scib** package makes this process easier, but it can still be a significant undertaking, relying on a good knowledge of the ground truth and interpretation of the metrics.

# ## Key Takeaways

# 1. Visualize your data before attempting to correct for batch effects to assess the extent of the issue. Batch effect correction is not always required and it might mask the biological variation of interest.
# 2. If cell labels are available and biological variation is the most important, the usage of methods that can use these labels (such as scANVI) is advised.
# 3. Consider running several integration methods on your dataset and evaluating them with the **scIB** metrics to use the integration that is most robust for your use case.

# ## Quiz

# 1. What are the sources of batch effects?
# 2. What is the difference between technical and biological variation?
# 3. How does one evaluate whether the integration worked well or not? What are useful metrics for this purpose?

# ## Session information

# ### Python

# In[49]:


import session_info

session_info.show()


# ### R

# In[50]:


get_ipython().run_cell_magic('R', '', 'sessioninfo::session_info()\n')


# ## References

# ```{bibliography}
# :filter: docname in docnames
# :labelprefix: int
# ```

# ## Contributors
# 
# We gratefully acknowledge the contributions of:
# 
# ### Authors
# 
# * Luke Zappia
# * Malte Lücken
# 
# ### Reviewers
# 
# * Lukas Heumos

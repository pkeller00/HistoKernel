# HistoKernel: Maximum Mean Discrepancy Kernels for Whole Slide Images

### Piotr Keller*, Muhammad Dawood and Fayyaz ul Amir Afsar Minhas
### Tissue Image Analytics Center, University of Warwick, United Kingdom

This repository contains the code for the following manuscript:

HistoKernel: Maximum Mean Discrepancy Kernels for Whole Slide Images, submitted to Nature Machine Intelligence for review.

## Introduction
In Computational Pathology (CPath) the use of multi-gigapixel images for various clinical tasks is common. However, due to the size of these images current methods are forced to make patch-level predictions which are then aggregated into slide-level predictions. This work proposes a novel solution to the aggregation problem. By utilizing Maximum Mean Discrepancy (MMD) to measure similarity between Whole Slide Images (WSIs) we generate a slide-level similarity kernel that common kernel based methods can leverage. We perform a comprehensive analysis of this novel approach in CPath. We use this method, with WSIs as input, to perform point mutation classification (n = 3419), drug sensitivity prediction (n = 551), survival analysis (n = 2291) and WSI retrieval (n = 9362) beating existing baselines. We also propose a novel perturbation based method to provide patch-level explainability of our model. This work opens up avenues for further exploration of kernel methods to perform slide-level tasks in CPath.

## Dependencies

## Usage
### Step 1. Data download
Download the FFPE whole slide images from GDC portal (https://portal.gdc.cancer.gov/).

Download corresponding gene point mutation and Disease Specific Survival from cBioPortal (https://www.cbioportal.org/).

Download drug sensitivity scores for breast cancer patients (https://github.com/engrodawood/HiDS).
### Step 2. Data processing
For each WSI perform:

- Tile extraction: extract 1024x1024 tiles from the large WSI at a spatial resolution of 0.50 microns-per-pixel
- Patches capturing less that 40% of informative tissue are discarded (mean pixel intensity above 200)
- Feature extraction: extract a feature vector for each tile using [`RetCCL`](https://github.com/Xiyue-Wang/RetCCL)

Details can be found in the paper.
### Step 3. MMD Kernel generation for 2048-dimensional feature representations of 9,324 TCGA slides 

Using the code under [`MMD_distance_matrix_generator`](https://github.com/pkeller00/Anubis/tree/main/MMD_distance_matrix_generator) to generate an $N \times N$ distance matrix using MMD where $N$ is the number of WSIs in a dataset.

Details can be found in the paper and [MMD_distance_matrix_generator](https://github.com/pkeller00/Anubis/tree/main/MMD_distance_matrix_generator).

### Step 4. Downstream Analysis
To perfrom the downstream tasks (point mutation prediction, [`Drug Sensitivty prediction`](https://github.com/pkeller00/Anubis/tree/main/DrugSensitivity), [`Survival Analysis`](https://github.com/pkeller00/Anubis/tree/main/SurvivalAnalysis) and [`WSI Retrival`](https://github.com/pkeller00/Anubis/tree/main/WSIRetrival) ) mentioned in the paper  navigate to the appropraite folder in this GitHub.

## Note

Some intermediate data are put into the folder [`data`](https://github.com/pkeller00/Anubis/tree/main/data).

--------

\* first author

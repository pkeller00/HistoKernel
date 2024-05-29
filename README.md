# Anubis: Maximum Mean Discrepancy Kernels for Whole Slide Images

### Piotr Keller*, Muhammad Dawood and Fayyaz ul Amir Afsar Minhas
### Tissue Image Analytics Center, University of Warwick, United Kingdom

This repository contains the code for the following manuscript:

Anubis: Maximum Mean Discrepancy Kernels for Whole Slide Images, submitted to Nature Machine Intelligence for review.

## Introduction

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

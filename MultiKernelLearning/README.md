# Multi Kernel Learning

## Introduction
For this task, we are interested in integrating multiple data modalities into a single kernel. In this task we want to combine information from different data sources such as morphological and genetic in order to capture complementary information about target variables. This, in theory, should achieve better and more robust predictions in contrast to using single-modal data. As a proof of concept we demonstrate preliminary results for multi-modal survival analysis in breast cancer however this framework is applicable to other clinical tasks and cancers.

## Usage
To perform this the distance matrix and its corresponding slide IDs [(from Google Drive)](https://drive.google.com/drive/folders/1gT7UDz9vjz9eHOgil-8ICfLvBKWw3GUr) are required. You can also use your own distance matrix by using `MMD_distance_matrix_generator`.

To perform Multi Kernel Learning for our MMD-based approach:
1) Run `MultiKernelLearning/.py` to generate the [topic based gene expression kernel](https://github.com/engrodawood/HiGGsXplore).
2) Run `MultiKernelLearning/.py` to perform multi-modal kernel analysis.

# WSI Retrieval

## Introduction
For this task, we are interested in returning the top $k$ most similar WSIs for a given query image, $X_q$. Here for a query WSI, $X_q$, given that we know the site of origin of $X_q$ we aim to retrieve the top $k$ most 'similar' WSIs. In this paper, similarity is defined as images from the same cancer sub-type. For example, if $X_q$ originates from the brain and is of sub-type brain lower grade glioma (LGG) a database of WSIs originating from the brain is searched and the algorithm is considered successful if it returns images of the same sub-type (in this case LGG).

## Usage
To perform this the distance matrix and its corresponding slide IDs located at `data/MMD_matrix/D_1052_blur_10.npy` and `data/MMD_matrix/slide_IDs_1052.npy` respectively are used. You can also use your distance matrix by using `MMD_distance_matrix_generator`.

To perform WSI Retrieval for the RetCCL comparison paper and our MMD-based approach:
1) Run `WSIRetrival/generate_mosaic.py` to generate the mosaics used by the comparison paper.
2) Run `WSIRetrival/parallel_search.py` to retrieve the top $k=5$ most similar slides using the comparison search method.
3) Run `WSIRetrival/macro_average.py` to calculate the majority vote at the top k search results ($mMv@k$) for each site by each method as well as the total macro average across all sites.

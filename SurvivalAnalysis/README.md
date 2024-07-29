# Survival Analysis

## Introduction
For this task, we are interested in using WSIs to predict the expected time until a clinically important event occurs, for example, disease progression or death, given other patients survival data.

## Usage
To perform this the distance matrix and its corresponding slide IDs [(from Google Drive)](https://drive.google.com/drive/folders/1gT7UDz9vjz9eHOgil-8ICfLvBKWw3GUr) are required. You can also use your own distance matrix by using `MMD_distance_matrix_generator`.

To perform Survival analysis for GNN comparison paper and our MMD-based approach:
1) Run `SurvivalAnalysis/extract_graphs.py` to generate the graphs used by the GNN approach.
2) Run `SurvivalAnalysis/gnn.py` to perform survival analysis for the 6 cancer types considered using the GNN comparison method.
3) Run `SurvivalAnalysis/mmd.py` to perform survival analysis for the 6 cancer types considered using our MMD-based method.

# Drug Sensitivity

## Introduction
For this task, we are interested in the prediction of patients’ sensitivity to multiple drugs from routine H\&E images.

## Usage
To perform this the distance matrix of the entire TCGA and its corresponding slide IDs from [Google drive](https://drive.google.com/drive/folders/1gT7UDz9vjz9eHOgil-8ICfLvBKWw3GUr). You can also use your own distance matrix by using `MMD_distance_matrix_generator`. In the code we then filter this matrix to only contain relavant BRCA samples inline with the baseline study.

To perform Dug Sensitivity prediction for our MMD-based approach Run `DrugSensitivity/predict_sensitivity.py` to first filter the relvant samples from BRCA and then train a Support Vector Regressor for each compound separately. The results of 5-fold cross validation are stored as a list in `DrugSensitivity/results.pickle`. Each entry of results corresponds to one of the folds such that a given entry is a $D\times 2$ matrix $R^j$ where D is the number of drugs used in the study ($D=427$) and $j$ is the fold $j \in 1..5$. THe two columns of the matrix, for a given drug at row $i$, correspond to the spearman rank coeffeicent ($R_{i,0}$) and its associated p-value ($R_{i,1}$) .

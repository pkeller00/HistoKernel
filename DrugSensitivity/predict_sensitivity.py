import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pickle

FOLDS = 5
SLIDE_IDS_PATH = '../data/MMDKernel/IDS.npy'
D_PATH = '../data/MMDKernel/D_full.npy'
DRUG_SENSITIVITY_PATH = '../data/DrugSensitivity/BRCA_Drug_sensitivity.csv'
GOOD_SLIDES_PATH = '../data/DrugSensitivity/slide_selection_final.txt'
OUTPUT_PATH = './results/results.pickle'

slides_mmd = np.load(SLIDE_IDS_PATH)
slides_mmd = [os.path.basename(i).split('.')[0] for i in slides_mmd]
slides_mmd = np.array(slides_mmd)
D_mmd = np.load(D_PATH)

DS = pd.read_csv(DRUG_SENSITIVITY_PATH)
DS.set_index('Patient ID', inplace=True)

# Reading slides with all drug sensitivites to filter by
slide_selection_file = open(GOOD_SLIDES_PATH, "r") 
data = slide_selection_file.read() 
good_slides = data.split("\n") 
slide_selection_file.close() 


filtered_mmd_idx = []
filtered_tags = []
for idx,TAG in enumerate(slides_mmd):
    # If Drug sensitivity data is missing
    if TAG not in good_slides:
        continue
    TAG = ('-').join(TAG.split('-')[:3])
    if TAG not in DS.index:
        continue
    filtered_mmd_idx.append(idx)
    filtered_tags.append(TAG)

filtered_mmd_idx = np.array(filtered_mmd_idx)
D_mmd = D_mmd[:,filtered_mmd_idx]
D_mmd = D_mmd[filtered_mmd_idx,:]
slides_mmd = slides_mmd[filtered_mmd_idx]

filtered_tags = [('-').join(i.split('-')[:3]) for i in filtered_tags]
filtered_tags = np.array(filtered_tags)

# Converting sensitivity data into z-score0
DS.loc[:,DS.columns] = StandardScaler().fit_transform(DS)

skf = KFold(n_splits=FOLDS, shuffle=False)

n_classes = DS.shape[1]

# Results for 5-fold CV
results = []
for _ in range(FOLDS):
    results.append(np.zeros((n_classes, 2))-2)

# For each drug do 5-fold CV seperately
for index in tqdm(range(n_classes)):
    labels = []

    for TAG in filtered_tags:
        tstatus = DS.loc[TAG, :].tolist()[index] 
        labels.append(tstatus)
    labels = np.array(labels)

    fold_idx = 0
    for trvi, test in skf.split(np.arange(0, D_mmd.shape[0])):
        train, valid = train_test_split(trvi, test_size=0.20, shuffle=True)
        

        K = np.exp(-1/np.median(D_mmd)*D_mmd)
        K_tr = K[train][:,train]
        K_tt = K[valid][:,train]

        y_tr = labels[train]
        y_tt = labels[valid]
        
        best_C = None
        best_gamma = None
        best_spearmanr = -np.inf

        for c in [0.0001, 0.001, 0.1, 1, 10, 100, 1000,10000]:
            for gamma in [4.0, 8.0, 16.0, 32.0, 64.0,128.0,256.0,512.0,1024.0]:
            # for gamma in ['scale']:
                svr = SVR(kernel='precomputed',gamma=gamma, C=c, epsilon=0.2)
                svr.fit(K_tr, y_tr) 
                output = svr.predict(K_tt)                
                score = stats.spearmanr(y_tt, output)[0]
                if score > best_spearmanr:
                    best_C = c
                    best_gamma = gamma
                    best_spearmanr = score
        
        K_tt = K[test][:,train]
        y_tt = labels[test]
        svr = SVR(kernel='precomputed',gamma=best_gamma, C=best_C, epsilon=0.2)
        svr.fit(K_tr, y_tr) 
        output = svr.predict(K_tt)

        results[fold_idx][index] = np.array(
                [stats.spearmanr(y_tt, output)[0], stats.spearmanr(y_tt, output)[1]])
        fold_idx += 1

import pickle
with open(OUTPUT_PATH, "wb") as fp:   #Pickling
    pickle.dump(results, fp)

# Sanity check to make sure file was saved correctly
with open(OUTPUT_PATH, "rb") as fp:   # Unpickling
    b = pickle.load(fp)
    print(b)
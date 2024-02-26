#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
from glob import glob
import os
import pickle
from statistics import mode, mean
from tqdm import tqdm
import numpy as np

BASE_PATH = '/home/u1904706/cloud_workspace/testingRetrival'
OUTPUT_DIR = '/home/u1904706/cloud_workspace/githubs/Anubis/WSIRetrival/results'
D_MMD_PATH = '/home/u1904706/cloud_workspace/dawood_survival/D_full.npy'
IDS_MMD_PATH = '/home/u1904706/cloud_workspace/dawood_survival/IDS.npy'
META_DATA_PATH = '/home/u1904706/cloud_workspace/githubs/Anubis/WSIRetrival/metadata.csv'
SITE_DICT = {"Pulmonary":["LUAD","LUSC","MESO"],
             "Urinary": ["BLCA","KIRC","KICH","KIRP"],
             "Gastrointestinal" : ["COAD","ESCA","READ","STAD"],
             "Melanocytic" : ["UVM","SKCM"],
             "Brain":["GBM","LGG"],
             "Liver": ["CHOL","LIHC","PAAD"],
             "Gynecologic": ["UCEC","CESC","UCS","OV"],
             "Endocrine": ["ACC","PCPG","THCA"],
             "Hematopoiesis": ["DLBC","THYM"],
             "Prostate":["TGCT","PRAD"],
             }

metadata = pd.read_csv(META_DATA_PATH)
metadata = metadata.set_index('file_name')

retccl_results = []
mmd_results = []

def filter_MMD_matrix(metadata,IDs,D):
    filt = []
    for num,id in enumerate(IDs):
        try:
            value = metadata.loc[id,'primary_site']
        except:
            continue
        if value == site:
            filt.append(num)
    filt = np.array(filt)
    D = D[filt,:]
    D = D[:,filt]
    IDs = IDs[filt]
    K = np.exp(-1/np.median(D)*D)#formulate kernel
    return K,IDs

def subtype_result_RetCCL(retrival,metadata,subtype):
    total = 0
    correct = 0
    patients = set()
    for image in retrival:
        with open(image, 'rb') as file:
            ret = pickle.load(file)

        ground_truth = metadata.loc[os.path.basename(image[:-11]+'.npy'),'project_name']

        if ground_truth != subtype:
            continue

        ret = sorted(ret, key=lambda x: x[2], reverse=True)[0:5]
        predcited_labels = [metadata.loc[os.path.basename(x[0]),'project_name'] for x in ret]
        predcited_labels_mode = mode(predcited_labels)
        if ground_truth == predcited_labels_mode:
            correct += 1
        total += 1
        patients.add(os.path.basename(image)[0:12])
    return total,len(list(patients)),correct/total

def subtype_result_MMD(retrival,metadata,IDs,K):
    total = 0
    correct = 0
    patients = set()
    for image in retrival:
        file_name = os.path.basename(image[:-11]+'.npy')
        ground_truth = metadata.loc[file_name,'project_name']

        if ground_truth != subtype:
            continue
        idx = np.where(IDs == np.array(file_name))[0][0]
        patient = file_name[0:12]
        row = K[idx, :] 

        D_sort = np.argsort(-row)

        count = 0
        preds = []
        start_id = 1 #Since index 0 will always be the slide itself (as only a slide with itself can have a distance of 0)
        
        # Ignore slides with same Patient ID (prevent information leakage)
        while count < 5:
            res_patient = IDs[D_sort[start_id]][0:12]
            if res_patient == patient:
                start_id += 1
                continue
            preds.append(IDs[D_sort[start_id]])
            start_id += 1
            count += 1
        predcited_labels = [metadata.loc[x,'project_name'] for x in preds]
        predcited_labels_mode = mode(predcited_labels)
        if ground_truth == predcited_labels_mode:
            correct += 1
        total += 1
        patients.add(patient)
    return total,len(list(patients)),correct/total

# Go through each site

retccl_results = []
mmd_results = []
for site in tqdm(SITE_DICT.keys()):
    site_df = []
    SITE_PATH = os.path.join(BASE_PATH,'results_search',site,'temp')
    retrival = glob(os.path.join(SITE_PATH,'*WSIRet.pkl'))

    D = np.load(D_MMD_PATH)
    IDs = np.load(IDS_MMD_PATH)
    IDs = np.array([os.path.basename(x) for x in IDs])

    K,IDs = filter_MMD_matrix(metadata,IDs,D)

    # Go through each subtype
    for subtype in SITE_DICT[site]:
        slide_num,patient_num, mmv5_retccl = subtype_result_RetCCL(retrival,metadata,subtype)
        _,_, mmv5_mmd = subtype_result_MMD(retrival,metadata,IDs,K)
        site_df.append([subtype,slide_num,patient_num,mmv5_retccl,mmv5_mmd])

        retccl_results.append(mmv5_retccl)
        mmd_results.append(mmv5_mmd)

    site_df = pd.DataFrame(site_df,columns=['WSI Type', '#WSI' ,'#Patient' ,'𝑚𝑀𝑉 @5 RetCCL', '𝑚𝑀𝑉 @5 MMD'])
    site_df.to_csv(f"{OUTPUT_DIR}/{site}.csv",index=False)

    
print(f'RetCCL Macro: {np.mean(retccl_results)}')
print(f'MMD Macro: {np.mean(mmd_results)}')
                
            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch; print(torch.__version__)
import torch; print(torch.version.cuda)

import numpy as np
import os
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree as sKDTree
from tqdm import tqdm
import pickle
import pandas as pd
import os
from tqdm import tqdm

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def toTensor(v,dtype = torch.float,requires_grad = True): 
    device = 'cpu'   
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)

def connectClusters(Cc,dthresh = 3000):
    tess = Delaunay(Cc)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    nx = neighbors    
    W = np.zeros((Cc.shape[0],Cc.shape[0]))
    for n in nx:
        nx[n] = np.array(list(nx[n]),dtype = int)
        nx[n] = nx[n][KDTree(Cc[nx[n],:]).query_ball_point(Cc[n],r = dthresh)]
        W[n,nx[n]] = 1.0
        W[nx[n],n] = 1.0        
    return W # neighbors of each cluster and an affinity matrix

def toGeometric(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

def build_graph(d,locs_centroids,label=1):
    x, y, F = locs_centroids[:,0],locs_centroids[:,1], d
    #import pdb; pdb.set_trace()
    C = np.asarray(np.vstack((x, y)).T, dtype=int)
    W = connectClusters(C, dthresh=4000) # dthresh: threshold value for connecting patches
    G = toGeometric(F, W, y=label)
    G.coords = toTensor(C, requires_grad=False)
    cpu = torch.device('cpu')
    G.to(cpu)
    with open(ofile, 'wb') as f:
          pickle.dump(G, f)


FEATURES_DIR = f'../data/Features/RetSSL-FEATS'
GRAPHS_DIR_BASE = f'./graphs'
SURVIVAL_DATA_PATH = '../data/SurvivalAnalysis/NIHMS978596-supplement-1.xlsx'
UPDATED_BRAIN_CLASSES_PATH = '../data/SurvivalAnalysis/ijms-2057006_Table S2.xlsx'
#Filter wsis by type
import pandas as pd

allCancers = ['KIRC','UCEC','BLCA','LUAD','Astrocytoma']
for cancer in allCancers:

    GRAPHS_DIR = f'{GRAPHS_DIR_BASE}/{cancer}'  

    # path to where to dump the Graphs
    mkdirs(GRAPHS_DIR)

    
    if cancer == 'Astrocytoma':
        # To filter the correct subset of brain cancers
        new_classification = pd.read_excel(UPDATED_BRAIN_CLASSES_PATH)
        new_classification = new_classification[new_classification['WHO_CNS5_diagnosis'].str.contains(cancer)]
        new_classification = new_classification['Case_ID'].tolist()

        # TCGA has the wrong LGG vs GBM split so we have to load both of their 'GBM' and 'LGG' labels
        cancer = ['GBM','LGG']

    else:
        cancer = [cancer]
        new_classification = None

    survival_data = pd.read_excel(SURVIVAL_DATA_PATH)
    survival_data = survival_data[survival_data["type"].isin(cancer)]
    patient_ids = survival_data["bcr_patient_barcode"].tolist()

    ex_list = [] # add problamatic files
    import glob
    from pathlib import Path
    for wsi_path in tqdm(glob.glob(f'{FEATURES_DIR}/*_feat.npy')):
        wsi_name = Path(wsi_path).stem

        ofile = f'{GRAPHS_DIR}/{wsi_name}.pkl'
        if os.path.isfile(ofile):
            continue
        if wsi_name[0:12] not in patient_ids:
            continue
        if new_classification is not None:
            if wsi_name[0:12] not in new_classification:
                continue 

        d = np.load(wsi_path,allow_pickle=True)
        
        locs_path = f'{FEATURES_DIR}/{wsi_name[:-5]}_pos.npy'
        locs = np.load(locs_path,allow_pickle=True)
        locs_temp = np.copy(locs)
        # (tl_x,tl_y,br_x,br_y) ---> (br_x,br_y,tl_x,tl_y)
        locs_temp[:, [0,1,2,3]] = locs[:, [2,3,0,1]]
        # we want (locs + locs_temp) / 2 to find centroid
        sum_locs = np.add(locs,locs_temp)/2
        locs_centroids = sum_locs[:,[0,1]]

        try:
            build_graph(d,locs_centroids)
            print('Finished:', wsi_name)  
        except Exception as e: 
            print(e)
            ex_list.append(wsi_name)
            np.savez(f'{GRAPHS_DIR}/execp.npz',ex_list)

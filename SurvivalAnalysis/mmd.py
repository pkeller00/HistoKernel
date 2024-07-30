import os
os.environ['CUDA_PATH'] = '/home/u1904706/cloud_workspace/condaEnvs/MMDKernels_v2'

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, Sampler
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from copy import deepcopy
from numpy.random import randn
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
from torch_geometric.nn import GINConv,EdgeConv, PNAConv,DynamicEdgeConv,global_add_pool, global_mean_pool, global_max_pool
import time
from tqdm import tqdm
from scipy.spatial import distance_matrix, Delaunay
import random
from torch_geometric.data import Data, DataLoader
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold
import pdb
from statistics import mean, stdev
from glob import glob
import os
import pandas as pd
import numpy as np
import pickle
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
from random import shuffle
from itertools import islice
from lifelines.utils import concordance_index as cindex
from lifelines import KaplanMeierFitter
from sklearn.model_selection import StratifiedShuffleSplit
from collections import OrderedDict
import re
import shutil
from lifelines.statistics import logrank_test
from sksurv.svm import FastKernelSurvivalSVM
from lifelines.utils import survival_table_from_events


#In the orginal implentaion of the baseline study there were two formats for the graphs (e..g label names etc)
# e depending on the features used
#In our paper we only used the same format as their baseliners shufflenet features with our RetCCL features thus
# this should always be kept to true
SHUFFLE_NET = True

VARIABLES = 'DSS'
TIME_VAR = VARIABLES + '.time'
USE_CUDA = torch.cuda.is_available()
rng = np.random.default_rng()
device = {True:'cuda:0',False:'cpu'}[USE_CUDA]
FOLDS = 5

SLIDES_PATH = r'../data/MMDKernel/IDS.npy'
D_PATH = r'../data/MMDKernel/D_full.npy'
OUTPUT_PATH = r"../data/SurvivalAnalysis/results/mmd_results.csv"
SURVIVAL_PATH = r'../data/SurvivalAnalysis/NIHMS978596-supplement-1.xlsx'

def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = True):
    return torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad)

def toTensorGPU(v,dtype = torch.float,requires_grad = True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))

def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def toGeometric(Gb,y,tt=1e-3):
    return Data(x=Gb.x, edge_index=(Gb.get(W)>tt).nonzero().t().contiguous(),y=y)

def toGeometricWW(X,W,y,tt=0):
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))


def SplitBrcaData(dataset, numSplits, isShuffle, testSize):
    if isShuffle:
        eventVars = [dataset[i][1][0] for i in range(len(dataset))]
    else:
        eventVars = [int(dataset[i].event.detach().numpy()) for i in range(len(dataset))]
    x = np.zeros(len(dataset))

    shuffleSplit = StratifiedShuffleSplit(n_splits = numSplits, test_size = testSize,random_state=49)
    return shuffleSplit.split(x,eventVars)


def save_full_D(upper_D):
    """ Fills in the lower part of a matrix that only has the upper part computed.

        Parameters
        ----------
        upper_D : N * N array
            Kernel matrix that only has upper portion computed
        filename : stry
            the filename of the kernel to be saved
        savePath : str
            Where to save the kernel
    """
    D_cpy = upper_D.copy()
    D_cpy = D_cpy + D_cpy.T - np.diag(np.diag(D_cpy))
    return D_cpy
  
if __name__ == '__main__':
    idx = np.load(SLIDES_PATH,mmap_mode='r') #slide IDs
    D = np.load(D_PATH, mmap_mode='r') #MMD distance matrix
    idx = [os.path.basename(x) for x in idx]
    basenames = np.array([x[:-4] for x in idx])
    patients = np.array([i[0:12] for i in idx])

    K = np.exp(-1/np.median(D)*D)#formulate kernel
    GAMMA = 1/np.median(D)
 
    results_df = []

    allCancers = ["Astrocytoma","Glioblastoma","KIRC","UCEC","BLCA","LUAD"]
    for CANCER in allCancers:
            
        device = {True:'cuda:0',False:'cpu'}[USE_CUDA]
        import pandas as pd
        import os
        from natsort import natsorted
        # This is set up to run on colab vvv
        survival_file = SURVIVAL_PATH
        cols2read = [VARIABLES,TIME_VAR]
        TS = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID')  # path to clinical file
        # TS = TS[cols2read][TS.type == CANCER]

        if CANCER == 'Astrocytoma' or CANCER == 'Glioblastoma':
            TS = TS[cols2read][TS.type.isin(["GBM","LGG"])]
        else:
            TS = TS[cols2read][TS.type == CANCER]

        if SHUFFLE_NET:
            bdir = r'./graphs/'+CANCER+'/'
            
            # Set up directory for on disk dataset
            directory = r'./graph_surv/'+CANCER+'/'
            
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
        Exid = 'Slide_Graph CC_feats'
        from glob import glob
        graphlist = glob(os.path.join(bdir, "*.pkl"))

        device = 'cuda:0'
        cpu = torch.device('cpu')

        graphlist = natsorted(graphlist)
        dataset = []
        from tqdm import tqdm
        patients_done = set()
        for graph in tqdm(graphlist):
            TAG = os.path.split(graph)[-1].split('_')[0][:12]
            if TAG in patients_done:
                continue
            patients_done.add(TAG)
            
            try:
                status = TS.loc[TAG,:][1]
            except:
                continue
            event, event_time = TS.loc[TAG,:][0], TS.loc[TAG,:][1]
            if np.isnan(event):
                continue
            if SHUFFLE_NET:
                G = pickleLoad(graph)
                G.to('cpu')
            else:
                if USE_CUDA:
                    G = pickleLoad(graph)
                    G.to('cpu')
                else:
                    G = torch.load(graph, map_location=device)
            try:
                G.y = toTensorGPU([int(status)], dtype=torch.long, requires_grad = False)
            except ValueError:
                continue
            W = radius_neighbors_graph(toNumpy(G.coords), 1500, mode="connectivity",include_self=False).toarray()
            g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
            g.coords = G.coords
            g.event = toTensor(event)
            g.e_time = toTensor(event_time)
            if SHUFFLE_NET:
                dataset.append([TAG,(event,event_time)])
            else:
                dataset.append(g)
        
        trainingDataset = dataset
        event_vector = np.array([int(g[1][0]) for g in trainingDataset])

        from tqdm import tqdm
        
        print(trainingDataset)
        skf = SplitBrcaData(trainingDataset,FOLDS,SHUFFLE_NET,0.2)
        splits = []
        total = []
        for train_index,test_index in skf:
            total = len(train_index) + len(test_index)
            splits.append([train_index,test_index])
        
        C = []
        P = []
        run_counter = 0
        for data in splits:
            train_index, vali_index = data[0], data[1]
            
            x_train = [trainingDataset[i] for i in train_index]
            testDataset = [trainingDataset[i] for i in vali_index]                
            
            kernel_train_index = []
            Y_train = []
            T_train = []
            E_train = []

            for item in x_train:
                if item[1][1] <= 0:
                    continue
                slides = np.where(patients == item[0])[0]
                if slides.size == 0:
                    continue
                Y_train.append(item[1])

                kernel_train_index.append(slides[0])
                T_train.append(item[1][1])
                E_train.append(item[1][0])

            kernel_test_index = []
            Y_test = []
            T_test = []
            E_test = []
            
            for item in testDataset:
                slides = np.where(patients == item[0])[0]
                if slides.size == 0:
                    continue
                Y_test.append(item[1])
                kernel_test_index.append(slides[0])
                T_test.append(item[1][1])
                E_test.append(item[1][0])

            total_slides = len(kernel_train_index) + len(kernel_test_index)
            kernel_test_index = np.array(kernel_test_index)
            kernel_train_index = np.array(kernel_train_index)

            Y_test = np.array(Y_test,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            Y_train = np.array(Y_train,dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
                
            T_test = np.array(T_test)
            E_test = np.array(E_test)

            T_train = np.array(T_train)
            E_train = np.array(E_train)

            K_train = K[kernel_train_index][:,kernel_train_index]

            kssvm = FastKernelSurvivalSVM(kernel="precomputed", alpha = 0.125, max_iter  = 2000)
            kssvm = kssvm.fit(K_train,Y_train)

            Z = kssvm.predict(K[:,kernel_train_index])
            Z_train,Z_test = Z[kernel_train_index], Z[kernel_test_index]

            Tmax = 10*365
            tidx = T_train>Tmax
            T_train[tidx]=Tmax
            E_train[tidx]=False
            tidx = T_test>Tmax
            T_test[tidx]=Tmax
            E_test[tidx]=False
            
            c_ttx = cindex(T_test, -Z_test, E_test)
            C.append(c_ttx)

            thr = np.median(Z_train)

            results = logrank_test(T_test[Z_test>thr], T_test[Z_test<=thr], E_test[Z_test>thr],E_test[Z_test<=thr])
            p_tt = results.p_value
            P.append(p_tt)
            print(c_ttx)

        #   Plot Kaplan Meier curve
            fig = plt.figure()
            ax = plt.gca() 
            try:
                from lifelines.plotting import add_at_risk_counts
                test_low = KaplanMeierFitter().fit(T_test[Z_test<=thr], event_observed=E_test[Z_test<=thr],label="Low")#NelsonAalenFitter()
                test_high = KaplanMeierFitter().fit(T_test[Z_test>thr], event_observed=E_test[Z_test>thr],label="High")#NelsonAalenFitter()

                table_low = survival_table_from_events(T_test[Z_test<=thr],E_test[Z_test<=thr])
                table_high = survival_table_from_events(T_test[Z_test>thr],E_test[Z_test>thr])
                
                ax = test_high.plot_survival_function(ax=ax, show_censors = False)
                ax = test_low.plot_survival_function(ax=ax, show_censors = False)
                
                add_at_risk_counts(test_low,test_high,ax=ax)
                
                tt = ax.get_xticklabels()

                plt.title('logrank p-value: %0.3f c-index: %0.3f' %(p_tt,c_ttx))
                    
                plt.ylim(0.5, 1)
                ax.get_legend().remove()
                curves_dir = f'./kmcurves/{CANCER}'
                if not os.path.exists(curves_dir):
                    os.mkdir(curves_dir)
                plt.savefig(f'{curves_dir}/{run_counter}.svg',dpi=600,bbox_inches='tight')
                plt.savefig(f'{curves_dir}/{run_counter}.png',dpi=600,bbox_inches='tight')
                table_low.to_csv(f'{curves_dir}/{run_counter}_low.svg')    
                table_high.to_csv(f'{curves_dir}/{run_counter}_high.svg')                           
    
            except Exception as e:
                print("Figure doesn't work")
                print(e)
            plt.close(fig)
            run_counter += 1
        print(np.mean(C),np.std(C),2*np.median(P))
        results_df.append([CANCER,np.mean(C),np.std(C),2*np.median(P),total_slides])
        pd.DataFrame(results_df,columns=["Cancer","C","std","p-value","total_slides"]).to_csv(OUTPUT_PATH,index=False)
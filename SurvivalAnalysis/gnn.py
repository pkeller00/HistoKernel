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


LEARNING_RATE = 0.00002
WEIGHT_DECAY = 0.005
L1_WEIGHT = 0.001
SCHEDULER = None
BATCH_SIZE = 10
NUM_BATCHES = 2000
NUM_LOGS = 150 # How many times in training the loss value is stored

#In the orginal implentaion of the baseline study there were two formats for the graphs (e..g label names etc)
#depending on the features used
#In our paper we only used the same format as their baseliners shufflenet features with our RetCCL features thus
# this should always be kept to true
SHUFFLE_NET = True

VALIDATION = True
NORMALIZE = False
CENSORING = True
FRAC_TRAIN = 0.8
CONCORD_TRACK = True
FILTER_TRIPLE = False
EARLY_STOPPING = True
VARIABLES = 'DSS'
TIME_VAR = VARIABLES + '.time'
ON_GPU = True
USE_CUDA = torch.cuda.is_available()
rng = np.random.default_rng()
device = {True:'cuda:0',False:'cpu'}[USE_CUDA]

OUTPUT_PATH = "/home/u1904706/cloud_workspace/githubs/Anubis/SurvivalAnalysis/results/gnn_results.csv"
MODEL_PATH = '/home/u1904706/cloud_workspace/dawood_survival/Best_model/'
SURVIVAL_PATH = r'/home/u1904706/cloud_workspace/dawood_survival/NIHMS978596-supplement-1.xlsx'

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


def pair_find(graphs,features):
    indexes = []
    for j in range(len(graphs)):
        graph_j = graphs[j]
        if features == 'BRCA-SHUFFLE':
            event_j = graph_j[1][0]
            time_j = graph_j[1][1]
        else:
            event_j, time_j = graph_j.event, graph_j.e_time
        if event_j == 1:
            for i in range(len(graphs)):
                graph_i = graphs[i]
                if features == 'BRCA-SHUFFLE':
                    time_i = graph_i[1][1]
                else:
                    time_i = graph_i.e_time
                if graph_j != graph_i and time_i > time_j:
                    indexes.append((i,j))
    shuffle(indexes)
    return indexes

def SplitBrcaData(dataset, numSplits, isShuffle, testSize):
    if isShuffle:
        eventVars = [dataset[i][1][0] for i in range(len(dataset))]
    else:
        eventVars = [int(dataset[i].event.detach().numpy()) for i in range(len(dataset))]
    x = np.zeros(len(dataset))
    shuffleSplit = StratifiedShuffleSplit(n_splits = numSplits, test_size = testSize,random_state=49)
    return shuffleSplit.split(x,eventVars)

def disk_graph_load(batch):
    return [torch.load(directory + '/' + graph + '.g') for graph in batch]

def get_predictions(model,graphs,features = 'BRCA-CC',device=torch.device('cuda:0')) -> list:
    outputs = []
    e_and_t = []
    model.eval()
    with torch.no_grad():
        for i in range(len(graphs)):
            graph = graphs[i]
            if features == 'BRCA-SHUFFLE':
                tag = [graph[0]]
                temp = [graph[1][0], graph[1][1]]
                graph = disk_graph_load(tag)
            else:
                temp = [graph.event.item(),graph.e_time.item()]
                graph = [graph]
            size = 1
            loader = DataLoader(graph, batch_size=size)
            for d in loader:
                d = d.to(device)
            z,_,_ = model(d)
            z = toNumpy(z)
            outputs.append(z[0][0])
            e_and_t.append(temp)
    return outputs, e_and_t


class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[16,16,8],pooling='max',dropout = 0.0,conv='GINConv',gembed=False,**kwargs) -> None:
        """
        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.
        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Returns
        -------
        None.
        """
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        self.gembed = gembed #if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim),GELU())
                self.linears.append(Sequential(Linear(out_emb_dim, dim_target),GELU()))

            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))
                subnet = Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim))
                if conv=='GINConv':
                    self.nns.append(subnet)
                    self.convs.append(GINConv(self.nns[-1], **kwargs))  # Eq. 4.2 eps=100, train_eps=False
                elif conv=='EdgeConv':
                    subnet = Sequential(Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim))
                    self.nns.append(subnet)
                    self.convs.append(EdgeConv(self.nns[-1],**kwargs))#DynamicEdgeConv#EdgeConv                aggr='mean'
                else:
                    raise NotImplementedError

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data) -> torch.tensor:

        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        pooling = self.pooling
        Z = 0
        import torch.nn.functional as F
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z+=z
                dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z+=z
                    dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout

        return out,Z,x
    
class NetWrapper:
    def __init__(self, model, device='cuda:0',features='BRCA-CC') -> None:
        self.model = model
        self.device = torch.device(device)
        self.features = features

    def loss_fn(self,batch,optimizer) -> float:
        z = toTensorGPU(0)
        loss = 0
        unzipped = [j for pair in batch for j in pair]
        # This can be changed when using a system with large RAM
        if self.features == 'BRCA-SHUFFLE':
            graph_set = list(set(unzipped))
            graphs = disk_graph_load(graph_set)
            unzipped = None
        else:
            graph_set = unzipped
            graphs = graph_set
            unzipped = None
        batch_load = DataLoader(graphs, batch_size = len(graphs))
        for d in batch_load:
            d = d.to(self.device)
        self.model.train()
        optimizer.zero_grad()
        output,_,_ = self.model(d)
        num_pairs = len(batch)
        for (xi,xj) in batch:
            graph_i, graph_j = graph_set.index(xi), graph_set.index(xj)
            # Compute loss function
            dz = output[graph_i] - output[graph_j]
            loss += torch.max(z, 1.0 - dz)
        loss = loss/num_pairs
        loss.backward()
        optimizer.step()
        return loss.item()

    def validation_loss_and_Cindex_eval(self,graphs,pairs) -> float:
        tot_loss = 0
        print('Number of Validation Pairs: ' + str(len(pairs)))
        predictions, e_and_t = get_predictions(self.model,graphs,self.features)
        for j in range(len(pairs)):
            p_graph_i = predictions[pairs[j][0]]
            p_graph_j = predictions[pairs[j][1]]
            dz = p_graph_i - p_graph_j
            loss = max(0, 1.0 - dz)
            tot_loss += loss
        epoch_val_loss = tot_loss / len(pairs)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        concord = cindex(T,predictions,E)
        return epoch_val_loss, concord

    def censor_data(self,graphs, censor_time): # The censor time measured in years
        cen_time = 365 * censor_time
        for graph in graphs:
            if self.features == 'BRCA-SHUFFLE':
                time = graph[1][1]
            else:
                time = graph.e_time
            if time > cen_time:
                if self.features == 'BRCA-SHUFFLE':
                    graph[1] = (0,cen_time)
                else:
                    graph.event = toTensor(0)
                    graph.e_time = toTensor(cen_time)
            else:
                continue
        return graphs

    def train(self,training_data,validation_data,max_batches=500,num_logs=50,optimizer=torch.optim.Adam,
              early_stopping = 10, return_best = False, batch_size = 10) -> float:
            return_best = return_best and validation_data is not None
            counter = 0 # To resolve list index errors with large NUM_BATCHES vals
            log_interval = max_batches // num_logs
            loss_vals = {}
            loss_vals['train'] = []
            loss_vals['validation'] = []
            concords = []
            c_best = 0.5
            best_batch = 1000
            patience = early_stopping
            training_indexes = pair_find(training_data,self.features)
            print("Number of batches used for training "+ str(max_batches))
            print('Num Pairs: ' + str(len(training_indexes)))
            best_model = deepcopy(self.model)
            for i in tqdm(range(1,max_batches + 1)):
                if counter < len(training_indexes) - batch_size:
                    batch_pairs = []
                    index_pairs = training_indexes[counter:counter+batch_size]
                    for j in range(len(index_pairs)):
                        if self.features == 'BRCA-SHUFFLE':
                            graph_i = training_data[index_pairs[j][0]][0]
                            graph_j = training_data[index_pairs[j][1]][0]
                        else:
                            graph_i = training_data[index_pairs[j][0]]
                            graph_j = training_data[index_pairs[j][1]]
                        batch_pairs.append((graph_i,graph_j))
                    loss = self.loss_fn(batch_pairs,optimizer)
                    counter += batch_size
                else:
                    counter = 0
                loss_vals['train'].append(loss)
                if i % log_interval == 0:
                    if validation_data is not None:
                        val_loss, c_val = self.validation_loss_and_Cindex_eval(validation_data,validation_indexes)
                        loss_vals['validation'].append(val_loss)
                        concords.append(c_val)
                        print("Current Vali Loss Val: " + str(val_loss) + "\n")
                        print("\n" + "Current Loss Val: " + str(loss) + "\n")
                        if return_best and c_val > c_best:
                            c_best = c_val
                            #best_model = deepcopy(model)
                            best_batch = i
                        if i - best_batch > patience*log_interval:
                            print("Early Stopping")
                            #break
            return loss_vals, concords, self.model
class Evaluator:
    def __init__(self, model, device='cuda:0',features = 'BRCA-CC') -> None:
        self.model = model
        self.device = device
        self.features = features

    def get_predictions(self,model,graphs,device=torch.device('cuda:0')) -> list:
        outputs = []
        e_and_t = []
        model.eval()
        with torch.no_grad():
            for i in range(len(graphs)):
                graph = graphs[i]
                if self.features == 'BRCA-SHUFFLE':
                    tag = [graph[0]]
                    temp = [graph[1], graph[1]]
                    graph = disk_graph_load(tag)
                else:
                    temp = [graph.event.item(),graph.e_time.item()]
                    graph = [graph]
                size = 1
                loader = DataLoader(graph, batch_size=size)
                for d in loader:
                    d = d.to(device)
                z,_,_ = model(d)
                z = toNumpy(z)
                outputs.append(z[0])
                e_and_t.append(temp)
        return outputs, e_and_t

    def test_evaluation(self,testDataset):
        predictions, e_and_t = get_predictions(self.model,testDataset,self.features)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        concord = cindex(T,predictions,E)
        return concord

    def selectThreshold(self,dataset,minprop = 0.1):
        from lifelines.statistics import logrank_test
        
        predictions, e_and_t = get_predictions(self.model,dataset,self.features)
        TT = np.array([x[1] for x in e_and_t])
        EE = np.array([x[0] for x in e_and_t])
        Z = np.array(predictions)
        
        thr_list = np.percentile(Z,np.linspace(minprop,1-minprop,int((1-minprop)*100))*100)
        V = [logrank_test(TT[Z>thr], TT[Z<=thr], EE[Z>thr],EE[Z<=thr]).test_statistic for thr in thr_list]
        thr = thr_list[np.argmax(V)]
        return thr

    def K_M_Curves(self, graphs, split_val, mode = 'Train') -> None:
        outputs, e_and_t = get_predictions(self.model,graphs,self.features)
        T = [x[1] for x in e_and_t]
        E = [x[0] for x in e_and_t]
        mid = np.median(outputs)
        if mode != 'Train':
            if split_val > 0:
                mid = split_val
        else:
            print(mid)
        T_high = []
        T_low = []
        E_high = []
        E_low = []
        for i in range(len(outputs)):
          if outputs[i] <= mid:
            T_high.append(T[i])
            E_high.append(E[i])
          else:
            T_low.append(T[i])
            E_low.append(E[i])
        # km_high = KaplanMeierFitter()
        # km_low = KaplanMeierFitter()
        # ax = plt.subplot(111)
        # ax = km_high.fit(T_high, event_observed=E_high, label = 'High').plot_survival_function(ax=ax)
        # ax = km_low.fit(T_low, event_observed=E_low, label = 'Low').plot_survival_function(ax=ax)
        # from lifelines.plotting import add_at_risk_counts
        # add_at_risk_counts(km_high, km_low, ax=ax)
        # plt.title('Kaplan-Meier estimate')
        # plt.ylabel('Survival probability')
        # plt.show()
        # plt.tight_layout()
        from lifelines.statistics import logrank_test
        results = logrank_test(T_low, T_high, E_low, E_high)
        print("p-value %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))
        return results.p_value
        
        
if __name__ == '__main__':

    results_df = []
    allCancers = ["Astrocytoma","Glioblastoma","KIRC","UCEC","BLCA","LUAD"]
    allCancers = ["KIRC","UCEC","BLCA","LUAD"]
    for CANCER in allCancers:
            
        device = {True:'cuda:0',False:'cpu'}[USE_CUDA]
        import pandas as pd
        import os
        from natsort import natsorted
        # This is set up to run on colab vvv
        
        cols2read = [VARIABLES,TIME_VAR]
        TS = pd.read_excel(SURVIVAL_PATH).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID')  # path to clinical file
        
        if CANCER == 'Astrocytoma' or CANCER == 'Glioblastoma':
            TS = TS[cols2read][TS.type.isin(["GBM","LGG"])]
        else:
            TS = TS[cols2read][TS.type == CANCER]
        
        if SHUFFLE_NET:
            bdir = r'/home/u1904706/cloud_workspace/dawood_survival/graphs/'+CANCER+'/'
            # Set up directory for on disk dataset
            directory = r'/home/u1904706/cloud_workspace/dawood_survival/graph_surv/'+CANCER+'/'
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
        Exid = 'Slide_Graph CC_feats'
        from glob import glob
        graphlist = glob(os.path.join(bdir, "*.pkl"))#[0:100]
        print(len(graphlist))
        device = 'cuda:0'
        cpu = torch.device('cpu')

        try:
            os.mkdir(MODEL_PATH)
        except FileExistsError:
            pass
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
                torch.save(g,directory+'/'+TAG+'.g')
            else:
                dataset.append(g)
        
        trainingDataset = dataset
        event_vector = np.array([int(g[1][0]) for g in trainingDataset])

        folds = 5

        from tqdm import tqdm
        
        skf = SplitBrcaData(trainingDataset,folds,SHUFFLE_NET,0.2)
        splits = []
        total = []
        for train_index,test_index in skf:
            total = len(train_index) + len(test_index)
            splits.append([train_index,test_index])

        import glob
        files = glob.glob(f'{directory}/*.g')
        GRAPH_NAME = files[0]
        if SHUFFLE_NET:
            # G = dataset[0]
            G = torch.load(GRAPH_NAME)
        else:
            G = dataset[0]

        converg_vals = []
        fold_concord = []
        eval_metrics = []
        pvalues = []
        for data in splits:
            train_index, vali_index = data[0], data[1]
            # Set up model and optimizer
            model = GNN(dim_features=G.x.shape[1], dim_target = 1, layers = [16,16,8,8],
                        dropout = 0.0, pooling = 'mean', conv='EdgeConv', aggr = 'max')
            net = NetWrapper(model,device = device,features = 'BRCA-SHUFFLE')
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(),
                                    lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
            x_train = [trainingDataset[i] for i in train_index]
            testDataset = [trainingDataset[i] for i in vali_index]
            print(set([x[0] for x in x_train]).intersection(set([x[0] for x in testDataset])))

            # Only censoring the test data
            # x_val = net.censor_data(x_val,10)
            losses, concords, BestModel = net.train(x_train,
                                                    None,
                                                    optimizer = optimizer,
                                                    return_best = True,
                                                    max_batches = NUM_BATCHES)
            # Evaluate
            testDataset = net.censor_data(testDataset,10)
            eval = Evaluator(BestModel,features='BRCA-SHUFFLE')
            concord = eval.test_evaluation(testDataset)
            
            x_train = net.censor_data(x_train,10)
            thr = eval.selectThreshold(x_train)
            print(f'Threshold:{thr}')
            pvalue = eval.K_M_Curves(testDataset, thr, mode = 'Test')
            print(f'C:{concord} p:{pvalue}')
            eval_metrics.append(concord)
            converg_vals.append(losses)
            fold_concord.append(concords)
            pvalues.append(pvalue)
            #m = max(concords)

        avg_c = mean(eval_metrics)
        stdev_c = stdev(eval_metrics)
        print("Performance on test data over %d folds: \n" % folds)
        print(str(avg_c)+' +/- '+str(stdev_c))
        print(f"perf on each split was: {eval_metrics}")
        
        results_df.append([CANCER,avg_c,stdev_c,2*np.median(pvalues),total])
        pd.DataFrame(results_df,columns=["Cancer","C","std","p-value","Samples"]).to_csv(OUTPUT_PATH,index=False)

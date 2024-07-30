from hyperopt import fmin, tpe
from hyperopt import hp,STATUS_OK
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from lifelines.utils import concordance_index as cindex
from lifelines import KaplanMeierFitter
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.statistics import logrank_test
from sksurv.svm import FastKernelSurvivalSVM

def SplitBrcaData(dataset, numSplits, isShuffle, testSize):
    if isShuffle:
        eventVars = [dataset[i][1][0] for i in range(len(dataset))]
    else:
        eventVars = [int(dataset[i].event.detach().numpy()) for i in range(len(dataset))]
    x = np.zeros(len(dataset))

    shuffleSplit = StratifiedShuffleSplit(n_splits = numSplits, test_size = testSize)
    return shuffleSplit.split(x,eventVars)
    
def train_models(K_1,K_2,Y_train,T_train,E_train,kernel_train_index):
    # Make a validation set for param choice
    eventVars = [x[0] for x in Y_train]
    train_idx, val_idx, _, _ = train_test_split(
        range(len(kernel_train_index)), range(len(kernel_train_index)), 
        test_size=0.2, stratify=eventVars)  
    
    train_index = kernel_train_index[train_idx]
    valid_index = kernel_train_index[val_idx]
    
    YTr = Y_train[train_idx]
    YVal = Y_train[val_idx]
    
    TTr = T_train[train_idx]
    TVal = T_train[val_idx]
    
    ETr = E_train[train_idx]
    EVal = E_train[val_idx]
    
    Tmax = 10*365
    tidx = TTr>Tmax
    TTr[tidx]=Tmax
    ETr[tidx]=False
    tidx = TVal>Tmax
    TVal[tidx]=Tmax
    EVal[tidx]=False
            
    def objective_K_1(params):
        alpha = params['alpha']
        K_train = K_1[train_index][:,train_index]
        kssvm = FastKernelSurvivalSVM(kernel="precomputed",
                alpha = alpha,
                max_iter  = 2000)
        kssvm = kssvm.fit(K_train,YTr)
        Z = kssvm.predict(K_1[:,train_index])
        _,Z_val = Z[train_index], Z[valid_index]
        
        ci = cindex(TVal, -Z_val, EVal)
        return {'loss': -ci + (alpha),'status': STATUS_OK}
    
    def objective_K_2(params):
        alpha = params['alpha']
        K_train = K_2[train_index][:,train_index]
        kssvm = FastKernelSurvivalSVM(kernel="precomputed",
                alpha = alpha,
                max_iter  = 2000)
        kssvm = kssvm.fit(K_train,YTr)
        Z = kssvm.predict(K_2[:,train_index])
        _,Z_val = Z[train_index], Z[valid_index]
        
        ci = cindex(TVal, -Z_val, EVal)
        return {'loss': -ci + (alpha),'status': STATUS_OK}
    
    def objective_k_1_plus_k2(params):
        alpha = params['alpha']
        K = (K_1) + (K_2)
        
        K_train = K[train_index][:,train_index]
        kssvm = FastKernelSurvivalSVM(kernel="precomputed",
                alpha = alpha,
                max_iter  = 2000)
        kssvm = kssvm.fit(K_train,YTr)
        Z = kssvm.predict(K[:,train_index])
        _,Z_val = Z[train_index], Z[valid_index]
        
        ci = cindex(TVal, -Z_val, EVal)
        return {'loss': -ci + (alpha),'status': STATUS_OK}
    
    def objective_k_1_times_k2(params):
        alpha = params['alpha']
        K = (K_1) * (K_2)
        
        K_train = K[train_index][:,train_index]
        kssvm = FastKernelSurvivalSVM(kernel="precomputed",
                alpha = alpha,
                max_iter  = 2000)
        kssvm = kssvm.fit(K_train,YTr)
        Z = kssvm.predict(K[:,train_index])
        _,Z_val = Z[train_index], Z[valid_index]
        
        ci = cindex(TVal, -Z_val, EVal)
        return {'loss': -ci + (alpha),'status': STATUS_OK}
    
    # Only WSI kernel
    space_single = {
        'alpha': hp.uniform('alpha', 2.0 ** -12,0.125),
    }
    
    results = {}
    best_k1 = fmin(objective_K_1, space_single, algo=tpe.suggest, max_evals=50)
    results['K_topic'] = [best_k1['alpha'],1,0]
    
    best_k2 = fmin(objective_K_2, space_single, algo=tpe.suggest, max_evals=50)
    results['K_wsi'] = [best_k2['alpha'],0,1]
    
    best_k1_k2_equal = fmin(objective_k_1_times_k2, space_single, algo=tpe.suggest, max_evals=50)
    results['K_topic+K_wsi'] = [best_k1_k2_equal['alpha'],1,1]
    
    best_k1_k2_equal_mut = fmin(objective_k_1_times_k2, space_single, algo=tpe.suggest, max_evals=50)
    results['K_topic*K_wsi'] = [best_k1_k2_equal_mut['alpha'],1,1]
    
    return results

def test_single_model(K,
                      train_index,
                      test_index,
                      Y_train,
                      T_train,
                      E_train,
                      T_test,
                      E_test,
                      best_alpha):
    K_train = K[train_index][:,train_index]
    kssvm = FastKernelSurvivalSVM(kernel="precomputed", alpha = best_alpha, max_iter  = 2000)
    kssvm = kssvm.fit(K_train,Y_train)

    Z = kssvm.predict(K[:,train_index])
    Z_train,Z_test = Z[train_index], Z[test_index]
    
    Tmax = 10*365
    tidx = T_train>Tmax
    T_train[tidx]=Tmax
    E_train[tidx]=False
    tidx = T_test>Tmax
    T_test[tidx]=Tmax
    E_test[tidx]=False
    
    c_ttx = cindex(T_test, -Z_test, E_test)

    thr = np.median(Z_train)

    results = logrank_test(T_test[Z_test>thr], T_test[Z_test<=thr], E_test[Z_test>thr],E_test[Z_test<=thr])
    p_tt = results.p_value  
    
    return c_ttx,p_tt

def preprocess_result(K1,K2,results,key):
    result = results[key]
    
    alpha = result[0]
    k1_weight = result[1]
    k2_weight = result[2]
    
    K = k1_weight*K1 + k2_weight*K2
    
    return K,alpha
        
    
if __name__ == '__main__':
    results_df = []
    
    SHUFFLE_NET = True

    VARIABLES = 'DSS'
    TIME_VAR = VARIABLES + '.time'
    rng = np.random.default_rng()
    FOLDS = 3

    OUTPUT_PATH = f"./results/kernel_combo.csv"
    SURVIVAL_PATH = r'../data/SurvivalAnalysis/NIHMS978596-supplement-1.xlsx'
    TIME_THRESHOLD = 90


    
    # WSI kernel
    idx = np.load( f'../data/MultiModalKernels/IDS.npy',mmap_mode='r') #slide IDs
    D_wsi = np.load(f'../data/MultiModalKernels/D_wsi.npy', mmap_mode='r') #MMD distance matrix
    idx = [os.path.basename(x) for x in idx]
    patients = np.array([i[0:12] for i in idx])
    
    # Topic kernel
    K_wsi = np.exp(-1/np.median(D_wsi)*D_wsi)#formulate kernel
    K_topic = np.load(f'../data/MultiModalKernels/K_topics.npy', mmap_mode='r')
    
    CANCER = "BRCA"
    survival_file = SURVIVAL_PATH
    cols2read = [VARIABLES,TIME_VAR]
    TS = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID')  # path to clinical file
    TS = TS[cols2read][TS.type == CANCER]
    print(TS)
    
    graphlist = idx
    
    dataset = []
    patients_done = set()
    for graph in tqdm(graphlist):
        
        # Check for duplicate patients
        TAG = graph[0:12]
        if TAG in patients_done:
            continue
        
        # Skip if missing status
        try:
            status = TS.loc[TAG,:][1]
        except Exception as e:
            continue
        
        event, event_time = TS.loc[TAG,:][0], TS.loc[TAG,:][1]
            
        if np.isnan(event):
            continue
        
        try:
            int(status)
        except ValueError:
            continue
        
        if float(event_time) <= TIME_THRESHOLD:
            continue
        
        patients_done.add(TAG) #add tag that passed all tests
        
        # No missing label and not a duplicate so add to dataset
        dataset.append([TAG,(event,event_time)])
        
    trainingDataset = dataset
    skf = SplitBrcaData(trainingDataset,FOLDS,SHUFFLE_NET,0.2)
    
    splits = []
    total = []
    for train_index,test_index in skf:
        total = len(train_index) + len(test_index)
        splits.append([train_index,test_index])
        
    C = {}
    P = {}
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
            if item[1][1] <= 0:
                continue
            
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
        
        results = train_models(K_topic,K_wsi,Y_train,T_train,E_train,kernel_train_index)
        
        for idx,key in enumerate(results.keys()):
            if key not in C:
                C[key] = []
            if key not in P:
                P[key] = []
                
            K,best_alpha = preprocess_result(K_topic,K_wsi,results,key)
            c_ttx, pval = test_single_model(
                K=K,
                train_index=kernel_train_index,
                test_index=kernel_test_index,
                Y_train=Y_train,
                T_train=T_train,
                E_train=E_train,
                T_test=T_test,
                E_test=E_test,
                best_alpha=best_alpha
            )
            C[key].append(c_ttx)
            P[key].append(pval)
            print(f'fold {run_counter} {key} : C-index:{c_ttx} pval: {pval}')
        run_counter += 1
    for key in C.keys():   
        print(key,np.mean(C[key]),np.std(C[key]),2*np.median(P[key]))
        results_df.append([CANCER,key,np.mean(C[key]),np.std(C[key]),2*np.median(P[key]),total_slides])
    pd.DataFrame(results_df,columns=["Cancer","Kernel Combo", "C","std","2* median(p-value)","total_slides"]).to_csv(OUTPUT_PATH,index=False)
            
    
    
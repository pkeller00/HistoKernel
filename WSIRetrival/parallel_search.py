from os import makedirs
from os.path import join, basename, splitext, abspath
from statistics import mode, mean
import argparse
import glob
import pickle
import yaml
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm


sites_dict = {
    "Brain": "brain",
    "Breast": "breast",
    "Bronchus and lung": "lung",
    "Colon": "colon",
    "Liver and intrahepatic bile ducts": "liver",
}

sites_diagnoses_dict = {
    "Pulmonary":["LUAD","LUSC","MESO"],
            "Urinary": ["BLCA","KIRC","KICH","KIRP"],
    "Gastrointestinal" : ["COAD","ESCA","READ","STAD"],
             "Melanocytic" : ["UVM","SKCM"],
             "Brain":["GBM","LGG"],
             "Liver": ["CHOL","LIHC","PAAD"],
             "Gynecologic": ["UCEC","CESC","UCS","OV"],
             "Endocrine": ["ACC","PCPG","THCA"],
             "Hematopoiesis": ["DLBC","THYM"],
             "Prostate":["TGCT","PRAD"]
}



def cosine_sim(a, b):
    return np.dot(a, b)/(norm(a) * norm(b))



def calculate_weights(site,metadata):
    if site == "organ":
        factor = 30
        # Count the number of slide in each diagnosis (organ)
        latent_all = join(PATCHES_DIR, "*", "*", "patches", "*")
        type_of_organ = list(sites_diagnoses_dict.keys())
        total_slide = {k: 0 for k in type_of_organ}
        for latent_path in glob.glob(latent_all):
            anatomic_site = latent_path.split("/")[-4]
            total_slide[anatomic_site] += 1
    else:
        factor = 10
        # Count the number of slide in each site (organ)
        # latent_all = join(PATCHES_DIR, site, "*", "patches", "*")
        type_of_diagnosis = sites_diagnoses_dict[site]
        total_slide = {k: 0 for k in type_of_diagnosis}
        for diagnosis in total_slide.keys():
            count = metadata[metadata['project_name']==diagnosis].shape[0]
            total_slide[diagnosis] = count
        # for latent_path in glob.glob(latent_all):
        #     diagnosis = latent_path.split("/")[-3]
        #     total_slide[diagnosis] += 1
    
    # Using the inverse count as a weight for each diagnosis
    sum_inv = 0
    for v in total_slide.values():
        sum_inv += (1./v)

    # Set a parameter k  to make the weight sum to k (k = 10, here)
    norm_fact = factor / sum_inv
    weight = {k: norm_fact * 1./v for k, v in total_slide.items()}
    return weight

import time
def process_fname(fname, mosaics, test_mosaics, metadata, site, weight, cosine_threshold, temp_results_dir,temp_results_time_dir):
    print(f"processing {fname} for {site} started ...")
    WSI = test_mosaics.loc[fname]["features"]
    k = len(WSI)
    Bag = {}
    Entropy = {}
    mosaics_no_patient = mosaics[mosaics['patient_id'] != fname[0:12]]
    t0 = time.time()
    for patch_idx, patch_feature in enumerate(WSI):
        # Retreiving similar patches (creating Bag)
        if site == "organ":
            bag = [(idx, cosine_sim(patch_feature, row["features"])) for idx, row in mosaics_no_patient.iterrows() if cosine_sim(patch_feature, row["features"]) >= cosine_threshold]
        else:
            site_mosaics = mosaics_no_patient.loc[list(metadata.loc[mosaics_no_patient.loc[:, "file_name"], "primary_site"] == site)].copy()
            bag = [(idx, cosine_sim(patch_feature, row["features"])) for idx, row in site_mosaics.iterrows() if cosine_sim(patch_feature, row["features"]) >= cosine_threshold]
        Bag[patch_idx] = sorted(bag, key=lambda x: x[1], reverse=True)
        t = len(Bag[patch_idx])

        # Calculating entropy for each query patch in the Bag
        entropy = 0
        if site == "organ":
            u = set([sites_dict[metadata.loc[mosaics_no_patient.loc[idx, "file_name"], "primary_site"]] for (idx, _) in Bag[patch_idx]])
            for organ in u:
                num, denum = 0, 0
                for (idx, sim) in Bag[patch_idx]:
                    bag_organ = sites_dict[metadata.loc[mosaics_no_patient.loc[idx, "file_name"], "primary_site"]]
                    num += ((organ==bag_organ) * 1) * ((sim + 1) / 2) * weight[bag_organ]
                    denum += ((sim + 1) / 2) * weight[bag_organ]
                p = num / denum
                entropy -= p * np.log(p)
        else:
            u = set([metadata.loc[site_mosaics.loc[idx, "file_name"], "project_name"] for (idx, _) in Bag[patch_idx]])
            for diagnosis in u:
                num, denum = 0, 0
                for (idx, sim) in Bag[patch_idx]:
                    bag_diagnosis = metadata.loc[site_mosaics.loc[idx, "file_name"], "project_name"]
                    num += ((diagnosis==bag_diagnosis) * 1) * ((sim + 1) / 2) * weight[bag_diagnosis]
                    denum += ((sim + 1) / 2) * weight[bag_diagnosis]
                p = num / denum
                entropy -= p * np.log(p)
        Entropy[patch_idx] = entropy
        
    # Sorting Bag members in terms of descending entropy
    Bag = dict(sorted(Bag.items(), key=lambda x: Entropy[x[0]], reverse=True))

    # Calculating eta threshold for each query patch in the Bag
    eta_threshold = 0
    for patch_idx in range(len(WSI)):
        eta = np.mean([x[1] for x in Bag[patch_idx][:5]]) if len(Bag[patch_idx]) else 0
        # eta = 0 if np.isnan(eta) else eta
        eta_threshold += eta 
    eta_threshold = eta_threshold / k

    # Removing query patches in the Bag with small eta (eta < eta_threshold) 
    ids = []
    for idx, bag in Bag.items():
        eta = np.mean([x[1] for x in bag[:5]]) if len(bag) else 0
        # eta = 0 if np.isnan(eta) else eta
        if eta < eta_threshold:
            ids.append(idx)
    for idx in ids:
        del Bag[idx]

    # Majority voting for retrieving the results
    WSIRet = {}
    for idx, bag in Bag.items():
        if site == "organ":
            matches = [sites_dict[metadata.loc[mosaics_no_patient.loc[b[0], "file_name"], "primary_site"]] for b in bag[:5]]
            slides = [mosaics_no_patient.loc[b[0], "slide_path"] for b in bag[:5]]
        else:
            matches = [metadata.loc[site_mosaics.loc[b[0], "file_name"], "project_name"] for b in bag[:5]]
            slides = [site_mosaics.loc[b[0], "slide_path"] for b in bag[:5]]
        sims = [b[1] for b in bag[:5]]
        # Using slide path as the key
        slide_path = slides[matches.index(mode(matches))]
        if slide_path not in WSIRet:
            WSIRet[slide_path] = (slide_path, sims[matches.index(mode(matches))], mean(sims))
    WSIRet = list(WSIRet.values())

    total = time.time() - t0
    with open(join(temp_results_dir, f"{splitext(fname)[0]}_bag.pkl"), "wb") as f:
        pickle.dump(Bag, f)
    with open(join(temp_results_dir, f"{splitext(fname)[0]}_entropy.pkl"), "wb") as f:
        pickle.dump(Entropy, f)
    with open(join(temp_results_dir, f"{splitext(fname)[0]}_WSIRet.pkl"), "wb") as f:
        pickle.dump(WSIRet, f)

    with open(join(temp_results_time_dir, f"{splitext(fname)[0]}.txt"),'w') as file:
        file.write(str(total))
    print(f"processing {fname} for {site} ended successfully ...")

    return Bag, Entropy, eta, WSIRet

def worker(args):
    fname, mosaics, test_mosaics, metadata, site, weight, cosine_threshold, temp_results_dir, temp_results_time_dir = args
    return fname, *process_fname(fname, mosaics, test_mosaics, metadata, site, weight, cosine_threshold, temp_results_dir,temp_results_time_dir)

def wsi_query(mosaics, test_mosaics, metadata, site, weight, cosine_threshold, results_dir, temp_results_dir,temp_results_time_dir):
    Bags = {}    # Dictionary to store each Bag for each query WSI
    Entropies = {}    # Dictionary to store entropies for each patch in each Bag for each query WSI
    Etas = {}    # Dictionary to store eta thresholds for each patch in each Bag for each query WSI
    Results = {}    # Dictionary to store top-N similar WSIs to query WSI
    
    # Prepare arguments to be passed to the worker function
    import os
    files = glob.glob(f'/home/u1904706/cloud_workspace/testingRetrival/results_search/{site}/temp/*_WSIRet.pkl')
    files = [os.path.basename(x)[:-11]+'.npy' for x in files]
    args_list = [(fname, mosaics, test_mosaics, metadata, site, weight, cosine_threshold, temp_results_dir,temp_results_time_dir) for fname in test_mosaics.index.unique() if fname not in files]

    # Use ProcessPoolExecutor to process the tasks in parallel
    with ProcessPoolExecutor(max_workers=16) as executor:
        # Parallel execution of worker function
        for fname, Bag, Entropy, eta, WSIRet in tqdm(executor.map(worker, args_list)):
            Bags[fname] = Bag
            Entropies[fname] = Entropy
            Etas[fname] = eta
            Results[fname] = WSIRet

    with open(join(results_dir, f"Bags.pkl"), "wb") as f:
        pickle.dump(Bags, f)
    with open(join(results_dir, f"Entropies.pkl"), "wb") as f:
        pickle.dump(Entropies, f)
    with open(join(results_dir, f"Etas.pkl"), "wb") as f:
        pickle.dump(Etas, f)
    with open(join(results_dir, f"Results.pkl"), "wb") as f:
        pickle.dump(Results, f)

    return Results, Bags, Entropies, Etas

if __name__ == "__main__":
    cosine_threshold = 0.7

    metadata_path = f'../data/WSIRetrival/metadata.csv'
    mosaics_path = f'./mosaics'
    RESULTS_DIR = f'./retcll_search'
    RESULTS_TIME_DIR = f'./search_times'


    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index('file_name')

    # Pick a specfic organ you would like to do
    # With this you can submit multiple jobs, one per site
    # organ = 'Melanocytic' #DONE
    # organ = 'Liver' #DONE
    # organ = 'Gastrointestinal'#DONE
    # organ = 'Endocrine'#DONE
    # organ = 'Pulmonary'#DONE
    organ = 'Gynecologic'
    # organ = 'Urinary'#DONE
    # organ = 'Prostate' #DONE
    # organ = 'Brain' #DONE

    mosaics = pd.read_hdf(f'{mosaics_path}/mosaics_{organ}.h5', 'df')
    mosaics['patient_id'] = mosaics['file_name'].apply(lambda x: x[0:12])

    def get_site(file_name):
        return metadata.loc[file_name,'primary_site']
    test_mosaics = mosaics.copy()
    test_mosaics['site'] = test_mosaics['file_name'].apply(get_site)
    test_mosaics = test_mosaics[test_mosaics['site'] == organ].copy()
    test_mosaics = test_mosaics.set_index(['file_name'], inplace=False)


    results_dir = join(RESULTS_DIR, f"{organ}")
    temp_results_dir = join(results_dir, "temp")
    makedirs(temp_results_dir, exist_ok=True)
    
    results_time_dir = join(RESULTS_TIME_DIR, f"{organ}")
    temp_results_time_dir = join(results_time_dir, "temp")
    makedirs(temp_results_time_dir, exist_ok=True)

    weight = calculate_weights(organ,metadata)
    Results, Bags, Entropies, Etas = wsi_query(mosaics, test_mosaics, metadata, organ, weight, cosine_threshold, results_dir, temp_results_dir,temp_results_time_dir)
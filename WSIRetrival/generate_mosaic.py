from glob import glob
import os
from tqdm import tqdm
import numpy as np
from os.path import join, abspath
import pandas as pd
WSIS_PATH_FILTERED = '/home/u1904706/Desktop/MMDKernels/Features/TCGA/RetSSL-FEATS-FILTERED/'
RESULTS = '/home/u1904706/Desktop/MMDKernels/wsiRetrivalExternal/results'
images = glob(os.path.join(WSIS_PATH_FILTERED,'*_feat.npy'))
mosaics = []

SITE_DICT = {"Pulmonary":["TCGA-LUAD","TCGA-LUSC","TCGA-MESO"],
             "Urinary": ["TCGA-BLCA","TCGA-KIRC","TCGA-KICH","TCGA-KIRP"],
             "Gastrointestinal" : ["TCGA-COAD","TCGA-ESCA","TCGA-READ","TCGA-STAD"],
             "Melanocytic" : ["TCGA-UVM","TCGA-SKCM"],
             "Brain":["TCGA-GBM","TCGA-LGG"],
             "Liver": ["TCGA-CHOL","TCGA-LIHC","TCGA-PAAD"],
             "Gynecologic": ["TCGA-UCEC","TCGA-CESC","TCGA-UCS","TCGA-OV"],
             "Endocrine": ["TCGA-ACC","TCGA-PCPG","TCGA-THCA"],
             "Hematopoiesis": ["TCGA-DLBC","TCGA-THYM"],
             "Prostate":["TCGA-TGCT","TCGA-PRAD"],
             }

mosaics = {"Pulmonary":[],
             "Urinary": [],
             "Gastrointestinal" : [],
             "Melanocytic" : [],
             "Brain":[],
             "Liver": [],
             "Gynecologic": [],
             "Endocrine": [],
             "Hematopoiesis": [],
             "Prostate":[]
             }

metadata = pd.read_csv('/home/u1904706/Desktop/MMDKernels/wsiRetrivalExternal/metadata.csv')
metadata = metadata.set_index('file_name')

total = 0
for slide_id in tqdm(images):
    slide_path = slide_id
    file_name = os.path.basename(slide_id)
    wsi = np.load(slide_id)
    site = metadata.loc[file_name,'primary_site']

    if site not in list(mosaics.keys()):
        continue
    total += 1
    for row in wsi:
        mosaics[site].append([slide_path,file_name,row])


for key in mosaics.keys():
    df = pd.DataFrame(mosaics[key],columns=['slide_path','file_name','features'])
    df.to_hdf(join(RESULTS, f"mosaics_{key}.h5"), key="df", mode="w")
    print(df.head())
import os
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from tqdm import tqdm 

K_1 = 9
R = 0.2

WSIS_PATH = '../data/Features/RetSSL-FEATS/'
WSIS_PATH_FILTERED = '../data/Features/RetSSL-FEATS-FILTERED/'

images = glob(os.path.join(WSIS_PATH,'*_feat.npy'))

for image_path in tqdm(images):
    wsi = np.load(image_path)

    # first kmeans
    final_indicies = []
    if wsi.shape[0] < 9:
        K_1 = wsi.shape[0]
    
    kmeans = KMeans(n_clusters=K_1,random_state=46, n_init="auto").fit(wsi)
    cluster_labels = kmeans.labels_

    for cluster_idx in range(0,K_1):
        patch_idx = np.where(cluster_labels==cluster_idx)[0]
        K_2 = max(1,round(len(patch_idx)*0.2))

        image_pos_path = os.path.join(WSIS_PATH,os.path.basename(image_path)[:-9]+'_pos.npy')
        wsi_pos = np.load(image_pos_path)

        wsi_pos_filtered = wsi_pos[patch_idx,:]


        kmeans_second = KMeans(n_clusters=K_2,random_state=46, n_init="auto").fit(wsi_pos_filtered)

        closest, distances = vq(kmeans_second.cluster_centers_, wsi_pos_filtered)

        final_indicies = final_indicies + list(patch_idx[closest])


    final_indicies = np.array(list(set(final_indicies)))
    filtered_wsi = wsi[final_indicies,:]
    np.save(os.path.join(WSIS_PATH_FILTERED,os.path.basename(image_path)),filtered_wsi)
    # Reset incase it got updated if wsi.shape[0] < 9
    K_1 = 9      
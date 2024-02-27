# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from geomloss import SamplesLoss

def create_slide_ids(SLIDE_IDS_PATH,FEATURES_PATH):
    print(SLIDE_IDS_PATH)

    if os.path.exists(SLIDE_IDS_PATH):
        slide_IDs = np.load(SLIDE_IDS_PATH)
    else:
        import glob
        feat_filter_path = os.path.join(FEATURES_PATH,'*_feat.npy')
        slide_IDs = glob.glob(feat_filter_path)
        np.save(SLIDE_IDS_PATH,slide_IDs)
    return slide_IDs

class DistanceMatrix:
  def __init__(self, slides):
    """
        Parameters
        ----------
        numFeatures : int
            The number of features of each patch represnation
        slides : array
            The IDs of the slides in the dataset
    """
    self.slides = slides

  def MMD_distance(self, distribution, blur, start,end, savePath, D=None):
    """ Calculates the MMD distance for the dataset.

        Parameters
        ----------
        distribution : str
            The distribution used for the dataset e.g. 'gaussian', 'laplacian' etc.
        blur : float
            the standard deviation of the kernel
        start : int
            If some rows of the matrx are already computed from a previous run then can set start row to the last compiled row
        savePath : str
            As colab times out every 10 rows the kernel progress is saved to this destination
        D: N * N array
            Can pass in a partially computed kernel from a previous run to carry on without resetting progress
    """

    loss = SamplesLoss(loss=distribution, blur=blur, backend="auto")
    elements = len(self.slides)

    if (D is None):
      D = np.zeros([elements,elements], dtype=np.float32)

    rowsDone = start

    for i in range(start,end):
 
      x = torch.tensor(np.load(self.slides[i],mmap_mode='r'),dtype=torch.float32,device=torch.device('cuda:0'))
      for j in range(i+1,elements):
        if(D[i,j] != 0.0):
          continue

        y = torch.tensor(np.load(self.slides[j],mmap_mode='r'),dtype=torch.float32,device=torch.device('cuda:0'))
        L = loss(x,y)
        

        D[i,j] = L.item()
    
        del y,L
      del x

      rowsDone += 1
      if(rowsDone % 200 == 0 or rowsDone==end):
        np.save(os.path.join(savePath, str(rowsDone) + '.npy'), D)
    
    return self.save_full_D(D,'D_full.npy',savePath)

  def full_D(self,upper_D,filename,output_path):
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

    path = os.path.join(output_path,filename)
    np.save(path,D_cpy)
    return D_cpy

if __name__ == "__main__":

    FEATURES_PATH = '/media/u1904706/Data/Features/RetSSL-FEATS'
    SAVE_PATH = '/media/u1904706/Data/MMDKernels/RetSSL-FEATS'


    SLIDE_IDS_PATH = os.path.join(SAVE_PATH,'IDS.npy')
    slide_IDs = create_slide_ids(SLIDE_IDS_PATH,FEATURES_PATH)
    slide_IDs = [os.path.join(FEATURES_PATH,i) for i in slide_IDs]
    
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('start', type=int,default=0,
                    help='Row to start the computation')
    parser.add_argument('end', type=int,default=0,
                    help='Row to end the computation')
    args = parser.parse_args()

    D_MATRIX_PATH = os.path.join(SAVE_PATH,"checkpoints")
    from pathlib import Path
    Path(D_MATRIX_PATH).mkdir(parents=True, exist_ok=True)

    START = args.start
    END = args.end
    if END == 0:
       END = len(slide_IDs)
    
    calcD = DistanceMatrix(slides = slide_IDs)
    
    D = calcD.MMD_distance(
        distribution = 'gaussian',
        blur=10.0,
        start=START,
        end=END,
        savePath=D_MATRIX_PATH,
        D=None)
    
    
    
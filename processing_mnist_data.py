#!/usr/bin/env python3
import matplotlib
import argparse
import random
import os

import numpy  as np
import scipy  as sp
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg          import norm, svd
from sklearn.decomposition import PCA
from tqdm                  import tqdm

#color_s = ['red', 'black', 'blue', 'brown', 'green']
color_s = list(matplotlib.colors.cnames.keys())
random.shuffle(color_s)

def parseArguments():
    description = 'Process MNIST data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument( '--input'    ,type=str, default = "data/test16_x_.npy" )
    parser.add_argument( '--threshold',type=float, default = 0.95 )
    parser.add_argument( '--verbose'  , action='store_true' )
    args = parser.parse_args()

    return args.input, args.threshold, args.verbose

def applySphericalPCA( X, r, lam=5, mu=5, K=200):
    residual_s = np.zeros(K)

    m, n = X.shape
    
    U = np.identity(m)[:,:r]
    V = np.random.random((r,n))
    
    for k in tqdm( range(K) ):
        for j in range(n):
            V[:,j] = (lam - 2)*V[:,j] + 2*U.T @ X[:,j]
            V[:,j] = V[:,j] / norm(V[:,j])
        M = 2*(X - U@V)@V.T + mu*U
        Y, _, Z = svd(M, full_matrices=False)
        U = Y[:,:r]@Z
        
        residual_s[k] = norm(X-U@V,'fro')
    
    return U, V,residual_s 

def applyBasePCA( X ):
    U, S, Vh = svd(X)
    return U, S, Vh

inputFilename, thresholdValue, verboseMode = parseArguments()
X_s  = np.load( inputFilename )

U_s, S_s, V_s = applyBasePCA( X_s.T )

index_s = np.argsort(-S_s)
cumsum_S_s = np.cumsum( S_s[index_s] )/np.sum(S_s)

r = np.argmax(cumsum_S_s > thresholdValue)

U_s, V_s, residual_s = applySphericalPCA( X_s.T, r, 10, 10, 1000 )
V_s = V_s.T

np.save( f"data/test{r}_x_pca_.npy", V_s)

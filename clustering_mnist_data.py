#!/usr/bin/env python3
import numpy  as np
import scipy  as sp
import pandas as pd

from sklearn         import metrics

from spherecluster import VonMisesFisherMixture
from spherecluster import sample_vMF

import argparse

#Generate test dataset
def generateDataset(num_clusters, num_samples, num_dimensions):
    mu_s = np.random.randn(num_clusters, num_dimensions)
    mu_s /= np.linalg.norm(mu_s, axis=0)
    
    kappa_value = 10
    kappa_s = [kappa_value] * num_clusters
    
    X_s_numpy = np.zeros( (num_clusters, num_samples, num_dimensions) )
    Y_s_numpy = np.zeros( (num_clusters, num_samples) )
    for index in range(num_clusters):
        X_s_numpy[index] = sample_vMF( mu_s[index], kappa_s[index], num_samples)
        Y_s_numpy[index] = index
    
    return X_s_numpy, Y_s_numpy

#Parse input arguments
def parseArguments():
    description = 'Clustering MNIST vector'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument( '--input'  , default='input.npy' ,type=str )
    parser.add_argument( '--mode'   , default='soft' )
    parser.add_argument( '--verbose', action='store_true' )
    args = parser.parse_args()

    return args.input, args.mode, args.verbose

inputFileName, processMode, verboseMode = parseArguments()


num_clusters   = 10
num_samples    = 10
num_dimensions = 8

X_s, Y_s = generateDataset(num_clusters, num_samples, num_dimensions)
X_s = X_s.reshape(num_clusters*num_samples, num_dimensions)
Y_s = Y_s.reshape(num_clusters*num_samples)

if processMode == 'soft':
    vmf_model = VonMisesFisherMixture(n_clusters = 10, posterior_type='soft')
    vmf_model.fit(X_s)
elif processMode == 'hard':
    vmf_model = VonMisesFisherMixture(n_clusters = 10, posterior_type='hard')
    vmf_model.fit(X_s)

print( vmf_model.concentrations_  )
print( vmf_model.labels_          )
print( vmf_model.weights_         )
print( vmf_model.cluster_centers_ )


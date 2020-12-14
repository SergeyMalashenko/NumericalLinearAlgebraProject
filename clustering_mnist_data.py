#!/usr/bin/env python3
import numpy  as np
import scipy  as sp
import pandas as pd

from spherecluster          import VonMisesFisherMixture
from spherecluster          import sample_vMF
from scipy.spatial.distance import cdist
from sklearn                import metrics

import argparse

def generateUniformUnitVectors(num_clusters, num_dimensions):
    mu_s = np.random.randn(num_clusters, num_dimensions)
    mu_s /= np.linalg.norm(mu_s, axis=0)
 
    kappa_value = 10
    kappa_s = [kappa_value] * num_clusters
    
    return mu_s, kappa_s


#Generate test dataset
def generateDataset( mu_s, kappa_s, num_samples):
    num_clusters, dim = mu_s.shape

    X_s_numpy = np.zeros( (num_clusters, num_samples, dim) )
    Y_s_numpy = np.zeros( (num_clusters, num_samples) )
    for index in range(num_clusters):
        X_s_numpy[index] = sample_vMF( mu_s[index], kappa_s[index], num_samples)
        Y_s_numpy[index] = index
    
    return X_s_numpy, Y_s_numpy

#Parse input arguments
def parseArguments():
    description = 'Clustering MNIST vector'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument( '--x_s'    , type=str )
    parser.add_argument( '--y_s'    , type=str )
    parser.add_argument( '--mu_s'   , type=str )
    parser.add_argument( '--mode'   , default='soft' )
    parser.add_argument( '--verbose', action='store_true' )
    args = parser.parse_args()

    return args.x_s, args.y_s, args.mu_s, args.mode, args.verbose

input_X_s, input_Y_s, input_mu_s, processMode, verboseMode = parseArguments()

X_s  = np.load( input_X_s  )
Y_s  = np.load( input_Y_s  )
mu_s = np.load( input_mu_s )

num_clusters = np.unique(Y_s).size

print(num_clusters)

if processMode == 'soft':
    vmf_model = VonMisesFisherMixture(n_clusters = num_clusters, posterior_type='soft')
    vmf_model.fit(X_s)
elif processMode == 'hard':
    vmf_model = VonMisesFisherMixture(n_clusters = num_clusters, posterior_type='hard')
    vmf_model.fit(X_s)

estimated_kappa_s = vmf_model.concentrations_
estimated_mu_s    = vmf_model.cluster_centers_

print(estimated_kappa_s )

cross_distance_s = cdist(estimated_mu_s, mu_s, metric='cosine')
print(cross_distance_s)

#print( vmf_model.labels_          )
#print( vmf_model.weights_         )


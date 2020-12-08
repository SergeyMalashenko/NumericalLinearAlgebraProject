#!/usr/bin/env python3
import argparse
import os

import numpy  as np
import scipy  as sp
import pandas as pd

from numpy.linalg import norm, svd

from spherecluster          import sample_vMF

def parseArguments():
    description = 'Process MNIST data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument( '--input'  , default='input.csv' ,type=str, help='input file' )
    parser.add_argument( '--output' , default='output.csv',type=str, help='output file')
    parser.add_argument( '--verbose', action='store_true' )
    args = parser.parse_args()

    return args.input, args.output, args.verbose

def generateUniformUnitVectors(num_clusters, num_dimensions):
    mu_s = np.random.randn(num_clusters, num_dimensions)
    mu_s /= np.linalg.norm(mu_s, axis=0)
 
    kappa_value = 10
    kappa_s = [kappa_value] * num_clusters
    
    return mu_s, kappa_s

def randomProjectUnitVectors( mu_s, kappa_s, num_dimensions):
    return mu_s, kappa_s

def generateDataset( mu_s, kappa_s, num_samples):
    num_clusters, dim = mu_s.shape

    X_s_numpy = np.zeros( (num_clusters, num_samples, dim) )
    Y_s_numpy = np.zeros( (num_clusters, num_samples) )
    for index in range(num_clusters):
        X_s_numpy[index] = sample_vMF( mu_s[index], kappa_s[index], num_samples)
        Y_s_numpy[index] = index
    
    return X_s_numpy, Y_s_numpy

def applySphericalPCA( X, r, lam ,mu, K ):
    m, n = X.shape
    U = np.zeros(m,r)
    V = np.array(r,n)
    
    for _ in range(K):
        M = 2*(X - U@V)@V.T + mu*U
        Y,S,Z = sdv(M)
        U = Y@Z.T
        for j in range(n):
            q = 2*U.T @ X[:,j] + (lam - 2)*V[:,j]
            V[:,j] = q / norm(q)
    return U, V
    '''
    target_data = None
    
    data_ori=data;
    lambda=75;
    mu=lambda;
    
    %[ss,vv,dd]=svd(data_ori,0);
    x = data;
    x = normc(x);
    
    
    I = eye(ori_dim);
    U = I(:,1:pro_dim);
    niter=200;
    V = rand(pro_dim,data_size);
    obj=zeros(niter,1);
    for i=1:niter
        %V_old = V;
        for j=1:data_size
            V(:,j) = (lambda-2)*V(:,j)+2*U'*x(:,j);
            V(:,j) = V(:,j)/norm(V(:,j));
        end
        %gra = sum(dot(U'*(U*V-x),V_old-V))
        M = 2*(x-U*V)*V'+mu*U;
        [s,~,d]=svd(M,0);
        U=s*d';
        %obj(i)=norm(x-U*V,'fro');
    end
    '''
    return target_data, tolerance

def main():
    #input_filename, output_filename, verbose = parseArguments()
    #input_df = pd.read_csv( input_filename )

    num_clusters   = 10
    num_samples    = 100
    num_dimensions = 100
    
    mu_s, kappa_s = generateUniformUnitVectors(num_clusters, num_dimensions)
    mu_s, kappa_s = randomProjectUnitVectors(mu_s, kappa_s, num_dimensions)
    X_s, Y_s = generateDataset(mu_s, kappa_s, num_samples)
    


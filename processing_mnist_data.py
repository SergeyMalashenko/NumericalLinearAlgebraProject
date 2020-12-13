#!/usr/bin/env python3
import argparse
import os

import numpy  as np
import scipy  as sp
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.linalg  import norm, svd
from spherecluster import sample_vMF

import scipy.io

#X_s = scipy.io.loadmat('temporary.mat')['data']
#X_s = X_s / norm(X_s,axis=0, keepdims = True)
#print(X_s)

def parseArguments():
    description = 'Process MNIST data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument( '--input'  , default='input.csv' ,type=str, help='input file' )
    parser.add_argument( '--output' , default='output.csv',type=str, help='output file')
    parser.add_argument( '--verbose', action='store_true' )
    args = parser.parse_args()

    return args.input, args.output, args.verbose

def generateUniformUnitVectors(num_clusters, num_dimensions):
    random_shift = 2*np.pi*np.random.random(1)
    theta = np.linspace(0 + random_shift[0], 2*np.pi + random_shift[0], num_clusters, endpoint=True)
    mu_s = np.concatenate([np.cos(theta)[:,np.newaxis], np.sin(theta)[:,np.newaxis]], axis=1) 
    
    kappa_value = 1000
    kappa_s = [kappa_value] * num_clusters
    return mu_s, kappa_s

def randomRotateUnitVectors( mu_s, X_s ):
    theta = 2*np.pi*np.random.random(3)
    rotate_z = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]), 0],
        [np.sin(theta[2]),  np.cos(theta[2]), 0],
        [               0,                 0, 1]
    ])
    rotate_y = np.array([
        [np.cos(theta[1]), 0, -np.sin(theta[1])],
        [               0, 1,                 0],
        [np.sin(theta[1]), 0,  np.cos(theta[1])]
    ])
    rotate_x = np.array([
        [1,                0,                0],
        [0, np.cos(theta[0]),-np.sin(theta[0])],
        [0, np.sin(theta[0]), np.cos(theta[0])]
    ])
    
    updated_X_s  = X_s  @ rotate_z @ rotate_y @ rotate_x  
    updated_mu_s = mu_s @ rotate_z @ rotate_y @ rotate_x  
    return updated_mu_s, updated_X_s

def generateDataset( mu_s, kappa_s, num_samples):
    num_clusters, dim = mu_s.shape

    X_s_numpy = np.zeros( (num_clusters, num_samples, dim) )
    Y_s_numpy = np.zeros( (num_clusters, num_samples) )
    for index in range(num_clusters):
        X_s_numpy[index] = sample_vMF( mu_s[index], kappa_s[index], num_samples)
        Y_s_numpy[index] = index
    
    X_s_numpy = X_s_numpy.reshape((num_clusters*num_samples, dim))
    Y_s_numpy = Y_s_numpy.reshape((num_clusters*num_samples))

    return X_s_numpy, Y_s_numpy

def applySphericalPCA( X, r, lam=5, mu=5, K=200):
    residual_s = np.zeros(K)

    m, n = X.shape
    
    U = np.identity(m)[:,:r]
    V = np.random.random((r,n))
    
    for k in range(K):
        for j in range(n):
            V[:,j] = (lam - 2)*V[:,j] + 2*U.T @ X[:,j]
            V[:,j] = V[:,j] / norm(V[:,j])
        M = 2*(X - U@V)@V.T + mu*U
        Y, _, Z = svd(M, full_matrices=False)
        U = Y[:,:r]@Z
        
        residual_s[k] = norm(X-U@V,'fro')
    
    return U, V,residual_s 

def applyBasePCA():
    return

color_s = ['red', 'black', 'blue', 'brown', 'green']

num_clusters   = 5
num_samples    = 100
num_dimensions = 2

mu_s, kappa_s = generateUniformUnitVectors(num_clusters, num_dimensions)
X_s, Y_s = generateDataset(mu_s, kappa_s, num_samples)

X_s  = np.concatenate([X_s,  np.zeros((X_s .shape[0],1))], axis=1)
mu_s = np.concatenate([mu_s, np.zeros((mu_s.shape[0],1))], axis=1)

mu_s, X_s = randomRotateUnitVectors(mu_s, X_s)

fig1 = plt.figure( figsize=plt.figaspect(0.5) )
ax1 = fig1.add_subplot(1, 2, 1, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x, y, z, alpha=0.1)

zeros = np.zeros(mu_s.shape[0])
for c in range(len(np.unique(Y_s))):
    ax1.plot(X_s[Y_s==c, 0], X_s[Y_s==c, 1], X_s[Y_s==c, 2], '.', color=color_s[c])
ax1.quiver3D(zeros, zeros, zeros, mu_s[:,0], mu_s[:,1], mu_s[:,2], color='black', arrow_length_ratio=0.001)

U_s, V_s, residual_s = applySphericalPCA( X_s.T, 2, 10, 10, 1000 )
V_s = V_s.T

print( residual_s )

ax2 = fig1.add_subplot(1, 2, 2)

u = np.linspace(0, 2 * np.pi, 100)
x = np.cos(u)
y = np.sin(u)
ax2.plot(x, y, alpha=0.1)

for c in range(len(np.unique(Y_s))):
    ax2.plot(V_s[Y_s==c, 0], V_s[Y_s==c, 1], '.', color=color_s[c])

plt.show()


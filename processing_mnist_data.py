#!/usr/bin/env python3
import argparse
import os

import numpy  as np
import scipy  as sp
import pandas as pd

def parseArguments():
    description = 'Process MNIST data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument( '--input'  , default='input.csv' ,type=str, help='input file' )
    parser.add_argument( '--output' , default='output.csv',type=str, help='output file')
    parser.add_argument( '--verbose', action='store_true' )
    args = parser.parse_args()

    return args.input, args.output, args.verbose

def applySphericalPCA( source_data ):
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
    input_filename, output_filename, verbose = parseArguments()
    
    input_df = pd.read_csv( input_filename )

    


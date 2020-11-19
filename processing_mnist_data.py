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

def main():
    input_filename, output_filename, verbose = parseArguments()
    
    input_df = pd.read_csv( input_filename )




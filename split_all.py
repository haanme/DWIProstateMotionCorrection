# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:52:00 2014

@author: merisaah
"""

#!/usr/bin/env python
output_path = 'splitted'

from argparse import ArgumentParser
import sys
import os
import time
import numpy as np
import bfitASCII_IO
import glob
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nosub", dest="subs", help="Number of subdivisions per file", required=True)
    args = parser.parse_args()

    subs = int(float(args.subs))

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filenames_raw = glob.glob(('*ASCII.txt'))

    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    for fname_raw in filenames_raw:
        splitted_raw = fname_raw.split('ASCII')
        print splitted_raw
        print 'Reading ASCII file: ' + (fname_raw)
        # Read data in
        data = ASCIIio.Read((fname_raw), True)
        values = data['data']
        total_SIs = values.shape[0]
        print 'Total number of SIs is:' + str(total_SIs)
        start_i = 0
        # Resolve iteration so that the SIs are divided evenly among files
        iter = np.ceil(total_SIs/subs)
        if np.mod(total_SIs, subs) != 0:
            iter += 1
        if iter==0:
            iter = 1
        total_subs = 0
        while start_i < total_SIs:
            total_subs += 1
            start_i += iter
        print 'Total number of division files is:' + str(total_subs)
        # Write division data into output path
        file_i = 1
        start_i = 0
        while start_i < total_SIs:
            filename = splitted_raw[0] + str(file_i) + 'of' + str(total_subs) + '_ASCII' + splitted_raw[1]
            subdata = data
            subvalues = values[start_i:start_i+iter,:]
            print 'Writing ' + str(subvalues.shape[0]) + ' values from ' + str(start_i) + ' to ' + str(start_i+iter) + ' into ' + filename
            subdata['data'] = subvalues
            ASCIIio.Write3D(output_path + os.sep + filename, subdata)
            start_i += iter
            file_i += 1

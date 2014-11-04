# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:52:00 2014

@author: merisaah
"""

#!/usr/bin/env python
input_path = 'splitted_results'

from argparse import ArgumentParser
import sys
import os
import time
import numpy as np
import bfitASCII_IO
import glob
import numpy as np

if __name__ == "__main__":
    #    parser = ArgumentParser()
    #parser.add_argument("--nosub", dest="subs", help="Number of subdivisions per file", required=True)
    #args = parser.parse_args()

    #subs = int(float(args.subs))

    errors = 0

    filenames_raw = glob.glob((input_path + os.sep + '*_results.txt'))

    processed_subjects = []

    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    for fname_raw in filenames_raw:
        data_SI_orig = ASCIIio.Read(fname_raw, True)
        
        splitted_raw = fname_raw.split(os.sep)
        basename = splitted_raw[-1]
        splitted_basename = basename.split('_')
        subjectname = "_".join(splitted_basename[:-5])
        suffix = "_".join(splitted_basename[-3:])
        processed=False
        for processed_subject in processed_subjects:
            if processed_subject == subjectname:
                processed=True
                break
        if processed:
            continue
        processed_subjects.append(subjectname)
                
        str_index = splitted_basename[-4]
        str_index = str_index.split('of')
        str_no = float(str_index[0])
        str_total = int(float(str_index[1]))
        print "Combining " + subjectname
        execution_time_total = 0
        for file_no in range(1,str_total+1):
            filename = subjectname + '_' + splitted_basename[-5] + '_' + str(file_no) + 'of' + str(str_total) + '_' + suffix
            if not os.path.isfile(input_path + os.sep + filename):
                print "ERROR file " + filename + " does not exist"
            # Read data in
            data = ASCIIio.Read((input_path + os.sep + filename), False)
            values = data['data']
            if file_no == 1:
                values_all = values
            else:
                values_all = np.concatenate((values_all, values),0)
            total_SIs = values.shape[0]
            executiontime_str = data['executiontime'].split()
            execution_time_total += int(float(executiontime_str[0]))
            print '\t' + filename + ' ' + str(total_SIs) + ' SIs execution time ' + data['executiontime']
        print 'Total number of SIs is:' + str(values_all.shape)
        out_filename = subjectname + '_' + splitted_basename[-5] + '_' + suffix
        print "Writing " + out_filename
        data['data'] = values_all
        data['executiontime'] = execution_time_total
        ASCIIio.Write3D(out_filename, data, False)


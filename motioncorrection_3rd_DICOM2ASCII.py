#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'

#
# Convert DICOM data to ASCII for fitting
#
# in_dir     - DICOM directory
# out_prefix - patient subdir
# offset     - offset that is used to shrink imge region in xy-planes
#
def DICOM2ASCII(in_dir, out_dir, out_prefix, subwindow):
    import dicom
    import DicomIO
    import bfitASCII_IO
    import numpy as np
    import os
    import shutil

    outdir = out_dir + '/' + out_prefix + '_' + in_dir + '_ASCII.txt'
    if os.path.isfile(outdir):
        os.remove(outdir)

    # Resolve new frame list
    dcmio = DicomIO.DicomIO()
    frame_list = dcmio.ReadDICOM_frames(experiment_dir + '/' + out_prefix + '/' + in_dir)
    slice_1st = frame_list[0][0]
    xdim = slice_1st.Columns
    ydim = slice_1st.Rows
    zdim = slice_1st.NumberOfSlices
    tdim = slice_1st.NumberOfTimeSlices

    ROI_No = 0

    bset = []
    for frame_i in range(len(frame_list)):
        bset.append(int(frame_list[frame_i][0].FrameReferenceTime)/1000)

    ROIslice = []
    for slice_i in range(len(frame_list[0])):
        ROIslice.append(slice_i+1)

    name = in_dir
    # Save in data in order z,y,x
    SIs = []
    for z_i in range(zdim):
        for y_i in range(offset, ydim-offset):
            for x_i in range(offset, xdim-offset):
                SI = []
                for t_i in range(tdim):
                    SI.append(frame_list[t_i][z_i].pixel_array[y_i, x_i] * frame_list[t_i][z_i].RescaleSlope + frame_list[t_i][z_i].RescaleIntercept)
                SIs.append(SI)
        print str(z_i+1) + '/' + str(zdim)
    print "total SIs:" + str(len(SIs))
    data = { 'subwindow': subwindow, 'ROI_No': ROI_No, 'bset': bset, 'ROIslice': ROIslice, 'name': name, 'SIs': SIs }
    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    print ("Writing " + outdir)
    ASCIIio.Write3D(outdir, data)

    return outdir

from argparse import ArgumentParser
import sys
import os
import DicomIO
import conversions as conv
import time
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--basedir", dest="basedir", help="base directory that has patient names under it", required=True)
    parser.add_argument("--outdir", dest="outdir", help="output directory", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    parser.add_argument("--subdir", dest="subdir", help="sub directory under DICOM dir to read", required=True)
    args = parser.parse_args()

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Read subregion coordinates of non-corrected and corrected DICOMs
    subregion_in_originalDICOM = np.loadtxt(experiment_dir + '/' + args.subject + '/subregion.txt')

    # Convert DICOM file into singel ASCII file
    print "Converting"
    try:
        DICOM2ASCII(args.subdir, args.subject, subregion_in_originalDICOM)
    except Exception as inst:
        raise
    sys.exit(0)

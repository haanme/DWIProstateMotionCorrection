#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'

#
# Convert DICOM data to ASCII for fitting
#
#
def DICOM2ASCII(out_prefix, sub_dir, out_dir, subwindow):
    import dicom
    import DicomIO
    import bfitASCII_IO
    import numpy as np
    import os
    import shutil

    outfile = out_dir + os.sep + out_prefix + '_' + sub_dir + '_ASCII.txt'
    if os.path.isfile(outfile):
        os.remove(outfile)

    # Resolve new frame list
    dcmio = DicomIO.DicomIO()
    frame_list = dcmio.ReadDICOM_frames(experiment_dir + os.sep + out_prefix + os.sep + sub_dir)
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

    name = sub_dir
    # Save in data in order z,y,x
    SIs = []
    for z_i in range(zdim):
        print type(frame_list[0][z_i].pixel_array[0, 0])
        print float(frame_list[0][z_i].pixel_array[0, 0])
        pixel_array3 = frame_list[0][z_i].pixel_array
        pixel_array3 = pixel_array3  * frame_list[0][z_i].RescaleSlope + frame_list[0][z_i].RescaleIntercept
        print pixel_array3[0, 0]
        print "Mean:" + str(np.mean(pixel_array3)) + " Min:" + str(np.amin(pixel_array3)) + " Max:" + str(np.amax(pixel_array3))
        for y_i in range(ydim):
            for x_i in range(xdim):
                SI = []
                for t_i in range(tdim):
                    SI.append(frame_list[t_i][z_i].pixel_array[y_i, x_i] * frame_list[t_i][z_i].RescaleSlope + frame_list[t_i][z_i].RescaleIntercept)
                SIs.append(SI)
        print str(z_i+1) + '/' + str(zdim)
    print "total SIs:" + str(len(SIs))
    data = { 'subwindow': subwindow, 'number': ROI_No, 'bset': bset, 'ROIslice': ROIslice, 'name': name, 'SIs': SIs }
    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    print ("Writing " + outfile)
    ASCIIio.Write3D(outfile, data)

    return outfile

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
    subregion_in_originalDICOM = np.loadtxt(experiment_dir + os.sep + args.subject + os.sep + 'subregion10.txt')

    # Convert DICOM file into singel ASCII file
    print "Converting"
    try:
        outfile = DICOM2ASCII(args.subject, args.subdir, args.outdir, subregion_in_originalDICOM)
        print outfile + " written"
    except Exception as inst:
        raise
    sys.exit(0)

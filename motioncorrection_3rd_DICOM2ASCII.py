#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
param_rigid = 'Par0001translation.txt'
param_BSpline = 'Par0001bspline08.txt'
#mask_matfile_basedir = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS/ROI_mat_files'
mask_matfile_basedir_hB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS_for_Motion_Cor/ROI_mat_files'
mask_matfile_basedir_lB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Lb_work_2rep/ROI_mat_files'

#
# Print all DICOM data
#
# input_dir    - DICOM directory that is printed
#
def print_dcm(input_file):
    import DicomIO
    dcmio = DicomIO.DicomIO()
    dwidcm = dcmio.ReadDICOM_frames(input_file)
    for vol_i in range(len(dwidcm)):
        print "frame " + str(vol_i+1)
        for z_i in range(len(dwidcm[vol_i])):
            print str(dwidcm[vol_i][z_i].FrameReferenceTime) + " - " + str(dwidcm[vol_i][z_i].ImagePositionPatient[2])

#
# Get subvolumes
#
# dwidcm        - DICOM source data
# volume_list   - list of volume indexes for output
# bounds        - bounds of subvolumes
# output_prefix - output prefix
#
def get_subvolumes(input_dir, volume_list, bounds, output_prefix):
    import dicom
    import DicomIO
    import shutil
    import numpy as np
    dcmio = DicomIO.DicomIO()
    from nipype.utils.filemanip import split_filename
    # resolve output directory and volumes
    out_dir_base = experiment_dir + '/' + output_prefix + '/' + 'subvolumes'
    filenames_all = []
    outdirs_all = []
    out_vols_all = []
    for vol_i in range(len(volume_list)):
        out_dir = out_dir_base + '_' + str(volume_list[vol_i])
        out_vols = []
        dwivolume = dwidcm[volume_list[vol_i]]
        #take subregion from volume
        for slice_i in range(len(dwivolume)):
            pixel_array = dwivolume[slice_i].pixel_array[bounds[2]:bounds[3],bounds[0]:bounds[1]]
            dwivolume[slice_i].PixelData = pixel_array.astype(np.uint16).tostring()
            dwivolume[slice_i].Columns = bounds[1]-bounds[0]
            dwivolume[slice_i].Rows = bounds[3]-bounds[2]
        #append volume to lists
        out_vols.append(dwivolume)
        out_vols_all.append(dwivolume)
        # Create output directory if it does not exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        filenames = dcmio.WriteDICOM_frames(out_dir, out_vols, 'IM')
        filenames_all.append(filenames)
        outdirs_all.append(out_dir)
    return outdirs_all, filenames_all, out_vols_all

#
# Convert DICOM data to ASCII for fitting
#
# in_dir     - DICOM directory
# out_prefix - patient subdir
# offset     - offset that is used to shrink imge region in xy-planes
# bounds     - box coordinates of DICOM inside the original DICOM data [xmin, xmax, ymin, ymax, zmin, zmax]
#
def DICOM2ASCII(in_dir, out_prefix, offset, bounds):
    import dicom
    import DicomIO
    import bfitASCII_IO
    import numpy as np
    import os
    import shutil

    outdir = experiment_dir + '/' + out_prefix + '/' + out_prefix + '_' + in_dir + '_ASCII.txt'
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

    subwindow_dimensions = [xdim-(2*offset), ydim-(2*offset), zdim, tdim]
    subwindow = [bounds[0]+offset, bounds[0]+xdim-offset, bounds[2]+offset, bounds[2]+ydim-offset]
    print "subwindow dimensions:" + str(subwindow_dimensions)
    print "subwindow:" + str(subwindow)

    ROI_No = 0

    bset = []
    for frame_i in range(len(frame_list)):
        bset.append(int(frame_list[frame_i][9].FrameReferenceTime)/1000)

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
    parser.add_argument("--dicomdir", dest="dicomdir", help="dicomdir", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    args = parser.parse_args()

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists((experiment_dir + '/' + args.subject + '/Noncorrected')):
        os.makedirs((experiment_dir + '/' + args.subject))

    # Read subregion coordinates of non-corrected and corrected DICOMs
    subregion_in_originalDICOM = np.loadtxt(experiment_dir + '/' + args.subject + '/subregion.txt')
    # Offset 45 means that there is 5 voxels outdise original region
    offset = 0
    # Write motioncorrected as single multifile DICOM subfolder
    print "Converting non-corrected"
    try:
        DICOM2ASCII('Noncorrected', args.subject, offset, subregion_in_originalDICOM)
    except Exception as inst:
        raise
    print "Converting motion corrected"
    try:
        DICOM2ASCII('Motioncorrected', args.subject, offset, subregion_in_originalDICOM)
    except Exception as inst:
        raise

    sys.exit(0)

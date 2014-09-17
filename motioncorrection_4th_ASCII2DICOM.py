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
# Convert ASCII fitting results to DICOM
#
# in_file    - ASCII input file
# in_dir     - DICOM directory for reference headers
# out_prefix - patient subdir
# bounds     - box coordinates of DICOM inside the original DICOM data [xmin, xmax, ymin, ymax, zmin, zmax]
#
def ASCII2DICOM(in_file, in_dir, out_prefix, bounds):
    import dicom
    import DicomIO
    import bfitASCII_IO
    import numpy as np
    import os
    import shutil

    outdir_basename = experiment_dir + '/' + out_prefix + '/' + in_file.rstrip('.txt')

    # Resolve new frame list
    dcmio = DicomIO.DicomIO()
    frame_list = dcmio.ReadDICOM_frames(experiment_dir + '/' + out_prefix + '/' + in_dir)
    slice_1st = frame_list[0][0]
    xdim = slice_1st.Columns
    ydim = slice_1st.Rows
    zdim = slice_1st.NumberOfSlices
    tdim = slice_1st.NumberOfTimeSlices
    sample_frame = frame_list[0]
    del frame_list

    # Read data = { 'subwindow': subwindow, 'ROI_No': ROI_No, 'bset': bset, 'ROIslice': ROIslice, 'name': name, 'SIs': SIs }
    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    print ("Reading " + in_file)
    data = ASCIIio.Read(in_file)
    pmap_subwindow = data['subwindow']
    pmap_SIs = data['data']
    pmap_names = data['parameters']
    pmap_slices = data['ROIslice']-1

    subwindow = [pmap_subwindow[0]-bounds[0], pmap_subwindow[1]-bounds[0], pmap_subwindow[2]-bounds[2], pmap_subwindow[3]-bounds[2]]
    print "subwindow:" + str(subwindow)

    # Save in data in order z,y,x
    out_dirs = []
    for p_i in range(len(pmap_names)):
        SI_i = 0
        out_vols = []
        outvolume = sample_frame
        print "Writing " + pmap_names[p_i]
        for slice_i in range(len(pmap_slices)):
            z_i = pmap_slices[slice_i]
            # Initialize slice intensity values
            pixel_array = np.array([[0]*ydim]*xdim)
            # Place data into slice subregion
            for y_i in range(subwindow[2], subwindow[3]+1):
                for x_i in range(subwindow[0], subwindow[1]+1):
                    pixel_array[y_i, x_i] = pmap_SIs[p_i][SI_i]
                    SI_i = SI_i + 1
            # Place data into slice
            outvolume[z_i].PixelData = pixel_array.astype(np.uint16).tostring()
            outvolume[z_i].Columns = xdim
            outvolume[z_i].Rows = ydim
            outvolume[z_i].NumberOfSlices = zdim
            outvolume[z_i].NumberOfTimeSlices = 1
        # Append volume to lists
        out_vols.append(outvolume)
        # Create output directory if it does not exist
        out_dir = outdir_basename + '_' + pmap_names[p_i]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        # Write output DICOM
        filenames = dcmio.WriteDICOM_frames(out_dir, out_vols, 'IM')
        out_dirs.append(out_dir)
    return out_dirs

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

    # Read subregion coordinates of non-corrected and corrected DICOMs
    subregion_in_originalDICOM = np.loadtxt(experiment_dir + '/' + args.subject + '/subregion.txt')
    # Write motioncorrected as single multifile DICOM subfolder
    ASCIIbase = experiment_dir + '/' + args.subject + '/' + args.subject + '_'
    DICOMbase = experiment_dir + '/' + out_prefix + '/'
    print "Converting non-corrected"
    try:
        ASCII2DICOM(ASCIIbase + 'Noncorrected_ASCII.txt', DICOMbase + 'Noncorrected', args.subject, subregion_in_originalDICOM)
    except Exception as inst:
        raise
    print "Converting motion corrected"
    try:
        ASCII2DIOM(ASCIIbase + 'Motioncorrected_ASCII.txt', DICOMbase + 'Motioncorrected', args.subject, subregion_in_originalDICOM)
    except Exception as inst:
        raise

    sys.exit(0)
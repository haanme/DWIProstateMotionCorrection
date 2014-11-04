#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
param_rigid = 'Par0001translation.txt'
param_BSpline = 'Par0001bspline08_mod.txt'
prostatemask_DICOM = r'/Users/eija/Desktop/prostate_MR/Carimasproject_files_Hb_outline_v1013/DICOMmasks'

#
# Splits subject ID into parts
#
# subjectid  - subject ID
#
def split_subjectid(subjectid):
    parts = subjectid.split('_')
    patient_no_str = parts[0]
    patientname_str = parts[1]
    bset_str = parts[2]
    rep_str = parts[3]
    return patient_no_str, patientname_str, bset_str, rep_str

#
# Resolves ROI that is a square-shaped bounding box around ROI pixels
#
# ROIpixel_array - 2-dimensional array
# padding        - number of empty pixels around ROI
#
def resolve_boundingbox(ROIpixel_array, padding):
    import numpy as np

    # Find minimum and maximum coordinates [xmin,xmax,ymin,ymax]
    bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]
    xlen = ROIpixel_array.shape[0]
    ylen = ROIpixel_array.shape[1]
    for xi in range(xlen):
        for yi in range(ylen):
            if ROIpixel_array[xi][yi] != 0:
                if xi < bounds[0]:
                    bounds[0] = xi
                if xi > bounds[1]:
                    bounds[1] = xi
                if yi < bounds[2]:
                    bounds[2] = yi
                if yi > bounds[3]:
                    bounds[3] = yi
    # Add padding
    bounds[0] = bounds[0] - padding
    bounds[1] = bounds[1] + padding
    bounds[2] = bounds[2] - padding
    bounds[3] = bounds[3] + padding
    if bounds[0] < 0:
        bounds[0] = 0
    if bounds[1] > xlen-1:
        bounds[1] = xlen-1
    if bounds[2] < 0:
        bounds[2] = 0
    if bounds[3] > ylen-1:
        bounds[3] = ylen-1
    # Create bounding box ROI
    outROI = np.zeros(ROIpixel_array.shape)
    for xi in range(bounds[0], bounds[1]+1):
        for yi in range(bounds[2], bounds[3]+1):
            outROI[xi][yi] = 1
    return outROI, bounds

#
# Resolve mat-filename containing ROI masks
#
# output_prefix - output prefix
#
def resolve_matfilename(output_prefix):
    # Resolve mat-file name
    parts = output_prefix.split('_')
    patient_no_str, patientname_str, bset_str, rep_str = split_subjectid(output_prefix)
    #    if not (bset_str=='hB' or bset_str=='lB'):
    if not (bset_str=='hB'):
        raise Exception((output_prefix + " UNSUPPORTED B-SET"))
    if (bset_str=='hB'):
        matfilename = mask_matfile_basedir_hB + '/' + patient_no_str + '_' + rep_str + '_DICOMconverted.mat'
    if (bset_str=='lB'):
        matfilename = mask_matfile_basedir_lB + '/' + patient_no_str + '_' + rep_str + '_DICOMconverted.mat'
    return matfilename

#
# Get mask image in DICOM from DICOM mask data
#
# input_shape   - input frame shape
# matfilename   - mat-file containing ROIs
# padding       - number of empty pixels around ROI
#
def get_boundsmask_from_DICOM(input_shape, ROIpixel_array, ROIname, padding):
    import scipy.io
    import os
    import numpy as np
    import copy

    shape = [input_shape[0], input_shape[1]]

    # Create mask around combined ROIs
    ROIpixel_array_combined = np.zeros([input_shape[0], input_shape[1]])
    for roi_i in range(len(ROIpixel_array)):
        for z in range(ROIpixel_array[roi_i].shape[2]):
            ROIpixel_array_combined = ROIpixel_array_combined + ROIpixel_array[roi_i][:,:,z]
    for xi in range(shape[0]):
        for yi in range(shape[1]):
            if ROIpixel_array_combined[xi][yi] != 0:
                ROIpixel_array_combined[xi][yi] = 1
    ROIpixel_array, bounds = resolve_boundingbox(ROIpixel_array_combined, padding)

    # Add z bounds to make [xmin,xmax,ymin,ymax,zmin,zmax]
    bounds.append(0)
    bounds.append(input_shape[2]-1)
    return bounds

#
# Get mask image in DICOM from mat-file data
#
# input_shape   - input frame shape
# matfilename   - mat-file containing ROIs
# ROIindexes    - ROI indexes that are used to create bounding mask
# padding       - number of empty pixels around ROI
#
def get_boundsmask_from_mat(input_shape, matfilename, ROIindexes, padding):
    import scipy.io
    import os
    import numpy as np
    import copy
    
    mat = scipy.io.loadmat(matfilename)
    # Get list of ROIs
    ROIs = mat['ROIs'].tolist()[0]
    # Get list of slices where ROIs are located
    ROIslices = mat['ROIslices'][0].tolist()
    # Create and write mask images
    print str(len(ROIs)) + " ROIs"
    shape = [input_shape[0], input_shape[1]]

    # Create mask around combined ROIs
    ROIpixel_array_combined = np.zeros(shape)
    for roi_i in range(len(ROIindexes)):
        ROIlist = ROIs[ROIindexes[roi_i]].tolist()
        ROIname = str(ROIlist[0][0][0][0])
        ROIpixel_array = ROIlist[0][0][1]
        print "catenating " + ROIname
        ROIpixel_array_combined = ROIpixel_array_combined + ROIpixel_array
    for xi in range(shape[0]):
        for yi in range(shape[1]):
            if ROIpixel_array_combined[xi][yi] != 0:
                ROIpixel_array_combined[xi][yi] = 1
    ROIpixel_array, bounds = resolve_boundingbox(ROIpixel_array_combined, padding)
    # Add z bounds to make [xmin,xmax,ymin,ymax,zmin,zmax]
    bounds.append(0)
    bounds.append(input_shape[2]-1)

    return bounds

#
# Get subvolumes
#
# dwidcm        - DICOM source data
# volume_list   - list of volume indexes for output
# bounds        - bounds of subvolumes
# output_prefix - output prefix
#
def get_subvolumes(input_plans, input_shape, volume_list, bounds, output_prefix):
    import dicom
    import DicomIO
    import shutil
    import numpy as np
    from nipype.utils.filemanip import split_filename

    dcmio = DicomIO.DicomIO()

    print "Original frame dimensions are:" + str(input_shape)

    # Resolve output directory and volumes
    out_dir_base = experiment_dir + os.sep + output_prefix + os.sep + 'subvolumes'
    filenames_all = []
    outdirs_all = []
    out_vols_all = []
    for vol_i in range(len(volume_list)):
        out_dir = out_dir_base + '_' + str(volume_list[vol_i])
        out_vols = []
        dwivolume = dwidcm[volume_list[vol_i]]
        # Take subregion from volume
        for slice_i in range(input_shape[2]):
            pixel_array = dwivolume[slice_i].pixel_array
            pixel_array = pixel_array[bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1]
            dwivolume[slice_i].PixelData = pixel_array.astype(np.uint16).tostring()
            dwivolume[slice_i].Columns = pixel_array.shape[1]
            dwivolume[slice_i].Rows = pixel_array.shape[0]
            dwivolume[slice_i].NumberOfSlices = input_shape[2]

        # Append volume to lists
        out_vols.append(dwivolume)
        out_vols_all.append(dwivolume)
        # Create output directory if it does not exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print "Writing to dir:" + out_dir
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
            print "Overwriting to dir:" + out_dir

        filenames = dcmio.WriteDICOM_frames(out_dir, out_vols, 'IM')
        filenames_all.append(filenames)
        outdirs_all.append(out_dir)

    # Resolve mask output name
    mask_out_dir = experiment_dir + os.sep + output_prefix + os.sep + 'bounds_mask'
    # Mask subregion from volume
    dwivolume = dwidcm[0]
    pixel_array = np.zeros([input_shape[0], input_shape[1]])
    pixel_array[bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1] = 1
    for slice_i in range(input_shape[2]):
        dwivolume[slice_i].PixelData = pixel_array.astype(np.uint16).tostring()
        dwivolume[slice_i].Columns = input_shape[0]
        dwivolume[slice_i].Rows = input_shape[1]
        dwivolume[slice_i].NumberOfSlices = input_shape[2]
        dwivolume[slice_i].NumberOfTimeSlices = 1
    # Create output directory if it does not exist
    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)
        print "Writing to dir:" + mask_out_dir
    else:
        shutil.rmtree(mask_out_dir)
        os.makedirs(mask_out_dir)
        print "Overwriting to dir:" + mask_out_dir
    mask_filenames = dcmio.WriteDICOM_frames(mask_out_dir, [dwivolume], 'IM')


    return outdirs_all, filenames_all, out_vols_all, mask_filenames, mask_out_dir

from argparse import ArgumentParser
import sys
import os
import DicomIO
import conversions as conv
import time
import numpy as np
import glob
import ROIutils

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dicomdir", dest="dicomdir", help="dicomdir", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    args = parser.parse_args()

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists((experiment_dir + os.sep + args.subject)):
        os.makedirs((experiment_dir + os.sep + args.subject))

    dcmio = DicomIO.DicomIO()
    print "READING DWI DICOM [" + args.dicomdir + "]"
    try:
        dwidcm = dcmio.ReadDICOM_frames(args.dicomdir)
        dwishape = [dwidcm[0][0].pixel_array.shape[0], dwidcm[0][0].pixel_array.shape[1], len(dwidcm[0])]
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    print "LOOKING FOR ROIS IN " + (prostatemask_DICOM + os.sep + args.subject + '*')
    ROIfolders = glob.glob(prostatemask_DICOM + os.sep + args.subject + '*')
    bounds_all = [1000, 0, 1000, 0, 1000, 0]
    if len(ROIfolders) == 0:
        print "NO ROIS WERE FOUND"
        sys.exit(1)
    for ROIfolder in ROIfolders:

        # Read DICOM mask data from all files matching subject
        print "READING ROI FROM DICOM [" + ROIfolder + "]"
        ROIpixel_array_all, ROInames = ROIutils.resolve_DICOMROI_imgdata(ROIfolder)

        # Resolve bounds
        print "RESOLVING BOUNDS"
        bounds = get_boundsmask_from_DICOM(dwishape, ROIpixel_array_all, ROInames, 10)
        if bounds[0] < bounds_all[0]:
            bounds_all[0] = bounds[0]
        if bounds[1] > bounds_all[1]:
            bounds_all[1] = bounds[1]
        if bounds[2] < bounds_all[2]:
            bounds_all[2] = bounds[2]
        if bounds[3] > bounds_all[3]:
            bounds_all[3] = bounds[3]
        if bounds[4] < bounds_all[4]:
            bounds_all[4] = bounds[4]
        if bounds[5] > bounds_all[5]:
            bounds_all[5] = bounds[5]

        print bounds
    print bounds_all

    # Extract first volume from dwi
    print "RESOLVING AND WRITING SUBVOLUMES"
    try:
        subvol_dirs, filenames_all, subvols_orig, mask_filenames, mask_out_dir = get_subvolumes(dwidcm, dwishape, range(len(dwidcm)), bounds_all, args.subject)
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    # Write non-corrected as single multifile DICOM subfolder for motion correction purposes
    print "COMBINING ORIGINAL SUB-WINDOWED IMAGES INTO ONE DICOM"
    try:
        conv.multidicom2multidicom(subvol_dirs, 'Noncorrected10', experiment_dir, args.subject)
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    # Save the used bound coordinates to txt file
    np.savetxt((experiment_dir + '/' + args.subject + '/' + 'subregion10.txt'), bounds, fmt='%f', header=('subject ' + args.subject))

    sys.exit(0)

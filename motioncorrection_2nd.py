#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
param_rigid = 'Par0001translation.txt'
param_BSpline = 'Par0001bspline08_mod.txt'
#mask_matfile_basedir = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS/ROI_mat_files'
mask_matfile_basedir_hB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS_for_Motion_Cor/ROI_mat_files'
mask_matfile_basedir_lB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Lb_work_2rep/ROI_mat_files'

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
# Get mask image in DICOM from mat-file data
#
# output_prefix - output prefix
# input_shape   - input frame shape
# input_plans   - DICOM sample slices
# matfilename   - mat-file containing ROIs
# ROIindexes    - ROI indexes that are used to create bounding mask
# padding       - number of empty pixels around ROI
#
def get_boundsmask(output_prefix, input_shape, input_plans, matfilename, ROIindexes, padding):
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

    ROI_filenames = []
    dcmio = DicomIO.DicomIO()
    # Resolve ROI data
    ROIlist = ROIs[roi_i].tolist()
    ROIname = "Boundingbox"
    print ROIname
    #print ROIpixel_array
    # Resolve output name
    out_dir = experiment_dir + '/' + output_prefix + '/' + 'ROImask' + str(roi_i+1) + '_' + ROIname
    # Place mask into intensity values
    output_frame = []
    #print str(len(input_frame[0])) + " slices of size " + str(shape)
    for slice_i in range(input_shape[2]):
        slice = copy.deepcopy(input_plans[slice_i])
        if slice_i != ROIslices[0]:
            #print "zero-slice:" + str(slice_i) + " " + str(shape)
            slice.PixelData = np.zeros(shape).astype(np.uint16).tostring()
        else:
            #print " ROI-slice:" + str(slice_i) + " " + str(ROIpixel_array.shape)
            slice.PixelData = ROIpixel_array.astype(np.uint16).tostring()
        output_frame.append(slice)
    # Create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Write data
    filenames = dcmio.WriteDICOM_frames(out_dir, [output_frame], 'IM')
    ROI_filenames.append(filenames[ROIslices[0]])

    return out_dir, ROI_filenames, ROIslices[0], bounds

#
# Run elastix
#
# input_file        - file that is co-registered
# nonmoved_file     - file where co-registration is targeted
# mask_file         - mask for GoF
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def elastix(input_file, target_file, mask_file, output_prefix, output_sub_prefix):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    import shutil
    import glob
    import os
    out_dir = experiment_dir + '/' + output_prefix + '/' + output_sub_prefix
    # Create output directory if it does not exist
    if os.path.exists(out_dir):
        print "rmtree: " + out_dir
        shutil.rmtree(out_dir)
    print "creating: " + out_dir
    os.makedirs(out_dir)

    cmd = CommandLine(('/Users/eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin/elastix -f %s -m %s -out %s -p %s -p %s -threads 8') % (target_file, input_file, out_dir, param_rigid, param_BSpline))
#   cmd = CommandLine(('/Users/eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin/elastix -f %s -m %s -out %s -p %s -threads 8') % (target_file, input_file, out_dir, param_rigid))

    print "elastix: " + cmd.cmd
    cmd.run()
    resultfiles = glob.glob(out_dir + os.sep + 'result.*.tiff')
    return resultfiles

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
# Convert multi-slice tiff 2 DICOM
#
# in_files   - single TIFF input file (.tiff) for each frame
# dicomdir   - output DICOM directory
# plans      - DICOM header templates for output, frames X slices
# out_prefix - subject specific prefix
#
def singletiff2multidicom(in_files, dicomdir, plans, out_prefix):
    import DicomIO
    import numpy as np
    import os
    import shutil
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile as tiff
    outdir = experiment_dir + '/' + out_prefix + '/' + dicomdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    # Resolve new frame list
    out_vols = plans
    for file_i in range(len(in_files)):
        print "Reading " + in_files[file_i]
        ds = tiff.imread(in_files[file_i])
        no_slices = ds.shape[0]
        for z_i in range(no_slices):
            out_vols[file_i][z_i].PixelData = ds[z_i].astype(np.uint16).tostring()

    dcmio = DicomIO.DicomIO()
    filenames = dcmio.WriteDICOM_frames(outdir, out_vols, 'IM')

    return outdir, filenames

#
# Convert single-slice DICOM 2 DICOM
#
# in_dirs    - single DICOM input directory for each frame
# dicomdir   - output DICOM directory
# plans      - DICOM header templates for output, frames X slices
# out_prefix - subject specific prefix
#
def multidicom2multidicom(in_dirs, dicomdir, plans, out_prefix):
    import dicom
    import DicomIO
    import numpy as np
    import os
    import shutil

    outdir = experiment_dir + '/' + out_prefix + '/' + dicomdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    # Resolve new frame list
    out_vols = plans
    dcmio = DicomIO.DicomIO()
    for dir_i in range(len(in_dirs)):
        print "Reading " + in_dirs[dir_i]
        frame_list = dcmio.ReadDICOM_frames(in_dirs[dir_i])
        no_slices = len(frame_list[0])
        for z_i in range(no_slices):
            out_vols[dir_i][z_i].PixelData = frame_list[0][z_i].pixel_array.astype(np.uint16).tostring()

    dcmio = DicomIO.DicomIO()
    filenames = dcmio.WriteDICOM_frames(outdir, out_vols, 'IM')

    return outdir, filenames

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
    if not os.path.exists((experiment_dir + '/' + args.subject)):
        os.makedirs((experiment_dir + '/' + args.subject))

    if os.path.exists((experiment_dir + '/' + args.subject + '/' + 'elapsed_time.txt')):
        print "SKIPPING"
        sys.exit(0)
    else:
        print "EXECUTING"

    matfilename = resolve_matfilename(args.subject)
    if not os.path.exists(matfilename):
        print (matfilename + " DOES NOT EXIST")
        sys.exit(1)

    dcmio = DicomIO.DicomIO()
    start_time = time.time()
    print "READING DICOM [" + args.dicomdir + "]"
    try:
        dwidcm = dcmio.ReadDICOM_frames(args.dicomdir)
        dwishape = [dwidcm[0][0].pixel_array.shape[0], dwidcm[0][0].pixel_array.shape[1], len(dwidcm[0])]
    except:
        errors = errors + 1
        sys.exit(1)

    print "RESOLVING BOUNDS"
        #    try:
    mask_file, mask_file_ROIslice_filename, ROIslice_i, bounds = get_boundsmask(args.subject, dwishape, dwidcm[0], matfilename, [0], 20)
    np.savetxt((experiment_dir + '/' + args.subject + '/' + 'subregion.txt'),bounds, fmt='%f', header=('subject ' + args.subject))
        #except Exception as inst:
        #errors = errors + 1
        #print type(inst)     # the exception instance
        #print inst.args      # arguments stored in .args
        #print inst           # __str__ allows args to be printed directly
#sys.exit(1)

    # Extract first volume from dwi
    print "RESOLVING SUBVOLUMES"
    try:
        subvol_dirs, subvol_filenames_all, subvols_orig = get_subvolumes(dwidcm, range(len(dwidcm)), bounds, args.subject)
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    # Write motioncorrected as single multifile DICOM subfolder
    print "COMBINING ORIGINAL SUB-WINDOWED"
    try:
        multidicom2multidicom(subvol_dirs, 'Noncorrected', subvols_orig, args.subject)
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    # Run elastix for all frames, in mhd format, results in single-file DICOM
    print "DEFORMATION"
    try:
        subvol_0_file,rawfile,txtfile = conv.dicom2mhd(subvol_dirs[0], args.subject)
    except:
        errors = errors + 1
        sys.exit(1)
    result_frames_tiff = []
    for subvol_i in range(0,len(subvols_orig)):
        #    for subvol_i in range(0,2):
        try:
            subvols_i_file,rawfile,txtfile = conv.dicom2mhd(subvol_dirs[subvol_i], args.subject)
            print subvols_i_file + " > " + subvol_0_file
            mc_frame_files = elastix(subvols_i_file, subvol_0_file, mask_file_ROIslice_filename[0], args.subject, ('Motioncorrected_' + str(subvol_i) + '_to_' + str(0)))
            result_frames_tiff.append(mc_frame_files[-1])
        except Exception as inst:
            errors = errors + 1
            print type(inst)     # the exception instance
            print inst.args      # arguments stored in .args
            print inst           # __str__ allows args to be printed directly
            sys.exit(1)

    # Write motioncorrected as single multifile DICOM subfolder
    print "COMBINING RESULTS"
    print result_frames_tiff
    try:
        singletiff2multidicom(result_frames_tiff, 'Motioncorrected', subvols_orig, args.subject)
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    # Write elapsed time for the full process
    elapsed_time = time.time() - start_time
    print "Total elapsed time for process " + str(elapsed_time) + " seconds"
    np.savetxt((experiment_dir + '/' + args.subject + '/' + 'elapsed_time.txt'),[elapsed_time], fmt='%f seconds', header=('subject ' + args.subject), footer=(str(errors) + " errors"))

    sys.exit(0)

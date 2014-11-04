#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
param_rigid = 'Par0001rigid_T2toDWI.txt'
T2_DICOMpath_basedir = '/Users/eija/Desktop/prostate_MR/T2W_TSE_2.5m'
mask_DICOMpath_basedir = '/Users/eija/Desktop/prostate_MR/T2W_Carimas_project_files/DICOMmasks'

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
# Resolve DICOM containing ROI masks
#
# output_prefix - output prefix
#
def resolve_matfilename(output_prefix):
    import glob

    # Resolve mat-file name
    parts = output_prefix.split('_')
    patient_no_str, patientname_str, bset_str, rep_str = split_subjectid(output_prefix)
    paths = glob.glob('')

    return matfilename

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

    cmd = CommandLine(('/Users/eija/Desktop/SW/Elastix/elastix_macosx64_v4.7/bin/elastix -f %s -m %s -out %s -p %s -threads 8') % (target_file, input_file, out_dir, param_rigid))

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


    # Run elastix for all frames, in mhd format, results in single-file DICOM
    print "CO-REGISTRATION"
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

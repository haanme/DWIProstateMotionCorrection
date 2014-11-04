#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
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
# Run elastix
#
# input_file        - file that is co-registered
# nonmoved_file     - file where co-registration is targeted
# mask_file         - mask for GoF
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def elastix(input_file, target_file, mask_file, output_prefix, output_sub_prefix, param_rigid, param_BSpline):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    import shutil
    import glob
    import os
    out_dir = experiment_dir + os.sep + output_prefix + os.sep + output_sub_prefix
    # Create output directory if it does not exist
    if os.path.exists(out_dir):
        print "rmtree: " + out_dir
        shutil.rmtree(out_dir)
    print "creating: " + out_dir
    os.makedirs(out_dir)

    cmd = CommandLine(('/Users/eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin/elastix -f %s -m %s -out %s -p %s -p %s -threads 8') % (target_file, input_file, out_dir, param_rigid, param_BSpline))

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

from argparse import ArgumentParser
import sys
import os
import DicomIO
import conversions as conv
import time
import numpy as np
import glob

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dicomdir", dest="dicomdir", help="dicomdir", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    parser.add_argument("--coreg_rigid", dest="param_rigid", help="Translation parameters file", required=True)
    parser.add_argument("--coreg_nonrigid", dest="param_BSpline", help="Deformation parameters file", required=True)
    parser.add_argument("--orig_subdir", dest="orig_subdir", help="Subdirectory name for original non-corrected data", required=True)
    args = parser.parse_args()

    param_rigid = args.param_rigid
    param_BSpline = args.param_BSpline

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists((experiment_dir + os.sep + args.subject)):
        os.makedirs((experiment_dir + os.sep + args.subject))

    if os.path.exists((experiment_dir + os.sep + args.subject + os.sep + 'elapsed_time.txt')):
        print "OVERWRITING"
    else:
        print "EXECUTING"

    subvol_dirs = glob.glob((experiment_dir + os.sep + args.subject + os.sep + 'subvolumes_*'))
    subvol_dirs2 = []
    for subvol_dir in subvol_dirs:
        if os.path.isdir(subvol_dir):
            subvol_dirs2.append(subvol_dir)
    subvol_dirs = subvol_dirs2

    # Run elastix for all frames, in mhd format, results in single-file DICOM
    start_time = time.time()
    print "DEFORMATION"
    try:
        subvol_0_file,rawfile,txtfile = conv.dicom2mhd(subvol_dirs[0], experiment_dir, args.subject)
    except:
        errors = errors + 1
        sys.exit(1)
    result_frames_tiff = []
    for subvol_i in range(0,len(subvol_dirs)):
        #    for subvol_i in range(0,2):
        try:
            subvols_i_file,rawfile,txtfile = conv.dicom2mhd(subvol_dirs[subvol_i], experiment_dir, args.subject)
            print subvols_i_file + " > " + subvol_0_file
            out_dir = ('Motioncorrected_' + str(subvol_i) + '_to_' + str(0))
            mc_frame_files = elastix(subvols_i_file, subvol_0_file, None, args.subject, out_dir, param_rigid, param_BSpline)
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
        dcmio = DicomIO.DicomIO()
        "Original DICOM read from " + (experiment_dir + os.sep + args.subject + os.sep + args.orig_subdir)
        subvols_orig = dcmio.ReadDICOM_frames((experiment_dir + os.sep + args.subject + os.sep + args.orig_subdir))
        conv.singletiff2multidicom(result_frames_tiff, 'Motioncorrected', subvols_orig, experiment_dir, args.subject)
    except Exception as inst:
        errors = errors + 1
        print type(inst)     # the exception instance
        print inst.args      # arguments stored in .args
        print inst           # __str__ allows args to be printed directly
        sys.exit(1)

    # Write elapsed time for the full process
    elapsed_time = time.time() - start_time
    print "Total elapsed time for process " + str(elapsed_time) + " seconds"
    np.savetxt((experiment_dir + os.sep + args.subject + os.sep + 'elapsed_time.txt'),[elapsed_time], fmt='%f seconds', header=('subject ' + args.subject), footer=(str(errors) + " errors"))

    sys.exit(0)

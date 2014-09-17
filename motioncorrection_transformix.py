#!/usr/bin/env python

elastix_dir = '/Users/eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin/'
experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
param_rigid = 'Par0001translation.txt'
param_BSpline = 'Par0001bspline08.txt'
#mask_matfile_basedir = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS/ROI_mat_files'
mask_matfile_basedir_hB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS_for_Motion_Cor/ROI_mat_files'
mask_matfile_basedir_lB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Lb_work_2rep/ROI_mat_files'

#
# Create coordinates file
#
# input_file        - elastix co-registration parameters file
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def create_input_points(input_file, output_prefix, output_sub_prefix):
    # Read image size from parameters file
    # Resolve name of output file
    # Write point set to output file
    # Return name of output file, and created points

#
# Create 2D grid for QC purposes
#
# shape             - shape of 2D grid
#
def create_2D_grid(shape):

    Xs = []
    Ys = []
    # Go through all horizontal lines
    for xi in range(shape[0]):
        for yi in range(shape[1]):
            Xs.append(xi)
            Ys.append(yi)
    # Go through all vertical lines
    for yi in range(shape[1]):
        for xi in range(shape[0]):
            Xs.append(xi)
            Ys.append(yi)
    return Xs, Ys
#
# Create coordinates file
#
# input_file        - elastix transformation tiff file with RGB channels for x,y,z directions
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def plot_deformation_vectors(input_file):
    import matpltolib.pyplot as plt
    import skimage.io

    im = skimage.io.imread('deformationField.tiff', plugin='tifffile')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create quiver plot for each slice
    plt.quiver()

    ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])
    # Create grid
    x = np.arange(0, shape[0], 0.5)
    y = np.arange(0, shape[1], 0.5)
>>> xx, yy = meshgrid(x, y, sparse=True)

#
# Run transformix
#
# input_file        - elastix co-registration parameters file
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def elastix_AnalyzeWarp(input_file, output_prefix, output_sub_prefix):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    out_dir = experiment_dir + '/' + output_prefix + '/' + output_sub_prefix + '/QC'
    # Create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cmd = CommandLine((elastix_dir + 'transformix -def all -jac all -jacmat all -out %s -tp %s') % (out_dir, input_file))

    print "transformix: " + cmd.cmd
    cmd.run()
    return (out_dir + '/' + 'deformationField.tiff'), (out_dir + '/' + 'spatialJacobian.tiff'), (out_dir + '/' + 'fullSpatialJacobian.tiff'), (out_dir + '/' + 'transformix.log')

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

    subdirs = os.listdir(experiment_dir + '/' + args.subject)
    subdirs_for_QC = []
    for subdir_i in range(len(subdirs)):
        if os.path.isdir(experiment_dir + '/' + args.subject + '/' + subdirs[subdir_i]) and (subdirs[subdir_i].find('Motioncorrected_') != -1):
            subdirs_for_QC.append(subdirs[subdir_i])

    for subdir_i in range(len(subdirs_for_QC)):
        parameter_file = experiment_dir + '/' + args.subject + '/' + subdirs_for_QC[subdir_i] + '/' + 'TransformParameters.1.txt'
        print "Creating deformation field images non-corrected [" + subdirs_for_QC[subdir_i] + "]"
        try:
            disp_field, jacdet_map, jacmat_map, logfile = elastix_AnalyzeWarp(parameter_file, args.subject, subdirs_for_QC[subdir_i])
        except Exception as inst:
            raise
        print disp_field
        print jacdet_map
        print jacmat_map
        print logfile
        break

    sys.exit(0)

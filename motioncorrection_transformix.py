#!/usr/bin/env python

elastix_dir = '/Users/eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin/'
experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
mask_matfile_basedir_hB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS_for_Motion_Cor/ROI_mat_files'
mask_matfile_basedir_lB = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Lb_work_2rep/ROI_mat_files'

#
# Write input points for transformix
#
# X - x-coordinates
# Y - y-coordinates
# Z - z-coordinates
#
def write_inputPoints_txt(X, Y, Z):
    outfilename = 'transformix_inputPoints.txt'
    f = open(outfilename,'w')
    f.write('point\n')
    f.write(str(X.shape[0]*X.shape[1]) + '\n')
    ydim = X.shape[0]
    xdim = X.shape[1]
    print xdim
    print ydim
    for i in range(ydim):
        for j in range(xdim):
            f.write(str(X[i,j]) + ' ' + str(Y[i,j]) + ' ' + str(Z[i,j]) + '\n')
    f.close()
    return outfilename

#
# Read output points from transformix
#
# filname           - ASCII fiel containing transformix output
#
def read_outputPoints_txt(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    X = []
    Y = []
    Z = []
    for line_i in range(len(lines)):
        splitted = lines[line_i].split()
        if len(splitted) > 31:
            X.append(float(splitted[30]))
            Y.append(float(splitted[31]))
            Z.append(float(splitted[32]))
    f.close()
    return X, Y, Z

#
# Run transformix
#
# input_file        - elastix co-registration parameters file
# inputpoints_file  - input points ASCII file
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def elastix_PointsWarp(input_file, inputpoints_file, output_prefix, output_sub_prefix):
    import os
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    #    out_dir = experiment_dir + '/' + output_prefix + '/' + output_sub_prefix + '/QC'
    out_dir = '.'
    # Create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cmd = CommandLine((elastix_dir + 'transformix -out %s -def %s -tp %s') % (out_dir, inputpoints_file, input_file))
    print "transformix: " + cmd.cmd
    cmd.run()
    return (out_dir + os.sep + 'outputpoints.txt')

#
# Run transformix
#
# input_file        - elastix co-registration parameters file
# output_prefix     - output prefix
# output_sub_prefix - output subfolder prefix
#
def elastix_AnalyzeWarp(input_file, output_prefix, output_sub_prefix):
    import os
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
    return (out_dir + os.sep + 'deformationField.tiff'), (out_dir + os.sep + 'spatialJacobian.tiff'), (out_dir + os.sep + 'fullSpatialJacobian.tiff'), (out_dir + os.sep + 'transformix.log')

#
# Run transformix
#
# input_file        - elastix co-registration parameters file
#
def read_log(input_file):
    import numpy as np

    f = open(input_file)
    lines = f.readlines()
    f.close()
    read_state = 0
    resolutions = []
    iterations = []
    filename = '<unknown file>'
    metric = '<unknown metric>'
    for line_i in range(len(lines)):
        line = lines[line_i]
        if line.startswith('Running elastix with parameter file'):
            filename = line[len('Running elastix with parameter file'):]
        if line.startswith('(Metric'):
            splitted_line = line.split('"')
            metric = splitted_line[1]
        if read_state == 0 and line.startswith('1:ItNr'):
            read_state = 1
            resolution_names = []
            splitted_line = line.split()
            for splitted_i in range(len(splitted_line)-1):
                splitted_name = splitted_line[splitted_i].split(':')
                resolution_names.append(splitted_name[1])
            continue
        if read_state == 1 and line.startswith('Time spent'):
            read_state = 0
            resolutions.append({'data':np.array(iterations), 'names':resolution_names, 'endcondition':lines[line_i+1], 'filename':filename, 'metric':metric})
            iterations = []
            continue
        if read_state == 1:
            iterations.append(line.split())
#return resolutions[0:-2]
    return resolutions

#
# Plot data in ASCII file
#
# filename        - original filename
# data_raw_orig   - raw original data
# data_raw_corr   - raw corrected data
#
def plot_data(basedir, filename, data_raw_orig, data_raw_corr):
    import matplotlib.pyplot as plt
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    import numpy as np
    from matplotlib import cm
    from matplotlib.backends.backend_pdf import PdfPages
    import plot_utils

    # Open pdf
    slice_1st = data_raw_orig[0][0]
    xdim = slice_1st.Columns
    ydim = slice_1st.Rows
    zdim = slice_1st.NumberOfSlices
    tdim = slice_1st.NumberOfTimeSlices
    dim = [xdim, ydim, zdim, tdim]
    print slice_1st.pixel_array.shape
    print dim
    #pw = int(np.ceil(np.sqrt(dim[2])))
    pw = 2

    bset = []
    for frame_i in range(len(data_raw_orig)):
        bset.append(int(data_raw_orig[frame_i][0].FrameReferenceTime)/1000)
    pdfnames = []
    for t_i in range(tdim):
        pdfname = (filename + '_b' + ('%d' % bset[t_i]) + '.pdf')
        pdf = PdfPages(basedir + os.sep + pdfname)
        frame_orig = np.ndarray([ydim, xdim, zdim])
        frame_corr = np.ndarray([ydim, xdim, zdim])
        for z_i in range(zdim):
            frame_orig[:,:,z_i] = np.add(np.multiply(data_raw_orig[t_i][z_i].pixel_array, data_raw_orig[t_i][z_i].RescaleSlope), data_raw_orig[t_i][z_i].RescaleIntercept)
            frame_corr[:,:,z_i] = np.add(np.multiply(data_raw_corr[t_i][z_i].pixel_array, data_raw_corr[t_i][z_i].RescaleSlope), data_raw_corr[t_i][z_i].RescaleIntercept)
        vmin = 0
        vmax = np.amax(frame_orig)

        # Plot SI
        fig = plot_utils.plot_create_fig('Signal Intensity b' + ('%d' % bset[t_i]) + ' before and after')
        fig.subplots_adjust(left=0.01)
        fig.subplots_adjust(right=0.95)
        fig.subplots_adjust(wspace=0.05)
        plot_utils.plot_add_img_pairs(fig, plt, frame_orig[:,:,0:4], frame_corr[:,:,0:4], [0, 0, 4], range(0,4), pw, cm.gist_gray, vmin, vmax)
        pdf.savefig(fig)

        fig = plot_utils.plot_create_fig('Signal Intensity b' + ('%d' % bset[t_i]) + ' before and after')
        fig.subplots_adjust(left=0.01)
        fig.subplots_adjust(right=0.95)
        fig.subplots_adjust(wspace=0.05)
        plot_utils.plot_add_img_pairs(fig, plt, frame_orig[:,:,4:8], frame_corr[:,:,4:8], [0, 0, 4], range(4,8), pw, cm.gist_gray, vmin, vmax)
        pdf.savefig(fig)

        fig = plot_utils.plot_create_fig('Signal Intensity b' + ('%d' % bset[t_i]) + ' before and after')
        fig.subplots_adjust(left=0.01)
        fig.subplots_adjust(right=0.95)
        fig.subplots_adjust(wspace=0.05)
        plot_utils.plot_add_img_pairs(fig, plt, frame_orig[:,:,8:12], frame_corr[:,:,8:12], [0, 0, 4], range(8,12), pw, cm.gist_gray, vmin, vmax)
        pdf.savefig(fig)

        plt.close()
        pdf.close()
        pdfnames.append(pdfname)
    return pdfnames

from argparse import ArgumentParser
import sys
import os
import DicomIO
import conversions as conv
import time
import numpy as np
import glob
import shutil
import plot_utils

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dicomdir", dest="dicomdir", help="dicomdir", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    parser.add_argument("--outdir", dest="outdir", help="output directory for results", required=True)
    parser.add_argument("--orig_subdir", dest="orig_subdir", help="Subdirectory name for original non-corrected data", required=True)
    parser.add_argument("--subregionfilename", dest="subregionfilename", help="Subregion filename in working directory", required=True)
    args = parser.parse_args()
    subdirs = os.listdir(experiment_dir + os.sep + args.subject)
    subdirs_for_QC = []
    for subdir_i in range(len(subdirs)):
        if os.path.isdir(experiment_dir + os.sep + args.subject + os.sep + subdirs[subdir_i]) and (subdirs[subdir_i].find('Motioncorrected_') != -1):
            subdirs_for_QC.append(subdirs[subdir_i])
    B0_images = []
    parameter_files = []
    disp_fields = []

    print "Loading deformed DICOM image"
    dcmio = DicomIO.DicomIO()
    dwidcm = dcmio.ReadDICOM_frames(experiment_dir + os.sep + args.subject + os.sep + 'Motioncorrected')
    dwishape = [dwidcm[0][0].pixel_array.shape[0], dwidcm[0][0].pixel_array.shape[1], len(dwidcm[0]), len(dwidcm)]

    for subdir_i in range(len(subdirs_for_QC)):
        subdir = experiment_dir + os.sep + args.subject + os.sep + subdirs_for_QC[subdir_i]
        print str(subdir_i) + ':' + subdir
        loginfo = read_log(subdir + os.sep + 'elastix.log')
        plot_utils.plot_iterationinfo(loginfo, subdir)

        B0_images.append(dwidcm[subdir_i])
        TransformParameters_paths = glob.glob((experiment_dir + os.sep + args.subject + os.sep + subdirs_for_QC[subdir_i] + os.sep + 'TransformParameters.*.txt'))
        parameter_file = TransformParameters_paths[-1]
        parameter_files.append(parameter_file)
        print "Creating deformation field images non-corrected [" + subdirs_for_QC[subdir_i] + "]"
        try:
            disp_field, jacdet_map, jacmat_map, logfile = elastix_AnalyzeWarp(parameter_file, args.subject, subdirs_for_QC[subdir_i])
        except Exception as inst:
            raise
        disp_fields.append(disp_field)
    print "Creating QC pdf"
    plot_utils.plot_deformation_vectors(disp_fields, parameter_files, args.subject, subdirs_for_QC, B0_images, experiment_dir)

    dwi_orig = dcmio.ReadDICOM_frames(experiment_dir + os.sep + args.subject + os.sep + args.orig_subdir)
    pdfnames = plot_data(experiment_dir + os.sep + args.subject, 'deformation_QC_sidebyside', dwi_orig, dwidcm)

    out_dir = args.outdir + os.sep + args.subject
    print "Copying result data to " + out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(out_dir + os.sep + 'Motioncorrected'):
        shutil.rmtree(out_dir + os.sep + 'Motioncorrected')
    shutil.copytree(experiment_dir + os.sep + args.subject + os.sep + 'Motioncorrected', out_dir + os.sep + 'Motioncorrected')

    frame_paths = glob.glob(experiment_dir + os.sep + args.subject + os.sep + 'Motioncorrected_*')
    for frame_path_i in range(len(frame_paths)):
        frame_path = frame_paths[frame_path_i]
        shutil.copyfile(frame_path + os.sep + 'convergence_QC.pdf', out_dir + os.sep + 'convergence_' + str(frame_path_i+1) + '.pdf')
    shutil.copyfile(experiment_dir + os.sep + args.subject + os.sep + 'elapsed_time.txt', out_dir + os.sep + 'elapsed_time.txt')
    shutil.copyfile(experiment_dir + os.sep + args.subject + os.sep + args.subregionfilename, out_dir + os.sep + args.subregionfilename)
    shutil.copyfile(experiment_dir + os.sep + args.subject + os.sep + 'deformation_QC.pdf', out_dir + os.sep + 'deformation_QC.pdf')
    for pdfname in pdfnames:
        shutil.copyfile(experiment_dir + os.sep + args.subject + os.sep + pdfname, out_dir + os.sep + pdfname)

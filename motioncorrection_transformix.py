#!/usr/bin/env python

elastix_dir = '/Users/eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin/'
experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'
#mask_matfile_basedir = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis/Carimas27projectfiles_Hb_work_all_noGS/ROI_mat_files'
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
# Create coordinates file
#
# input_files         - elastix transformation tiff file with RGB channels for x,y,z directions
# parameter_files     - elastix co-registration parameters file
# output_prefix       - output prefix
# output_sub_prefixes - output subfolder prefix
# DWIimgs             - DWIimage for overlay purposes
#
def plot_deformation_vectors(input_files, parameter_files, output_prefix, output_sub_prefixes, DWIimgs):
    import matplotlib.pyplot as plt
    import skimage.io
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = experiment_dir + '/' + output_prefix
    im = skimage.io.imread(input_files[0], plugin='tifffile')
    shape = im.shape
    no_slices = shape[0]
    used_slices = range(shape[0])
    no_slices_per_page = 3
    pdf = PdfPages(out_dir + os.sep + 'deformation_QC.pdf')
    print shape

    # Go through slices
    for used_slice_i in range(len(used_slices)):
        slice_i = int(used_slices[used_slice_i])
        # Go through b-values
        for b_value in range(len(input_files)):
            print str(slice_i) + ' frame ' + str(b_value)
            # Start writing data for new page
            if np.mod(b_value, 3) == 0:
                fig = plt.figure()
                fig.subplots_adjust(hspace=.01)
                fig.subplots_adjust(wspace=.1)
                fig.subplots_adjust(left=0.05)
                fig.subplots_adjust(right=0.95)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(bottom=0.01)
                plot_i = 0
                slice_str = ''
            im = skimage.io.imread(input_files[b_value], plugin='tifffile')
            imslice = im[slice_i,:,:,:]
            plot_i = plot_slice(fig, imslice, shape, plot_i, no_slices_per_page, DWIimgs[b_value][slice_i].pixel_array)
            slice_str = slice_str + ' ' + ('%.0f' % b_value)
            # Save data to figure
            if np.mod(b_value, 3) == 2:
                fig.suptitle('[' + output_prefix + '] Deformation QC for slice ' + str(slice_i) + ' frames ' + slice_str, fontsize=12)
                pdf.savefig(fig)
                #plt.show()
                plt.close()
        # Save data to figure if last round did not involve saving
        if np.mod(shape[3]-1, 3) != 2:
            fig.suptitle('[' + output_prefix + '] Deformation QC for slice ' + str(slice_i) + ' frames ' + slice_str, fontsize=12)
            pdf.savefig(fig)
            #plt.show()
            plt.close()
    pdf.close()

#
# Plot displacement slice for QC purposes
#
# fig          - current figure where QC is plotted
# slice_img    - slice image containing displacement values
# no_slices    - total number of slices to be QC'd
# plot_i       - current plot index in QC subplots
# title        - title printed as y-axis label
# Dvals        - displacement values for printing statistics on x-axis label
#
def plot_displacement_slice(fig, slice_img, no_slices, plot_i, title, Dvals):
    import matplotlib.pyplot as plt

    ax = fig.add_subplot(no_slices, 5, plot_i)
    surf = ax.imshow(slice_img)
    surf.set_clim(-1.0,1.0)
    plt.ylabel(title, fontsize=5, labelpad=-13.0)
    plt.xlabel('$\mu=$' + ('%.3f' % np.mean(Dvals)) + '\n$\sigma=$' + ('%.3f' % np.std(Dvals)) + '\n$max=$' + ('%.3f' % np.ma.max(Dvals)), fontsize=4, labelpad=0)
    cb = fig.colorbar(surf, shrink=0.65, aspect=10, orientation='vertical', pad=0.025)
    ax.set_xticklabels([])
    [line.set_markersize(3) for line in ax.xaxis.get_ticklines()]
    ax.set_yticklabels([])
    [line.set_markersize(3) for line in ax.yaxis.get_ticklines()]
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(4)

#
# Plot QC of one slice
#
# fig         - current figure where QC in plotted
# imslice     - slice image containing displacement values in x,y,z directions
# shape       - shape of original displacement field image z,x,y,d, where d is displacement [u,v,w]
# plot_i      - current plot index in QC subplots
# no_slices   - total number of slices that are going to be QC'd
# dwislice    - DWI slice for overlay purposes
#
def plot_slice(fig, imslice, shape, plot_i, no_slices, dwislice):
    import scipy.ndimage
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import warnings

    ax = fig.add_subplot(no_slices, 5, plot_i+1)
    X = np.linspace(0, shape[2], shape[2])
    Y = np.linspace(0, shape[1], shape[1])
    X, Y = np.meshgrid(X, Y)
    zfactor = 0.15

    # Catch user warning about changing result size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = scipy.ndimage.zoom(X, zfactor, order=0)
        Y = scipy.ndimage.zoom(Y, zfactor, order=0)
        imsliceRu = scipy.ndimage.zoom(imslice[:,:,0], zfactor, order=2)
        imsliceRv = scipy.ndimage.zoom(imslice[:,:,1], zfactor, order=2)
        imsliceRw = scipy.ndimage.zoom(imslice[:,:,2], zfactor, order=2)
    u = imsliceRu
    v = imsliceRv
    w = imsliceRw
    dir(ax.quiver)
    ax.quiver(X,Y,w,u,v,w)
    ax.set_aspect('equal')
    plt.axis('off')

    ydim = X.shape[0]
    xdim = X.shape[1]
    Xw = np.ndarray(shape=X.shape)*0
    Yw = np.ndarray(shape=X.shape)*0
    Zw = np.ndarray(shape=X.shape)*0
    for i in range(ydim):
        for j in range(xdim):
            Xw[i,j] = X[i,j] + imsliceRu[i,j]*(1/zfactor)
            Yw[i,j] = Y[i,j] + imsliceRv[i,j]*(1/zfactor)
            Zw[i,j] = 0 + imsliceRw[i,j]*0
    Xd = np.absolute(imslice[:,:,0])
    Yd = np.absolute(imslice[:,:,1])
    Zd = np.absolute(imslice[:,:,2])
    print 'Mean:' + str(np.mean(Xd)) + ' SD:' + str(np.std(Xd)) + ' Max:' + str(np.ma.max(Xd))
    print 'Mean:' + str(np.mean(Yd)) + ' SD:' + str(np.std(Yd)) + ' Max:' + str(np.ma.max(Yd))
    print 'Mean:' + str(np.mean(Zd)) + ' SD:' + str(np.std(Zd)) + ' Max:' + str(np.ma.max(Zd))
    ax = fig.add_subplot(no_slices, 5, plot_i+2)
    ax.imshow(dwislice, cmap = cm.Greys_r)
    ax.plot(Xw, Yw,'b-', linewidth=0.5)
    ax.plot(Xw.T, Yw.T,'b-', linewidth=0.5)
    ax.set_aspect('equal')
    plt.axis('off')

    plot_displacement_slice(fig, imslice[:,:,0], no_slices, plot_i + 3, 'X-Displacement in voxels', Xd)
    plot_displacement_slice(fig, imslice[:,:,1], no_slices, plot_i + 4, 'Y-Displacement in voxels', Yd)
    plot_displacement_slice(fig, imslice[:,:,2], no_slices, plot_i + 5, 'Z-Displacement in voxels', Zd)

    plot_i = plot_i + 5
    return plot_i

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
    for line_i in range(len(lines)):
        line = lines[line_i]
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
            resolutions.append({'data':np.array(iterations), 'names':resolution_names, 'endcondition':lines[line_i+1]})
            iterations = []
            continue
        if read_state == 1:
            iterations.append(line.split())
    return resolutions[0:-2]

#
# Plots iteration info into pdf
#
# loginfo             - information file 
# out_dir             - output directory
#
def plot_iterationinfo(loginfo, out_dir):
    import matplotlib.pyplot as plt
    import skimage.io
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(out_dir + os.sep + 'convergence_QC.pdf')

    # Go through resolutions
    no_resolutions = len(loginfo)
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    fig.subplots_adjust(wspace=.01)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.05)
    fig.suptitle('Deformation convergence', fontsize=12)
    for resolution_i in range(no_resolutions):
        resolution = loginfo[resolution_i]['data']
        ax = fig.add_subplot(no_resolutions, 1, resolution_i)
        x1 = np.array(range(1,resolution.shape[0]+1)).T
        ax.plot(x1, resolution[:,1].T, 'yo-')
        ax.set_xlabel(loginfo[resolution_i]['endcondition'], fontsize=8, labelpad=0)
        ax.set_ylabel(loginfo[resolution_i]['names'][1], labelpad=0.1)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    pdf.savefig(fig)
    pdf.close()

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
    args = parser.parse_args()
    subdirs = os.listdir(experiment_dir + os.sep + args.subject)
    subdirs_for_QC = []
    for subdir_i in range(len(subdirs)):
        if os.path.isdir(experiment_dir + os.sep + args.subject + os.sep + subdirs[subdir_i]) and (subdirs[subdir_i].find('Motioncorrected_') != -1):
            subdirs_for_QC.append(subdirs[subdir_i])
    B0_images = []
    parameter_files = []
    disp_fields = []
    for subdir_i in range(len(subdirs_for_QC)):
        subdir = experiment_dir + os.sep + args.subject + os.sep + subdirs_for_QC[subdir_i]
        loginfo = read_log(subdir + os.sep + 'elastix.log')
        plot_iterationinfo(loginfo, subdir)
        continue
        print "Loading B-0 of deformed image"
        dcmio = DicomIO.DicomIO()
        dwidcm = dcmio.ReadDICOM_frames(experiment_dir + os.sep + args.subject + '/Motioncorrected')
        dwishape = [dwidcm[0][0].pixel_array.shape[0], dwidcm[0][0].pixel_array.shape[1], len(dwidcm[0]), len(dwidcm)]
        B0_images.append(dwidcm[0])
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
#    plot_deformation_vectors(disp_fields, parameter_files, args.subject, subdirs_for_QC, B0_images)

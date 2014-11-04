#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'

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
from nipype.interfaces.base import CommandLine
import PIL


def DICOM2animatedGIF(dcmdir, outputpath, slice_i, suffix):

    dcmio = DicomIO.DicomIO()
    dwidcm = dcmio.ReadDICOM_frames(dcmdir)
    # Write all frames of slice into set of png files
    for frame_i in range(len(dwidcm)):
        slice = dwidcm[frame_i][slice_i].pixel_array
        dimx = slice.shape[0]
        dimy = slice.shape[1]
        newImg1 = PIL.Image.new('L', (dimx, dimy))
        pixels1 = newImg1.load()
        for i in range (0, dimx):
            for j in range (0, dimy):
                pixels1[i, j] = float(slice[i, j])
        #pixels1[i, j] = float(slice[i, j]) * dwidcm[frame_i][slice_i].RescaleSlope + dwidcm[frame_i][slice_i].RescaleIntercept
        newImg1.save((outputpath + '_' + ('%02d' % frame_i) + '.png'),'PNG')
    cmd = CommandLine('convert -delay 25 -loop 0 %s_*.png %s_%s.gif' % (outputpath, outputpath, suffix))
    cmd.run()
    for frame_i in range(len(dwidcm)):
        os.remove((outputpath + '_' + ('%02d' % frame_i) + '.png'))
    print "convert (ImageMagick):" + cmd.cmd
    return (outputpath + '.gif')

def DICOM2animatedGIF_sidebyside(dcmdir_l, dcmdir_r, outputpath, slice_i, suffix):

    dcmio = DicomIO.DicomIO()
    dwidcm_l = dcmio.ReadDICOM_frames(dcmdir_l)
    dwidcm_r = dcmio.ReadDICOM_frames(dcmdir_r)
    # Write all frames of slice into set of png files
    for frame_i in range(len(dwidcm_l)):
        slice_l = dwidcm_l[frame_i][slice_i].pixel_array.T
        slice_r = dwidcm_r[frame_i][slice_i].pixel_array.T
        dimx = slice_l.shape[0]
        dimy = slice_l.shape[1]
        newImg1 = PIL.Image.new('L', (dimx*2, dimy))
        pixels1 = newImg1.load()
        for i in range (0, dimx):
            for j in range (0, dimy):
                pixels1[i, j] = float(slice_l[i, j])
                pixels1[dimx+i, j] = float(slice_r[i, j])
        #pixels1[i, j] = float(slice[i, j]) * dwidcm[frame_i][slice_i].RescaleSlope + dwidcm[frame_i][slice_i].RescaleIntercept
        newImg1.save((outputpath + '_' + ('%02d' % frame_i) + '.png'),'PNG')
    cmd = CommandLine('convert -delay 25 -loop 0 %s_*.png %s_%s.gif' % (outputpath, outputpath, suffix))
    cmd.run()
    for frame_i in range(len(dwidcm_l)):
        os.remove((outputpath + '_' + ('%02d' % frame_i) + '.png'))
    print "convert (ImageMagick):" + cmd.cmd
    return (outputpath + '.gif')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    parser.add_argument("--outdir", dest="outdir", help="output directory for results", required=True)
    parser.add_argument("--slice", dest="slice", help="slice number", required=True)
    parser.add_argument("--orig_subdir", dest="orig_subdir", help="Subdirectory name for original non-corrected data", required=True)
    args = parser.parse_args()

    print "Converting deformed DICOM image to animated GIF"
    out_dir = args.outdir + os.sep + args.subject
    print "Copying result data to " + out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    slice_no = int(float(args.slice))

    dcmdir_l = experiment_dir + os.sep + args.subject + os.sep + args.orig_subdir
    dcmdir_r = experiment_dir + os.sep + args.subject + os.sep + 'Motioncorrected'
    if not os.path.exists(dcmdir_l):
        print dcmdir_l + ' does not exist'
        sys.exit(1)
    if not os.path.exists(dcmdir_r):
        print dcmdir_r + ' does not exist'
        sys.exit(1)

    print 'Creating animated GIF'
    DICOM2animatedGIF_sidebyside(dcmdir_l, dcmdir_r, (out_dir + os.sep + args.subject + '_slice' + ('%02d' % slice_no)), slice_no, args.orig_subdir + '_vs_Corrected')


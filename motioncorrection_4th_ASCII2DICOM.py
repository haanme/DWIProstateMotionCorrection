#!/usr/bin/env python

experiment_dir = '/Users/eija/Desktop/prostate_MR/pipelinedata'

#
# Convert ASCII fitting results to DICOM
#
# in_file    - ASCII input file
# out_prefix - patient subdir
# out_basedir     - DICOM directory for reference headers
# bounds     - box coordinates of DICOM inside the original DICOM data [xmin, xmax, ymin, ymax, zmin, zmax]
#
def ASCII2DICOM(in_refdir, in_file, out_prefix, out_basedir, bounds):
    import dicom
    import DicomIO
    import bfitASCII_IO
    import numpy as np
    import os
    import shutil
    import copy

    outdir_basename =  out_basedir + out_prefix

    # Resolve new frame list
    dcmio = DicomIO.DicomIO()
    frame_list = dcmio.ReadDICOM_frames(in_refdir + '/' + out_prefix + '/' + 'DICOMconverted')
    slice_1st = frame_list[0][0]
    xdim = slice_1st.Columns
    ydim = slice_1st.Rows
    zdim = slice_1st.NumberOfSlices
    tdim = slice_1st.NumberOfTimeSlices
    print "Original data dimensions:" + str([xdim, ydim, zdim, tdim])
    sample_frame = frame_list[0]
    del frame_list

    # Read data = { 'subwindow': subwindow, 'ROI_No': ROI_No, 'bset': bset, 'ROIslice': ROIslice, 'name': name, 'SIs': SIs }
    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    print ("Reading " + in_file)
    data = ASCIIio.Read(in_file, False)
    pmap_subwindow = data['subwindow']
    pmap_SIs = data['data']
    pmap_names = data['parameters']
    pmap_names = [s.strip('[],\'') for s in pmap_names]
    pmap_slices = data['ROIslice']
    pmap_slices = [s-1 for s in pmap_slices]
    # Save in data in order z,y,x
    out_dirs = []
    for p_i in range(len(pmap_names)):
        SI_i = 0
        outvolume = copy.deepcopy(sample_frame)
        out_dir = outdir_basename + '_' + pmap_names[p_i]
        print "Writing " + pmap_names[p_i] + ' to ' + out_dir
        for z_i in range(zdim):
            # Initialize slice intensity values
            pixel_array = np.zeros([xdim, ydim])
            # Place data into slice subregion
            for x_i in range(pmap_subwindow[0], pmap_subwindow[1]+1):
                for y_i in range(pmap_subwindow[2], pmap_subwindow[3]+1):
                    pixel_array[y_i, x_i] = pmap_SIs[SI_i,p_i]
                    SI_i = SI_i + 1
            pixel_array = pixel_array.T
            intercept = np.amin(pixel_array)
            pixel_array = pixel_array - intercept
            slope = np.amax(pixel_array)-np.amin(pixel_array)
            slope = slope/65535.0
            pixel_array = np.round(pixel_array/slope)
            pixel_array = pixel_array.astype(np.uint16)
            # Place data into slice
            outvolume[z_i].RescaleSlope = slope
            outvolume[z_i].RescaleIntercept = intercept
            outvolume[z_i].PixelData = pixel_array.tostring()
            outvolume[z_i].Columns = xdim
            outvolume[z_i].Rows = ydim
            outvolume[z_i].NumberOfSlices = zdim
            outvolume[z_i].NumberOfTimeSlices = 1
            outvolume[z_i].ImageIndex = z_i+1
        # Create output directory if it does not exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        # Write output DICOM
        filenames = dcmio.WriteDICOM_frames(out_dir, [outvolume], 'IM')
        out_dirs.append(out_dir)
    return out_dirs

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
    parser.add_argument("--in_refdir", dest="in_refdir", help="input reference DICOM directory", required=True)
    parser.add_argument("--in_ASCIIdir", dest="in_ASCIIdir", help="input directory for ASCII", required=True)
    parser.add_argument("--out_basedir", dest="out_basedir", help="output base directory for DICOMs", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    parser.add_argument("--suffix", dest="suffix", help="suffix for read files", required=True)
    args = parser.parse_args()

    # Read subregion coordinates of non-corrected and corrected DICOMs
    subregion_in_originalDICOM = np.loadtxt(experiment_dir + '/' + args.subject + '/subregion10.txt')
    print "Subregion in original dicom " + str(subregion_in_originalDICOM)

    print (args.in_ASCIIdir + os.sep + args.subject + '*' + args.suffix + '.txt')
    filenames = glob.glob((args.in_ASCIIdir + os.sep + args.subject + '*' + args.suffix + '.txt'))

    for filename in filenames:
        print "Converting ASCII " + filename + " to DICOM"
        # Write motioncorrected as single multifile DICOM subfolder
        DICOMbase = args.out_basedir + '/' + args.subject + '/'
        print "Output basedir " + DICOMbase
        print "Converting to DICOM from ASCII parameter file"
        try:
            ASCII2DICOM(args.in_refdir, filename, args.subject, DICOMbase, subregion_in_originalDICOM)
        except Exception as inst:
            raise
        sys.exit(0)

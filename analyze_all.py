# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:52:00 2014

@author: merisaah
"""

#!/usr/bin/env python
output_path = 'results_RMSEchange'
original_raw_path = 'raw_Noncorrected'
corrected_raw_path = 'raw_Motioncorrected'
original_path = 'results_Noncorrected'
corrected_path = 'results_Motioncorrected'
prostatemask_DICOM = r'/Users/eija/Desktop/prostate_MR/Carimasproject_files_Hb_outline_v1013/DICOMmasks'
original_DICOM = r'/Users/eija/Desktop/prostate_MR/PET_MR_dwis'


#
# Resolves image data from ASCII data for use
#
# data             - data structure from ASCII file
# mask_img         - optional voxewise masking image
#
def resolve_ASCIIimgdata(data, dim, mask_img=None):
    import numpy as np 

    values = data['data']
    print 'Resolving img data from ASCII: length=' + str(len(values)) + 'x' + str(values.shape[1]) + ' ' + str(dim)
    img = np.ndarray(shape=(dim[0], dim[1], dim[2], values.shape[1]))
    if mask_img == None:
        for frame_i in range(values.shape[1]):
            frame_img = values[:, frame_i].reshape([dim[2], dim[0], dim[1]])
            for z in range(dim[2]):
                img[:,:,z,frame_i] = frame_img[z, :, :]
    else:
        print 'Mask image size:' + str(mask_img.shape)
        for frame_i in range(values.shape[1]):
            frame_img = np.zeros(shape=(dim[0], dim[1], dim[2]))
            SI_i = 0
            for z in range(dim[2]):
                for y in range(dim[1]):
                    for x in range(dim[0]):
                        if mask_img[x,y,z] != 0:
                            frame_img[x,y,z] = values[SI_i, frame_i]
                            SI_i += 1
            print str(SI_i) + ' ROI voxels found'
            img[:,:,:,frame_i] = frame_img
    return img

#
# Resolves dimensions in ASCII data
#
# data             - data structure from ASCII file
#
def resolve_ASCIIdimdata(data):
    slice_numbers = data['ROIslice']
    xy_bounds = data['subwindow']
    values = data['data']
    dim = [xy_bounds[3]-xy_bounds[2], xy_bounds[1]-xy_bounds[0], len(slice_numbers), values.shape[1]]
    return dim    

#
# Resolves content of ASCII data for use
#    
# data             - data structure from ASCII file
# mask_img         - optional voxewise masking image
#
def resolve_ASCIIparamdata(data, mask_img=None):
   
    parameternames = data['parameters']
    slice_numbers = data['ROIslice']
    xy_bounds = data['subwindow']
    bset = data['bset']
    name = data['name']
    values = data['data']
    if mask_img == None:
        dim = [xy_bounds[3]-xy_bounds[2], xy_bounds[1]-xy_bounds[0], len(slice_numbers), values.shape[1]]
        img = resolve_ASCIIimgdata(data, dim)
    else:
        dim = [mask_img.shape[0], mask_img.shape[1], mask_img.shape[2], values.shape[1]]
        img = resolve_ASCIIimgdata(data, mask_img.shape, mask_img)

    return img, dim, parameternames, slice_numbers, xy_bounds, bset, name
   
#
# Resolves image data in DICOM containing ROIs
#
# DICOMpath           - DICOMpath containing ROI data
# dim                 - data dimensions
# xy_bounds           - bounds of ROI in original full image
# z_slices            - z-slices 
#
def resolve_DICOMROI_imgdata(DICOMpath, xy_bounds=None, z_slices=None):
    import scipy.io    
    import numpy as np
    import DicomIO    
    
    dcmio = DicomIO.DicomIO()
    DICOM_ROIdata = dcmio.ReadDICOM_frames(DICOMpath)
    
    # Create and write mask images
    print str(len(DICOM_ROIdata)) + " ROIs"
    
    # Create mask around combined ROIs
    ROIpixel_array_all = []
    ROInames = []
    
    # Go through all ROIs
    dim = [DICOM_ROIdata[0][0].pixel_array.shape[0], DICOM_ROIdata[0][0].pixel_array.shape[1], len(DICOM_ROIdata[0])]
    for roi_i in range(len(DICOM_ROIdata)):
        if not xy_bounds == None:
            ROI = np.zeros(shape=(xy_bounds[1]-xy_bounds[0], xy_bounds[3]-xy_bounds[2], dim[2]))
            for z in range(dim[2]):
                ROI[:,:,z] = DICOM_ROIdata[roi_i][z].pixel_array[xy_bounds[0]:xy_bounds[1], xy_bounds[2]:xy_bounds[3]]
        else:
            ROI = np.zeros(shape=(dim[0], dim[1], dim[2]))
            for z in range(dim[2]):
                ROI[:,:,z] = DICOM_ROIdata[roi_i][z].pixel_array

        ROInames.append('ROI' + ('%02d' % roi_i))
        ROIpixel_array_all.append(ROI)
                
    return ROIpixel_array_all, ROInames

#
# Resolves image data in mat-file containing ROIs
#
# matfilename         - mat-file containing ROI data
# dim                 - data dimensions
# xy_bounds           - bounds of ROI in original full image
# z_slices            - z-slices 
#
def resolve_MatROI_imgdata(matfilename, dim, xy_bounds, z_slices):
    import scipy.io    
    import numpy as np
    
    mat = scipy.io.loadmat(matfilename)
    # Get list of ROIs
    ROIs = mat['ROIs'].tolist()[0]
    # Get list of slices where ROIs are located
    ROIslices = mat['ROIslices'][0].tolist()
    # Create and write mask images
    print str(len(ROIs)) + " ROIs"
    
    # Create mask around combined ROIs
    ROIpixel_array_all = []
    ROInames = []
    # Go through all ROIs
    for roi_i in range(len(ROIs)):
        ROI = np.zeros(shape=(dim[0], dim[1], dim[2]))
        ROIlist = ROIs[roi_i].tolist()
        ROIname = str(ROIlist[0][0][0][0])
        ROIpixel_array = ROIlist[0][0][1]
        print "catenating " + ROIname
        ROI[:,:,ROIslices[roi_i]] = ROIpixel_array[xy_bounds[2]:xy_bounds[3],xy_bounds[0]:xy_bounds[1]]
        ROInames.append(ROIname)
        ROIpixel_array_all.append(ROI)
                
    return ROIpixel_array_all, ROInames

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
# Resolve mat-filename containing ROI masks
#
# output_prefix - output prefix
#
def resolve_matfilename(output_prefix):
    # Resolve mat-file name
    parts = output_prefix.split('_')
    patient_no_str, patientname_str, bset_str, rep_str = split_subjectid(output_prefix)
    if not (bset_str=='hB'):
        raise Exception((output_prefix + " UNSUPPORTED B-SET"))
    if (bset_str=='hB'):
        matfilename = mask_matfile_basedir_hB + os.sep + patient_no_str + '_' + rep_str + '_DICOMconverted.mat'
    if (bset_str=='lB'):
        matfilename = mask_matfile_basedir_lB + os.sep + patient_no_str + '_' + rep_str + '_DICOMconverted.mat'
    return matfilename    


#
# Resolve DICOM path containing ROI masks
#
# output_prefix - output prefix
#
def resolve_DICOMpath(output_prefix):
    import os
    import glob    
    
    # Resolve mat-file name
    parts = output_prefix.split('_')
    patient_no_str, patientname_str, bset_str, rep_str = split_subjectid(output_prefix)
    print 'Searching DICOM from:' + (prostatemask_DICOM + os.sep + patient_no_str + '_' + patientname_str + '_*')
    paths = glob.glob((prostatemask_DICOM + os.sep + patient_no_str + '_' + patientname_str + '_*'))
    return paths[0]
        
from argparse import ArgumentParser
import sys
import os
import DicomIO
import conversions as conv
import time
import numpy as np
import bfitASCII_IO
import glob
import plot_utils

if __name__ == "__main__":
#    parser = ArgumentParser()
#    parser.add_argument("--dicomdir", dest="dicomdir", help="dicomdir", required=True)
#    args = parser.parse_args()

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filenames_raw = glob.glob((original_raw_path + os.sep + '*ASCII.txt'))
    filenames_orig = glob.glob((original_path + os.sep + '*_results.txt'))
    filenames_corr = glob.glob((corrected_path + os.sep + '*_results.txt'))

    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    print filenames_raw
    for fname_raw in filenames_raw:
        splitted_raw = fname_raw.split(os.sep)
        splitted_raw = splitted_raw[1].split('_')
        splitted_raw = splitted_raw[:-2]
        patient_name = ''.join([item + '_' for item in splitted_raw])
        patient_name = patient_name[0:-1]
        print 'Patient name ' + patient_name
        if not patient_name.endswith('Set2'):
            continue
        # Reach for non-corrected fitted file
        found_fname_orig = ''
        for fname_orig in filenames_orig:
            splitted_orig = fname_orig.split(os.sep)
            splitted_orig = splitted_orig[1].split('_')
            splitted_orig = splitted_orig[:-4]
            if splitted_orig == splitted_raw:
                found_fname_orig = fname_orig
                break
        # Reach for corrected fitted file
        found_fname_corr = ''                
        for fname_corr in filenames_corr:
            splitted_corr = fname_orig.split(os.sep)
            splitted_corr = splitted_corr[1].split('_')
            splitted_corr = splitted_corr[:-4]
            if splitted_corr == splitted_raw:
                found_fname_corr = fname_corr
                break
        # Collect data for plotting
        print 'Reading non-corrected: ' + (found_fname_orig)
        data_orig = ASCIIio.Read((found_fname_orig), False)
        data_orig_dim = resolve_ASCIIdimdata(data_orig)
        data_orig_slice_numbers = data_orig['ROIslice']
        data_orig_xy_bounds = data_orig['subwindow']
        data_bounds = [data_orig_xy_bounds[0], data_orig_xy_bounds[1], data_orig_xy_bounds[2], data_orig_xy_bounds[3], data_orig_slice_numbers[0], data_orig_slice_numbers[-1]]
        data_bounds = np.subtract(data_bounds, 1)
        print 'Data bounds ' + str(data_bounds)
        print 'Reading corrected: ' + (found_fname_corr)
        data_corr = ASCIIio.Read((found_fname_corr), False)

        fname_raw_orig = original_raw_path + os.sep + patient_name + '_Noncorrected_ASCII.txt'
        fname_raw_corr = corrected_raw_path + os.sep + patient_name + '_Motioncorrected_ASCII.txt'
        print 'Reading non-corrected raw: ' + (fname_raw_orig)
        data_raw_orig = ASCIIio.Read((fname_raw_orig), True)
        print 'Reading corrected raw: ' + (fname_raw_corr)
        data_raw_corr = ASCIIio.Read((fname_raw_corr), True)

        # Read DICOM mask data        
        prefix = ''.join([item + '_' for item in splitted_raw])
        DICOMpath = prostatemask_DICOM + os.sep + patient_name
#        DICOMpath = resolve_DICOMpath(prefix)
        if not os.path.exists(DICOMpath):
            print (DICOMpath + " DOES NOT EXIST")
            continue
        ROIpixel_array_all, ROInames = resolve_DICOMROI_imgdata(DICOMpath)

        print 'ROIs:' + str(ROInames)
        print len(ROIpixel_array_all)

        # Do plotting of ROI position, displacement-RMSE correlation, RMSE boxplot inside ROI
        plot_utils.plotdataROI(fname_orig, data_corr, data_orig, data_raw_orig, data_raw_corr, ROIpixel_array_all, ROInames)
        # Do plotting of voxelwise parameter change
        # plot_utils.plot_data(fname_orig, data_corr, data_orig, data_raw_orig, data_raw_corr)
        # Write parametric map
        splitted = patient_name.split('_D')
        dicomname = splitted[0]
        print 'DICOM name:' + dicomname
        #ASCII2DICOM(data_orig, 'DICOMconverted', 'results_DICOM_pmaps_noncorrectd', dicomname, patient_name, data_bounds, ROIpixel_array_all, ROInames)
        #ASCII2DICOM(data_corr, 'DICOMconverted', 'results_DICOM_pmaps_corrected', dicomname, patient_name, data_bounds, ROIpixel_array_all, ROInames)
        break

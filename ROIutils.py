# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:52:00 2014

@author: merisaah
"""

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

#!/usr/bin/env python

DICOM_basedir = '/Users/eija/Desktop/prostate_MR/PET_MR_dwis'
ROI_pickle_dir = 'ROI_pickle_files'
ROI_mask_dir = 'ROI_mask_files'
ROI_mask_png_dir = 'ROI_mask_png_files'

# Split cpr name into its components
def split_cpr_name(filename):
    prefix = filename.split('hB')
    prefix = prefix[0]
    prefix = prefix[:len(prefix)-1]
    split_i = prefix.find('_')
    number_str = prefix[:split_i]
    name_str = prefix[split_i+1:]
    number = int(float(number_str))
    return number, name_str

def resolve_imagefilename_from_cpr(filename):

    f = open(filename)
    lines = f.readlines()
    f.close()

    # Find line that has entry for DICOM image
    DICOM_line = ''
    for line_i in range(len(lines)):
        line = lines[line_i]
        index = line.find(' Type="Carimas.Data.Images.TPCLibImage" >')
        if (index != -1):
            DICOM_line = lines[line_i+1]
            break

    # Resolve corresponding path in the current environment
    filename_start = [i for i, x in enumerate(DICOM_line) if x == ">"]
    filename_end = [i for i, x in enumerate(DICOM_line) if x == "<"]
    filename_str = DICOM_line[filename_start[0]+1:filename_end[1]]
    filename_str = filename_str.replace('F:\All_philips_DWI_dicom\PET_MR_dwis', (DICOM_basedir))
    filename_str = filename_str.replace('\\','/')
    filename_str = filename_str.replace('DICOMconverted_Gs3x3','DICOMconverted')
    return filename_str

# Resolve DICOM path
def resolve_tags_from_cpr(filename, dim):
    import numpy as np

    f = open(filename)
    lines = f.readlines()
    f.close()

    ROIs = []
    mask_lines = []
    for line_i in range(len(lines)):
        line = lines[line_i]
        index = line.find('Type="Carimas.Data.VOIs.ROI" >')
        if (index != -1):
            tag_inds = [i for i, x in enumerate(line) if x == "\""]
            ROI_name = line[tag_inds[2]+1:tag_inds[3]-1]
            line = lines[line_i+4]
            mask_start = line.find('[')
            mask_end = line.find(']')
            line = line[mask_start+1:mask_end-1]
            items = line.split(' ')
            packedmask = []
            for item_i in range(len(items)):
                if len(items[item_i]) > 0:
                    packedmask.append(float(items[item_i]))

            # unpack mask data
            position = 0
            mask = np.empty(dim)
            planelength = dim[0]*dim[1]
            ROIslice = -1
            inds_nonzero = []
            for mask_i in range(len(packedmask)):
                if packedmask[mask_i] > 0:
                    pos_z = np.floor(position/(planelength))
                    pos_y = np.floor((position-planelength*pos_z)/dim[0])
                    pos_x = int((position-planelength*pos_z)-dim[0]*pos_y)
                    for x_i in range(pos_x,pos_x + int(packedmask[mask_i])+1):
                        mask[x_i, pos_y, pos_z] = 1
                        inds_nonzero.append([x_i, pos_y, pos_z])
                    ROIslice = pos_z
                    position = position + packedmask[mask_i]
                else:
                    position = position - packedmask[mask_i]
            vol = len(inds_nonzero)
            ROIs.append({"name":ROI_name, "vol":vol, "mask":mask, "ROIslice":int(ROIslice), "dim":dim})
    return ROIs

# Read dimensions from DICOM file
def readDICOM_DIM(filename):
    import dicom
    ds = dicom.read_file(filename)
    xdim = ds.Columns
    ydim = ds.Rows
    zdim = ds.NumberOfSlices
    return [xdim, ydim, zdim]

# Read mask information from cpr
def read_cpr(filename):
    imagefilename = resolve_imagefilename_from_cpr(filename)
    xdim, ydim, zdim = readDICOM_DIM(imagefilename)
    ROIs = resolve_tags_from_cpr(filename, [xdim, ydim, zdim])
    return ROIs


################################################

import os
import pickle
import numpy as np
from PIL import Image

# Create output directory if it does not exist
if not os.path.exists(ROI_pickle_dir):
    os.makedirs(ROI_pickle_dir)
if not os.path.exists(ROI_mask_dir):
    os.makedirs(ROI_mask_dir)
if not os.path.exists(ROI_mask_png_dir):
    os.makedirs(ROI_mask_png_dir)

# Go through project files
names = os.listdir('.')
for name_i in range(len(names)):
    filename = names[name_i]
    basename, ext = os.path.splitext(filename)
    path, name = os.path.split(filename)
    if (ext=='.cpr'):
        number, name_str = split_cpr_name(name)
        print "Processing " + name + "[" + str(number) + "][" + name_str + "]"
        ROIs = read_cpr(filename)
        # Save ROIs
        for ROI_i in range(len(ROIs)):
            basename = (str(number) + '_' + name_str + '_' + str(ROI_i) + '_' + ROIs[ROI_i]['name'])
            # Save mask with numpy
            mask = ROIs[ROI_i]['mask']
            slice = ROIs[ROI_i]['ROIslice']
            img_basename = basename + '_slice' + str(slice)
            mask = mask[:,:,slice]
            print mask.shape
            np.savetxt((ROI_mask_dir + '/' + img_basename + '.txt'), mask)
            img = Image.fromarray(mask).convert('L')
            img.save((ROI_mask_png_dir + '/' + img_basename + '.png'))
            # Save entire structure with pickle in binary format
            afile = open((ROI_pickle_dir + '/' + basename + '.pkl'), 'wb')
            pickle.dump(ROIs[ROI_i], afile)
            afile.close()
        break


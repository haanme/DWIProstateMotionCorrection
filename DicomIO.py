# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 20:20:02 2014

@author: merisaah
"""
import dicom
import os
import numpy as np

class DICOMError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

class DICOMReadError(DICOMError):
     def __str__(self):
         return repr('Read error ' + self.value)


class DICOMWriteError(DICOMError):
     def __str__(self):
         return repr('Write error ' + self.value)
   
class DicomIO:
    """I/O functions for DICOM stacks"""  
    def ReadDICOM_slices(self, path):
        if not(os.access(path, os.F_OK)):
            raise DICOMReadError('Path ' + path + ' does not exist' )
        if not(os.access(path, os.R_OK)):
            raise DICOMReadError('Path ' + path + ' does not have read permission' )    
        #resolve filenames
        filenames = os.listdir(path)
        if len(filenames) == 0:
            raise DICOMReadError('Path ' + path + ' does not have files' )                
        #read slices into array
        slice_list = []
        error_list = []
        for filename_i in range(len(filenames)): 
            filename = os.path.join(path, filenames[filename_i])
            print 'Reading ' + filename
            #Skip files that could not be read
            if not(os.access(filename, os.R_OK)):
                print 'File ' + filename + ' does not have read permission'
                continue            
            try:
                ds = dicom.read_file(filename)
                slice_list.append([ds.ImagePositionPatient[2],ds])
            except Exception as e:
                print e.message
                error_list.append(DICOMReadError(e.message))
        slice_list.sort(key=lambda tup: tup[0])                
        slice_list = [tup[1] for tup in slice_list]
        return slice_list, error_list

    def ReadDICOM_frames(self, path):
        if not(os.access(path, os.F_OK)):
            raise DICOMReadError('Path ' + path + ' does not exist' )
        if not(os.access(path, os.R_OK)):
            raise DICOMReadError('Path ' + path + ' does not have read permission' )
        #resolve filenames
        filenames = os.listdir(path)
        if len(filenames) == 0:
            raise DICOMReadError('Path ' + path + ' does not have files' )
        #read slices into array
        slice_list = []
        FrameReferenceTimes = []
        frame_list = []
        print_step = np.floor(len(filenames)/10)
        for filename_i in range(len(filenames)):
            filename = os.path.join(path, filenames[filename_i])
            if np.mod(filename_i, print_step) == 0 or filename_i == (len(filenames)-1):
                print 'Reading ' + filename
            #Skip files that could not be read
            if not(os.access(filename, os.R_OK)):
                print 'File ' + filename + ' does not have read permission'
                continue
            try:
                ds = dicom.read_file(filename)
                if not (ds.FrameReferenceTime in FrameReferenceTimes):
                    FrameReferenceTimes.append(ds.FrameReferenceTime)
                    frame_list.append([])
                slice_list.append(ds)
            except Exception as e:
                raise DICOMReadError(e.message)
        #collect data for frame in order
        FrameReferenceTimes.sort()
        for slice_i in range(len(slice_list)):
            frame_index = FrameReferenceTimes.index(slice_list[slice_i].FrameReferenceTime)
            frame_list[frame_index].append(slice_list[slice_i])
        #sort frame data
        for frame_i in range(len(frame_list)):
            frame_list[frame_i].sort(key=lambda x: (x.ImagePositionPatient[2]))
        return frame_list

    def WriteDICOM_frames(self, path, frame_list, prefix):
        file_i = 0
        filenames = []
        # go through all frames
        for frame_i in range(len(frame_list)):
            # go through all slices
            for slice_i in range(len(frame_list[frame_i])):
                filename = path + '/' + prefix + ("%06d" % file_i)
                frame_list[frame_i][slice_i].save_as(filename)
                file_i = file_i + 1
                filenames.append(filename)
        return filenames

    #
    # Write DICOM from intensity values
    # 
    # pixel_array    - 2D numpy.ndarray
    # filename       - filename where data is written
    # itemnumber     - item number, default == 0, determines also (0020,0013),(0020,1041)
    # PhotometricInterpretation - 'MONOCHROME2', 'RGB'
    #
    def WriteDICOM_slice(self, pixel_array,filename, itemnumber=0, PhotometricInterpretation="MONOCHROME2"):
        from dicom.dataset import Dataset, FileDataset
        import numpy as np
        import datetime, time
        """
        INPUTS:
        pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
        filename: string name for the output file.
        """
        ## This code block was taken from the output of a MATLAB secondary
        ## capture.  I do not know what the long dotted UIDs mean, but
        ## this code works.
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
        file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
        ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-','')
        ds.ContentTime = str(time.time()) #milliseconds since the epoch
        ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
        ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
        ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        ds.SOPClassUID = 'Secondary Capture Image Storage'
        ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'
        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        if PhotometricInterpretation=="MONOCHROME2":
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'
        elif PhotometricInterpretation=="RGB":
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'    
            pixel_array = pixel_array[0]
            print pixel_array.shape
        ds.Columns = pixel_array.shape[0]
        ds.ItemNumber = str(itemnumber)
        ds.InstanceNumber = str(itemnumber)
        ds.SliceLocation = str(itemnumber)
        ds.Rows = pixel_array.shape[1]
        if pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)
            ds.PixelData = pixel_array.tostring()
        ds.save_as(filename)

        return filename

    #
    # Write DICOM from intensity values
    # 
    # pixel_arrays   - 3D numpy.ndarray
    # plans          - existing DICOM data for all planes where pixel_array is inserted
    # path           - path where data is to be written
    # prefix         - filename prefix where data is written
    #
    def WriteDICOM_slices(self, pixel_arrays, plans, path, prefix):
        import numpy as np        

        if pixel_arrays.shape[2] != len(plans):
            raise Exception('Number of slices in array and pland do not match')
            
        if not os.path.exists(path):
            os.makedirs(path)            
        
        filenames = []
        for slice_i in range(len(plans)):
            ds = plans[slice_i]
            if pixel_arrays[slice_i].dtype != np.uint16:
                pixel_array = pixel_arrays[slice_i].astype(np.uint16)
            ds.PixelData = pixel_array.tostring()
            filename = path + os.pathsep + prefix + ('%06d' % slice_i)
            ds.save_as(filename)
            filenames.append(filename)
        return filenames

    #
    # Write DICOM from intensity values
    # 
    # pixel_array               - 3D numpy.ndarray
    # prefix                    - filename prefix where data is written
    # PhotometricInterpretation - 'MONOCHROME2', 'RGB'
    #
    def WriteDICOM_slices_noplan(self, pixel_array, prefix, PhotometricInterpretation="MONOCHROME2"):
        filenames = []
        for slice_i in range(pixel_array.shape[2]):
            filename = prefix + ('%06d' % slice_i)
            print "Writing " + filename
            filenames.append(self.WriteDICOM_slice(pixel_array[:,:,slice_i], filename, slice_i, PhotometricInterpretation))
        return filenames

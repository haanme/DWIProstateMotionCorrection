# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 20:20:02 2014

@author: merisaah
"""
import dicom
import os

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
                raise DICOMReadError(e.message)
        slice_list.sort(key=lambda tup: tup[0])                
        slice_list = [tup[1] for tup in slice_list]
        return slice_list                

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
        for filename_i in range(len(filenames)):
            filename = os.path.join(path, filenames[filename_i])
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

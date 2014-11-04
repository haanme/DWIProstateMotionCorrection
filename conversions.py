#!/usr/bin/env python

mcverter_basedir = './'

#
# Moves file to results folder, overwriting the existing file
#
# filename   - file to be moved
# out_prefix - subject specific prefix
#
def move_to_results(filename, experiment_dir , out_prefix):
    import os
    import shutil
    outfile = experiment_dir + '/' + out_prefix + '/' + os.path.basename(filename)
    if os.path.isfile(outfile):
        os.remove(outfile)
    shutil.move(filename,outfile)
    return outfile

#
# Split filename into (<root>/<basename>.<extension>)
#
# filename - filename that is splitted
#
def split_ext(filename):
    import os
    # split path and filename
    root, basename = os.path.split(filename)
    # split extensions until none is found, catenating extensions
    basename, ext_new = os.path.splitext(basename)
    ext = ext_new
    while len(ext_new) > 0:
        basename, ext_new = os.path.splitext(basename)
        ext = ext_new + ext
    return root, basename, ext

#
# Renames file basename in its location
#
# filename     - filename that is renamed
# basename_new - new basename
#
def rename_basename_to(filename, basename_new):
    import os
    import shutil
    root, basename, ext = split_ext(filename)
    filename_new = os.path.join(root, (basename_new + ext))
    if os.path.isfile(filename_new):
        os.remove(filename_new)
    shutil.move(filename,filename_new)
    return filename_new

#
# Replaces pattern in file in its location
#
# filename - filename that is modified in-place
# pattern  - pattern that is replaced by subst
# subst    - substitution to pattern
#
def replace_inplace(filename, pattern, subst):
    from tempfile import mkstemp
    from shutil import move
    from os import remove, close
    #Create temp file
    fh, abs_path = mkstemp()
    new_file = open(abs_path,'w')
    old_file = open(filename)
    for line in old_file:
        new_file.write(line.replace(pattern, subst))
    #close temp file
    new_file.close()
    close(fh)
    old_file.close()
    #Remove original file
    remove(filename)
    #Move new file
    move(abs_path, filename)

#
# Convert DICOM to ITK's mhd
#
# dicomdir   - input DICOM directory
# out_prefix - subject specific prefix
#
def dicom2mhd(dicomdir, experiment_dir, out_prefix):
    from nipype.utils.filemanip import split_filename
    from nipype.interfaces.base import CommandLine
    import os
    _, name, _ = split_filename(dicomdir)
    outfile_mhd = experiment_dir + '/' + out_prefix + '/' + name + '_tmp/' + 'output' + '.mhd'
    outfile_raw = experiment_dir + '/' + out_prefix + '/' + name + '_tmp/' + 'output' + '.raw'
    outfile_txt = experiment_dir + '/' + out_prefix + '/' + name + '_tmp/' + 'output' + '_info.txt'
    outdir = experiment_dir + '/' + out_prefix + '/' + name + '_tmp'
    cmd = CommandLine((mcverter_basedir + 'mcverter %s -r -f meta -o %s -F-PatientName-SeriesDate-SeriesDescription-StudyId-SeriesNumber' % (dicomdir,outdir)))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()
    # Move to results folder
    outfile_mhd = move_to_results(outfile_mhd, experiment_dir, out_prefix)
    outfile_raw = move_to_results(outfile_raw, experiment_dir, out_prefix)
    outfile_txt = move_to_results(outfile_txt, experiment_dir, out_prefix)
    os.rmdir(outdir)
    # Rename basename, and reference in mhd header
    outfile_mhd = rename_basename_to(outfile_mhd, name)
    outfile_raw = rename_basename_to(outfile_raw, name)
    outfile_txt = rename_basename_to(outfile_txt, name)
    replace_inplace(outfile_mhd, ('output.raw'), (name + '.raw'))

    return outfile_mhd, outfile_raw, outfile_txt

#
# Convert DICOM to Nifti
#
# dicomdir   - input DICOM directory
# out_prefix - subject specific prefix
# out_suffix - output file suffix
#
def dicom2nii(dicomdir, experiment_dir, out_prefix, out_suffix):
    import os
    import shutil
    from nipype.interfaces.base import CommandLine

    dirnames = os.listdir(dicomdir)
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            os.remove(os.path.join(dicomdir, dirnames[d_i]))
        if fileExtension == '.bval':
            os.remove(os.path.join(dicomdir, dirnames[d_i]))
        if fileExtension == '.bvec':
            os.remove(os.path.join(dicomdir, dirnames[d_i]))

    from nipype.interfaces.base import CommandLine
    basename = experiment_dir + '/' + out_prefix + '/' + out_prefix + out_suffix
    cmd = CommandLine('/Users/eija/Documents/osx/dcm2nii -a Y -d N -e N -i N -p N -o %s %s' % (basename,dicomdir))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()

    dirnames = os.listdir(dicomdir)
    filename_nii = ''
    filename_bvec = ''
    filename_bval = ''
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            if len(filename_nii) > 0:
                raise "multiple copies of .nii.gz was found"
            filename_nii = fileName
        if fileExtension == '.bval':
            if len(filename_nii) > 0:
                raise "multiple copies of .bval was found"
            filename_bval = fileName
        if fileExtension == '.bvec':
            if len(filename_nii) > 0:
                raise "multiple copies of .bvec was found"
            filename_bvec = fileName

    outfile = move_to_results((dicomdir + '/' + filename_nii + '.gz'), experiment_dir, out_prefix)
    outfile_bval = ''
    outfile_bvec = ''
    if len(filename_bval) > 0:
        outfile_bval = move_to_results((dicomdir + '/' + filename_bval + '.bval'), experiment_dir, out_prefix)
    if len(filename_bvec) > 0:
        outfile_bvec = move_to_results((dicomdir + '/' + filename_bvec + '.bvec'), experiment_dir, out_prefix)

    return outfile, outfile_bval, outfile_bvec

#
# Gunzip (.nii.gz to .nii conversion)
#
# in_file    - input file (.nii.gz)
#
def gznii2nii(in_file):
    import os
    import shutil
    from nipype.interfaces.base import CommandLine

    fileName, fileExtension = os.path.splitext(in_file)
    cmd = CommandLine('gunzip -f -k %s.nii.gz' % (fileName))
    print "gunzip NII.GZ:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nii' % (fileName))

#
# Convert nii 2 nrrd
#
# filename_nii  - DTI file (.nii.gz)
# filename_bval - b-value file (ASCII)
# filename_bvec - b-vector file (ASCII)
# out_prefix - subject specific prefix
# out_suffix - output file suffix
#
def nii2nrrd(filename_nii, filename_bval, filename_bvec, experiment_dir , out_prefix, out_suffix):
    import os
    import shutil

    from nipype.interfaces.base import CommandLine
    basename = experiment_dir + '/' + out_prefix + '/' + out_prefix + out_suffix
    cmd = CommandLine('DWIConvert --inputVolume %s --outputVolume %s.nrrd --conversionMode FSLToNrrd --inputBValues %s --inputBVectors %s' % (filename_nii, basename, filename_bval, filename_bvec))
    print "NII->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nrrd' % (basename))

#
# Convert dicom 2 nrrd
#
# dicomdir   - input DICOM directory
# out_prefix - subject specific prefix
# out_suffix - output file suffix
#
def dicom2nrrd(dicomdir, experiment_dir, out_prefix, out_suffix):
    import os
    import shutil

    dirnames = os.listdir(dicomdir)
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            os.remove(dirnames[d_i])
        if fileExtension == '.bval':
            os.remove(dirnames[d_i])
        if fileExtension == '.bvec':
            os.remove(dirnames[d_i])
    
    from nipype.interfaces.base import CommandLine
    basename = experiment_dir + '/' + out_prefix + '/' + out_prefix + out_suffix
    cmd = CommandLine('/Users/eija/Documents/osx/dcm2nii -a Y -d N -e N -i N -p N -o %s %s' % (basename,dicomdir))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()

    dirnames = os.listdir(dicomdir)
    filename_nii = ''
    filename_bvec = ''
    filename_bval = ''
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            if len(filename_nii) > 0:
                raise "multiple copies of .nii.gz was found"
            filename_nii = fileName
        if fileExtension == '.bval':
            if len(filename_nii) > 0:
                raise "multiple copies of .bval was found"
            filename_bval = fileName
        if fileExtension == '.bvec':
            if len(filename_nii) > 0:
                raise "multiple copies of .bvec was found"
            filename_bvec = fileName

    move_to_results((dicomdir + '/' + filename_nii + '.gz'), experiment_dir, out_prefix)
    move_to_results((dicomdir + '/' + filename_bval + '.bval'), experiment_dir, out_prefix)
    move_to_results((dicomdir + '/' + filename_bvec + '.bvec'), experiment_dir, out_prefix)

    cmd = CommandLine('DWIConvert --inputVolume %s.nii.gz --outputVolume %s.nrrd --conversionMode FSLToNrrd --inputBValues %s.bval --inputBVectors %s.bvec' % (basename, basename, basename, basename))
    print "NII->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nrrd' % (basename))

#
# Convert nrrd to Nifti
#
# in_file    - input NRRD file (.nrrd)
# out_prefix - subject specific prefix
#
def nrrd2nii(in_file, experiment_dir, output_prefix):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    _, name, _ = split_filename(in_file)
    out_vol = experiment_dir + '/' + output_prefix + '/' + ('%s.nii.gz' % name)
    out_bval = experiment_dir + '/' + output_prefix + '/' + ('%s.bval' % name)
    out_bvec = experiment_dir + '/' + output_prefix + '/' + ('%s.bvec' % name)
    
    cmd = CommandLine(('DWIConvert --inputVolume %s --outputVolume %s --outputBValues %s'
                       ' --outputBVectors %s --conversionMode NrrdToFSL') % (in_file, out_vol,
                                                                             out_bval, out_bvec))

    print "NRRD->NIFTI:" + cmd.cmd
    cmd.run()
    return opap(out_vol), opap(out_bval), opap(out_bvec)

#
# Convert single frame nrrd to Nifti
#
# in_file    - input NRRD file (.nrrd)
# out_prefix - subject specific prefix
#
def nrrd2nii_pmap(in_file, output_prefix):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    _, name, _ = split_filename(in_file)
    out_vol = experiment_dir + '/' + output_prefix + '/' + ('%s.nii.gz' % name)

    cmd = CommandLine(('DWIConvert --inputVolume %s --outputVolume %s'
                       ' --conversionMode NrrdToFSL') % (in_file, out_vol))

    print "NRRD->NIFTI:" + cmd.cmd
    cmd.run()
    return opap(out_vol), opap(out_bval), opap(out_bvec)

#
# Convert ASCII fitting results to DICOM
#
# data         - data from ASCII input file
# in_dir       - DICOM directory for reference headers
# out_prefix   - patient subdir
# out_prefix   - patient output subdir
# bounds       - box coordinates of DICOM inside the original DICOM data [xmin, xmax, ymin, ymax, zmin, zmax]
# ROIimgs      - optional ROI mask image
# ROInames     - optional ROI names
#
def ASCII2DICOM(data, in_dir, outdir_basename, in_prefix, out_prefix, bounds, ROIimgs=None, ROInames=None):
    import dicom
    import DicomIO
    import numpy as np
    import os
    import shutil

    # Resolve new frame list
    dcmio = DicomIO.DicomIO()
    frame_list = dcmio.ReadDICOM_frames(original_DICOM + os.sep + in_prefix + os.sep + in_dir)
    slice_1st = frame_list[0][0]
    xdim = slice_1st.Columns
    ydim = slice_1st.Rows
    zdim = slice_1st.NumberOfSlices
    tdim = slice_1st.NumberOfTimeSlices
    sample_frame = frame_list[0]
    del frame_list

    # Read data = { 'subwindow': subwindow, 'ROI_No': ROI_No, 'bset': bset, 'ROIslice': ROIslice, 'name': name, 'SIs': SIs }
    if ROIimgs==None:
        img, dim, pmap_names, pmap_slices, pmap_subwindow, bset, name = resolve_ASCIIparamdata(data)
    else:
        img, dim, pmap_names, pmap_slices, pmap_subwindow, bset, name = resolve_ASCIIparamdata(data, ROIimgs[0])
        pmap_subwindow = [0, img.shape[0], 0, img.shape[1]]
        pmap_slices = range(0, img.shape[2])

    print "Pmap image shape:" + str(img.shape)
    print "subwindow:" + str(pmap_subwindow)

    if not os.path.exists(outdir_basename):
        os.makedirs(outdir_basename)
    if not os.path.exists(outdir_basename + os.sep + out_prefix):
        os.makedirs(outdir_basename + os.sep + out_prefix)

    # Save in data in order z,y,x
    out_dirs = []
    for p_i in range(len(pmap_names)):
        out_vols = []
        outvolume = sample_frame
        print "Writing " + pmap_names[p_i]
        for slice_i in range(len(pmap_slices)):
            z_i = pmap_slices[slice_i]-1
            # Initialize slice intensity values
            pixel_array = np.array([[0]*ydim]*xdim, dtype=np.float64)
            #            print pixel_array.shape
            #            print str(len(pmap_SIs[p_i]))
            #            print str(len(pmap_SIs))
            # Place data into slice subregion
            for y_i in range(pmap_subwindow[2], pmap_subwindow[3]):
                for x_i in range(pmap_subwindow[0], pmap_subwindow[1]):
                    pixel_array[y_i, x_i] = float(img[y_i-pmap_subwindow[2], x_i-pmap_subwindow[0], slice_i, p_i])
            # Place data into slice
            max_val = np.power(2,16)-1
            max_pixel_array = np.max(pixel_array)
            min_pixel_array = np.min(pixel_array)
            r_intercept = min_pixel_array
            r_slope = (max_pixel_array-min_pixel_array)/max_val
            if r_slope!=0:
                pixel_array = np.divide(np.subtract(pixel_array, r_intercept), r_slope)
            else:
                pixel_array = np.subtract(pixel_array, r_intercept)
            print (min_pixel_array, max_pixel_array, np.min(pixel_array), np.max(pixel_array), r_intercept, r_slope)
            outvolume[z_i].PixelData = pixel_array.astype(np.uint16).tostring()
            outvolume[z_i].Columns = xdim
            outvolume[z_i].Rows = ydim
            outvolume[z_i].NumberOfSlices = zdim
            outvolume[z_i].NumberOfTimeSlices = 1
            outvolume[z_i].RescaleSlope = r_slope
            outvolume[z_i].RescaleIntercept = r_intercept
        # Append volume to lists
        out_vols.append(outvolume)
        # Create output directory if it does not exist
        out_dir = outdir_basename + os.sep + out_prefix + os.sep + pmap_names[p_i]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        # Write output DICOM
        filenames = dcmio.WriteDICOM_frames(out_dir, out_vols, 'IM')
        out_dirs.append(out_dir)
    return out_dirs, filenames

#
# Convert multi-slice tiff 2 DICOM
#
# in_files   - single TIFF input file (.tiff) for each frame
# dicomdir   - output DICOM directory
# plans      - DICOM header templates for output, frames X slices
# out_prefix - subject specific prefix
#
def singletiff2multidicom(in_files, dicomdir, plans, experiment_dir, out_prefix):
    import DicomIO
    import numpy as np
    import os
    import shutil
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile as tiff
    outdir = experiment_dir + os.sep + out_prefix + os.sep + dicomdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    # Resolve new frame list
    out_vols = plans
    for file_i in range(len(in_files)):
        print "Reading " + in_files[file_i]
        ds = tiff.imread(in_files[file_i])
        no_slices = ds.shape[0]
        for z_i in range(no_slices):
            ds_slice = ds[z_i]
            #ds_slice = ds[z_i]*out_vols[file_i][z_i].RescaleSlope+out_vols[file_i][z_i].RescaleIntercept
            print "Mean before: " + str(np.mean(out_vols[file_i][z_i].pixel_array)*out_vols[file_i][z_i].RescaleSlope+out_vols[file_i][z_i].RescaleIntercept) + " " + str(np.mean(ds_slice))
            ds_slice_min = np.amin(ds_slice)
            ds_intercept = ds_slice_min
            ds_slice = ds_slice - ds_intercept
            ds_range = np.amax(ds_slice)
            ds_slope = ds_range/65535.0
            if ds_slope > 0:
                ds_slice = ds_slice/ds_slope
            ds_slice = ds_slice.astype(np.uint16)
            print "Mean after: " + str(np.mean(ds_slice) * ds_slope + ds_intercept)
            out_vols[file_i][z_i].PixelData = np.round(ds_slice).astype(np.uint16).tostring()
            out_vols[file_i][z_i].RescaleSlope = ds_slope
            out_vols[file_i][z_i].RescaleIntercept = ds_intercept

    dcmio = DicomIO.DicomIO()
    filenames = dcmio.WriteDICOM_frames(outdir, out_vols, 'IM')

    return outdir, filenames

#
# Convert single-slice DICOM (one slice per directory) to one DICOM (all slices and frame in one directory)
#
# in_dirs    - single DICOM input directory for each frame
# dicomdir   - output DICOM directory
# out_prefix - subject specific prefix
#
def multidicom2multidicom(in_dirs, dicomdir, experiment_dir, out_prefix):
    import dicom
    import DicomIO
    import numpy as np
    import os
    import shutil

    outdir = experiment_dir + '/' + out_prefix + '/' + dicomdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    # Resolve new frame list
    out_vols = []
    dcmio = DicomIO.DicomIO()
    for dir_i in range(len(in_dirs)):
        print "Reading directory:" + in_dirs[dir_i]
        frame_list = dcmio.ReadDICOM_frames(in_dirs[dir_i])
        no_slices = len(frame_list[0])
        for z_i in range(no_slices):
            frame_list[0][z_i].NumberOfTimeSlices = len(in_dirs)
        out_vols.append(frame_list[0])
    filenames = dcmio.WriteDICOM_frames(outdir, out_vols, 'IM')
    return outdir, filenames

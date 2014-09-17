#!/usr/bin/env python

import motioncorrection as mc
experiment_dir = mc.experiment_dir

#
# Moves file to results folder, overwriting the existing file
#
# filename   - file to be moved
# out_prefix - subject specific prefix
#
def move_to_results(filename, out_prefix):
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
def dicom2mhd(dicomdir, out_prefix):
    from nipype.utils.filemanip import split_filename
    from nipype.interfaces.base import CommandLine
    import os
    _, name, _ = split_filename(dicomdir)
    outfile_mhd = experiment_dir + '/' + out_prefix + '/' + name + '_tmp/' + 'output' + '.mhd'
    outfile_raw = experiment_dir + '/' + out_prefix + '/' + name + '_tmp/' + 'output' + '.raw'
    outfile_txt = experiment_dir + '/' + out_prefix + '/' + name + '_tmp/' + 'output' + '_info.txt'
    outdir = experiment_dir + '/' + out_prefix + '/' + name + '_tmp'
    cmd = CommandLine('mcverter %s -f meta -o %s -F-PatientName-SeriesDate-SeriesDescription-StudyId-SeriesNumber' % (dicomdir,outdir))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()
    # Move to results folder
    outfile_mhd = move_to_results(outfile_mhd, out_prefix)
    outfile_raw = move_to_results(outfile_raw, out_prefix)
    outfile_txt = move_to_results(outfile_txt, out_prefix)
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
def dicom2nii(dicomdir, out_prefix, out_suffix):
    import os
    import shutil

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

    outfile = move_to_results((dicomdir + '/' + filename_nii + '.gz'), out_prefix)
    outfile_bval = ''
    outfile_bvec = ''
    if len(filename_bval) > 0:
        outfile_bval = move_to_results((dicomdir + '/' + filename_bval + '.bval'), out_prefix)
    if len(filename_bvec) > 0:
        outfile_bvec = move_to_results((dicomdir + '/' + filename_bvec + '.bvec'), out_prefix)

    return outfile, outfile_bval, outfile_bvec

#
# Gunzip (.nii.gz to .nii conversion)
#
# in_file    - input file (.nii.gz)
#
def gznii2nii(in_file):
    import os
    import shutil

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
def nii2nrrd(filename_nii, filename_bval, filename_bvec, out_prefix, out_suffix):
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
def dicom2nrrd(dicomdir, out_prefix, out_suffix):
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

    move_to_results((dicomdir + '/' + filename_nii + '.gz'), out_prefix)
    move_to_results((dicomdir + '/' + filename_bval + '.bval'), out_prefix)
    move_to_results((dicomdir + '/' + filename_bvec + '.bvec'), out_prefix)

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
def nrrd2nii(in_file, output_prefix):
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

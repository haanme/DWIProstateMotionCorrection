# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:52:00 2014

@author: merisaah
"""

#!/usr/bin/env python
output_path = 'results_RMSEchange'
original_path = 'results_Noncorrected'
corrected_path = 'results_Motioncorrected'
mask_matfile_basedir_hB = 'ROI_mat_files_hB'
mask_matfile_basedir_lB = 'ROI_mat_files_lB'
prostatemask_DICOM = r'/Users/eija/Desktop/prostate_MR/Carimasproject_files_Hb_outline_v1013/DICOMmasks'
original_DICOM = r'/Users/eija/Desktop/prostate_MR/PET_MR_dwis'


#
# Add image as subplot
#
# fig           - figure where subplot is added
# plt           - plot for setting x label
# img           - shown image
# xlabel_txt    - xlabel text
# plot_i        - plot index
# pw            - plotting width
# img_cmap      - colormap for image
#
def plot_add_img(fig, plt, img, xlabel_txt, plot_i, pw, img_cmap):
    
    ax = fig.add_subplot(pw, pw, plot_i)
    plot_i += 1
    surf = ax.imshow(img, cmap = img_cmap)
    surf.set_interpolation('nearest')
    plt.xlabel(xlabel_txt, labelpad=-3, fontsize=8)
    ax.set_aspect('equal')
    cb = fig.colorbar(surf, shrink=0.85, aspect=10, orientation='vertical', pad=0.025)
    ax.set_xticklabels([])
    [line.set_markersize(1) for line in ax.xaxis.get_ticklines()]
    ax.set_yticklabels([])
    [line.set_markersize(1) for line in ax.yaxis.get_ticklines()]
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(6)  
    return plot_i
    
#
# Adds subplot to axis
#
# ax              - target axis
# rect            - subregion where subaxis is added
# axisbg          - background color for axis
#
def add_subplot_axes(ax, rect, axisbg='w'):
    import matplotlib.pyplot as plt    
    
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    subax.set_xticklabels([])
    [line.set_markersize(1) for line in subax.xaxis.get_ticklines()]
    subax.set_yticklabels([])
    [line.set_markersize(1) for line in subax.yaxis.get_ticklines()]            
    return subax    

#
# Adds subplot to axis
#
# ax              - target axis
# rect            - subregion where subaxis is added
# axisbg          - background color for axis
#
def add_subplot_axes_with_ticks(ax, rect, axisbg='w'):
    import matplotlib.pyplot as plt    
    
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    subax.set_xticklabels([])
    [line.set_markersize(1) for line in subax.xaxis.get_ticklines()]
    [line.set_markersize(1) for line in subax.yaxis.get_ticklines()]            
    return subax 

#
# Add two neighbouring images as subplot
#
# fig           - figure where subplot is added
# plt           - plot for setting x label
# img_left      - shown image on the left
# img_right     - shown image on the right
# dim           - image dimensions
# pw            - plotting width
# img_cmap      - colormap for image
#
def plot_add_img_pairs(fig, plt, img_left, img_right, dim, pw, img_cmap):
    import matplotlib as mlp

    fig.subplots_adjust(left=0.01)
    fig.subplots_adjust(wspace=0.15)
    axes = []
    subpos = [0.0, 0.0, 0.45, 1.0]
    subpos2 = [0.45, 0.0, 0.45, 1.0]
    # Build axes
    for i in range(pw*pw):
        axis = fig.add_subplot(pw,pw,i+1)
        axis.set_xticklabels([])   
        axis.set_yticklabels([])
        axes.append(axis)
        plt.axis('off')
    # Add images into the axis
    for axis_i in range(dim[2]):
        axis = axes[axis_i]
        [line.set_markersize(0) for line in axis.xaxis.get_ticklines()]
        axis.set_yticklabels([])
        [line.set_markersize(0) for line in axis.yaxis.get_ticklines()]
        axis.set_xlabel('slice ' + ('%04d' % axis_i), labelpad=-3, fontsize=8)        
        
        subax1 = add_subplot_axes(axis,subpos)
        plt.axis('off')
        subax2 = add_subplot_axes(axis,subpos2)
        plt.axis('off')
#        cbar_ax = add_subplot_axes(axis,cbar_pos)

        # Resolve image min/max so that all are the same
        vmax_left, vmin_left = np.max(img_left[:, :, axis_i]), np.min(img_left[:, :, axis_i])
        vmax_right, vmin_right = np.max(img_right[:, :, axis_i]), np.min(img_right[:, :, axis_i])
        vmax = np.max([vmax_left, vmax_right])
        vmin = np.min([vmin_left, vmin_right])
        print str(vmin) + '..' + str(vmax)

        surf = subax1.imshow(img_left[:, :, axis_i], cmap = img_cmap)
        subax1.set_aspect('equal')
        surf.set_interpolation('nearest')
        surf.set_clim(vmin,vmax)
        surf = subax2.imshow(img_right[:, :, axis_i], cmap = img_cmap)
        subax2.set_aspect('equal')
        surf.set_interpolation('nearest')
        surf.set_clim(vmin,vmax)
        # Add colorbar shared between the images
        norm = mlp.colors.Normalize(vmin, vmax)
        cax, kw = mlp.colorbar.make_axes(axis, shrink=0.8, pad=0, anchor=(0.65, 0.7))
        cb = mlp.colorbar.ColorbarBase(cax, cmap = img_cmap, norm = norm)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(6)
    
#
# Add two neighbouring stat displays as subplot
#
# fig           - figure where subplot is added
# plt           - plot for setting x label
# data_left     - shown image on the left
# data_right    - shown image on the right
# dim           - image dimensions
# pw            - plotting width
#
def plot_add_stat_pairs(fig, plt, data_left, data_right, dim, pw):
    import statsmodels.api as sm
    import scipy.stats

    fig.subplots_adjust(top=0.90)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(wspace=0.15)
    fig.subplots_adjust(hspace=.40)
    axes = []
    subpos = [0.0, 0.0, 0.45, 1.0]
    subpos2 = [0.45, 0.0, 0.45, 1.0]
    # Build axes
    for i in range(pw*pw):
        axis = fig.add_subplot(pw,pw,i+1)
        axis.set_xticklabels([])   
        axis.set_yticklabels([])
        axes.append(axis)
        plt.axis('off')
    # Add images into the axis
    for axis_i in range(dim[2]):
        axis = axes[axis_i]
        [line.set_markersize(0) for line in axis.xaxis.get_ticklines()]
        axis.set_yticklabels([])
        [line.set_markersize(0) for line in axis.yaxis.get_ticklines()]
        axis.set_xlabel('slice ' + ('%04d' % axis_i), labelpad=-3, fontsize=8)        

        stat_left = data_left[axis_i]
        stat_right = data_right[axis_i]
        
        # Display boxplots of values side-by-side
        subax1 = add_subplot_axes_with_ticks(axis,subpos)
        #plt.axis('off')        
        bp = plt.boxplot([stat_left, stat_right], notch=True, sym='+', vert=1, whis=1.5)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        xtickNames = plt.setp(subax1, xticklabels=['orig', 'corrected'])
        plt.setp(xtickNames, rotation=0, fontsize=8)        
#        ttest_results = sm.stats.ttest_ind(stat_left, stat_right)
#        plt.title('one-tailed t-test\np=' + ('(%04.3f)' % (ttest_results[1]/2)) + ' N=' + str(len(stat_left)), fontsize=8) 
        U_results = scipy.stats.mannwhitneyu(stat_left, stat_right)
        plt.title('Mann-Whitney U\np=' + ('(%04.3f)' % U_results[1]) + ' N=' + str(len(stat_left)), fontsize=8) 
        # Display regression line side-by-side
        subax2 = add_subplot_axes(axis,subpos2)
        X = sm.add_constant(stat_left)
        est = sm.OLS(stat_right, X)
        est = est.fit()
        str_rsquared = ('R^2:%04.3f' % est.rsquared)
        str_int = ('intercept:%04.3f' % est.params[0]) + ' ' + ('p=(%04.3f)' % est.pvalues[0])
        str_slope = ('slope:%04.3f' % est.params[1]) + ' ' + ('p=(%04.3f)' % est.pvalues[1])
        plt.title(str_rsquared + '\n' + str_int + '\n' + str_slope, fontsize=8) 
        plt.scatter(stat_left, stat_right,  color='black', s=1, alpha=0.8)
        plt.xlabel("orig RMSE", labelpad=-13, fontsize=8)
        plt.ylabel("corrected RMSE", labelpad=-15, fontsize=8)
        X.sort()
        y_hat = est.predict(X)
        plt.plot(X[:, 1], y_hat, 'b', alpha=1.0)
    
#
# Create figure for plotting
#
# titletext        - title text at the top of page 
#
def plot_create_fig(titletext):
    import matplotlib.pyplot as plt    
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=.18)
    fig.subplots_adjust(wspace=.05)
    fig.subplots_adjust(left=0.025)
    fig.subplots_adjust(right=0.975)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.05)    
    fig.suptitle(titletext, fontsize=12)    

    return fig

#
# Resolves image data from ASCII data for use
#
# data             - data structure from ASCII file
#
def resolve_ASCIIimgdata(data, dim):
    import numpy as np 

    values = data['data']
    img = np.ndarray(shape=(dim[0], dim[1], dim[2], values.shape[1]))
    for frame_i in range(values.shape[1]):
        frame_img = values[:, frame_i].reshape([dim[2], dim[0], dim[1]])
        for z in range(dim[2]):
            img[:,:,z,frame_i] = frame_img[z, :, :]
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
#
def resolve_ASCIIparamdata(data):
   
    parameternames = data['parameters']
    slice_numbers = data['ROIslice']
    xy_bounds = data['subwindow']
    bset = data['bset']
    name = data['name']
    values = data['data']
    dim = [xy_bounds[3]-xy_bounds[2], xy_bounds[1]-xy_bounds[0], len(slice_numbers), values.shape[1]]
    img = resolve_ASCIIimgdata(data, dim)
    return img, dim, parameternames, slice_numbers, xy_bounds, bset, name
   
#
# Plot data in ASCII file
#
# filename        - original filename
# data            - data from ASCII results file
# data2           - data from ASCII results file
# data_raw_orig   - raw original data
# data_raw_corr   - raw corrected data
#
def plot_data(filename, data, data2, data_raw_orig, data_raw_corr):
    import matplotlib.pyplot as plt
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    import numpy as np
    from matplotlib import cm
    from matplotlib.backends.backend_pdf import PdfPages

    filename_splitted = filename.split('_')
    studyname = ''.join([item + '_' for item in filename_splitted[:-3]])
    
    pvals_img, dim, parameternames, slice_numbers, xy_bounds, bset, name = resolve_ASCIIparamdata(data)
    pvals_img2 = resolve_ASCIIimgdata(data2, dim)
        
    SI_orig_img = resolve_ASCIIimgdata(data_raw_orig, dim)
    SI_b_0_orig = SI_orig_img[:, :, :, 0]
    SI_b_last_orig = SI_orig_img[:, :, :, -1]
    SI_corr_img = resolve_ASCIIimgdata(data_raw_corr, dim)
    SI_b_0_corr = SI_corr_img[:, :, :, 0]
    SI_b_last_corr = SI_corr_img[:, :, :, -1]

    # Open pdf
    pdf = PdfPages(output_path + os.sep + studyname + 'Differences.pdf')
    pw = int(np.ceil(np.sqrt(dim[2])))

    # Plot SI
    fig = plot_create_fig('Signal Intensity b0 before and after')
    # Add all slices
    plot_add_img_pairs(fig, plt, SI_b_0_orig, SI_b_0_corr, dim, pw, cm.gist_gray)  
    pdf.savefig(fig)
    plt.close()
    fig = plot_create_fig('Signal Intensity b' + ('%d' % bset[-1]) + ' before and after')
    # Add all slices
    plot_add_img_pairs(fig, plt, SI_b_last_orig, SI_b_last_corr, dim, pw, cm.gist_gray)  
    pdf.savefig(fig)
    plt.close()

    # Plot parameter differences
    for pname_i in range(len(parameternames)):
        pval0_r = pvals_img[:, :, :, pname_i]
        pval1_r = pvals_img2[:, :, :, pname_i]
        pval0_r = pval0_r - pval1_r
        
        fig = plot_create_fig('Difference: ' + parameternames[pname_i])
        plot_i = 1
        # Add all slices
        for slice_i in range(dim[2]):
            plot_i = plot_add_img(fig, plt, pval0_r[slice_i, :, :], 'slice ' + ('%04d' % slice_i), plot_i, pw, cm.gist_rainbow_r)                
        # Add sample slices from original SI data
        SI_slice_i = np.ceil(dim[2]/2)
        plot_i = plot_add_img(fig, plt, SI_b_0_orig[SI_slice_i, :, :], 'orig b0 slice '  + ('%04d' % SI_slice_i), plot_i, pw, cm.gist_gray)
        plot_i = plot_add_img(fig, plt, SI_b_last_orig[SI_slice_i, :, :], 'orig b' + ('%d' % bset[-1]) + ' slice ' + ('%04d' % SI_slice_i), plot_i, pw, cm.gist_gray)
        # Add sample slices fro corrected SI data
        plot_i = plot_add_img(fig, plt, SI_b_0_corr[SI_slice_i, :, :], 'corr b0 slice ' + ('%04d' % SI_slice_i), plot_i, pw, cm.gist_gray)
        plot_i = plot_add_img(fig, plt, SI_b_last_corr[SI_slice_i, :, :], 'corr b' + ('%d' % bset[-1]) + ' slice ' + ('%04d' % SI_slice_i), plot_i, pw, cm.gist_gray)
        
        pdf.savefig(fig)
        plt.close()
    pdf.close()

#
# Plot data in ASCII file
#
# filename        - original filename
# data            - data from ASCII results file
# data2           - data from ASCII results file
# data_raw_orig   - raw original data
# data_raw_corr   - raw corrected data
# ROIimgs         - ROI images
# ROInames        - ROInames
#
def plotdataROI(filename, data, data2, data_raw_orig, data_raw_corr, ROIimgs, ROInames):
    import matplotlib.pyplot as plt
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    import numpy as np
    from matplotlib import cm
    from matplotlib.backends.backend_pdf import PdfPages

    filename_splitted = filename.split('_')
    studyname = ''.join([item + '_' for item in filename_splitted[:-3]])
    
    pvals_img, dim, parameternames, slice_numbers, xy_bounds, bset, name = resolve_ASCIIparamdata(data)
    pvals_img2 = resolve_ASCIIimgdata(data2, dim)    
        
    SI_orig_img = resolve_ASCIIimgdata(data_raw_orig, dim)
    SI_b_0_orig = SI_orig_img[:, :, :, 0]
    SImax = np.max(SI_b_0_orig[:, :, :])
    SImin = np.min(SI_b_0_orig[:, :, :])
           
    # Open pdf
    #pdf = PdfPages(output_path + os.sep + studyname + 'ROI_QC.pdf')
    pw = int(np.ceil(np.sqrt(dim[2])))

    for ROI_i in range(len(ROInames)):
        SI_b_0_orig_ROI = np.ndarray(shape=(dim[0], dim[1], dim[2]))
        data_left = []
        data_right = []
        for z in range(SI_b_0_orig_ROI.shape[2]):
            pval0_r = []
            pval1_r = []            
            for y in range(SI_b_0_orig_ROI.shape[1]):
                for x in range(SI_b_0_orig_ROI.shape[0]):
                    if ROIimgs[ROI_i][x, y, z] != 0:
                        SI_b_0_orig_ROI[x, y, z] = SImax
                        pval0_r.append(pvals_img[x, y, z, -1])
                        pval1_r.append(pvals_img2[x, y, z, -1])
                    else:
                        SI_b_0_orig_ROI[x, y, z] = SI_b_0_orig[x, y, z]                   
            data_left.append(pval0_r)
            data_right.append(pval1_r)
        SImaxROI = np.max(SI_b_0_orig_ROI[:, :, :])
        SIminROI = np.min(SI_b_0_orig_ROI[:, :, :])
        # Plot SI
#        fig = plot_create_fig('Signal Intensity b0 with ROI')
#        plot_add_img_pairs(fig, plt, SI_b_0_orig, SI_b_0_orig_ROI, dim, pw, cm.gist_gray)  
        # Ass statistics of values inside ROIs
        fig = plot_create_fig('RMSE comparison inside ROI')
        plot_add_stat_pairs(fig, plt, data_left, data_right, dim, pw)
        #pdf.savefig(fig)
        #plt.close()
    #pdf.close()

#
# Resolves image data in DICOM containing ROIs
#
# DICOMpath           - DICOMpath containing ROI data
# dim                 - data dimensions
# xy_bounds           - bounds of ROI in original full image
# z_slices            - z-slices 
#
def resolve_DICOMROI_imgdata(DICOMpath, xy_bounds, z_slices):
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
        ROI = np.zeros(shape=(xy_bounds[1]-xy_bounds[0], xy_bounds[3]-xy_bounds[2], dim[2]))
        for z in range(dim[2]):
            ROI[:,:,z] = DICOM_ROIdata[roi_i][z].pixel_array[xy_bounds[0]:xy_bounds[1], xy_bounds[2]:xy_bounds[3]]
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
        print ROIpixel_array.shape
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
    #    if not (bset_str=='hB' or bset_str=='lB'):
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
    print (prostatemask_DICOM + os.sep + patient_no_str + '_' + patientname_str + '_*')
    paths = glob.glob((prostatemask_DICOM + os.sep + patient_no_str + '_' + patientname_str + '_*'))
    print paths
    return paths[0]
    
#
# Convert ASCII fitting results to DICOM
#
# data         - data from ASCII input file
# in_dir       - DICOM directory for reference headers
# out_prefix   - patient subdir
# bounds       - box coordinates of DICOM inside the original DICOM data [xmin, xmax, ymin, ymax, zmin, zmax]
#
def ASCII2DICOM(data, in_dir, outdir_basename, out_prefix, bounds):
    import dicom
    import DicomIO
    import numpy as np
    import os
    import shutil

    # Resolve new frame list
    dcmio = DicomIO.DicomIO()
    frame_list = dcmio.ReadDICOM_frames(original_DICOM + os.sep + out_prefix + os.sep + in_dir)
    slice_1st = frame_list[0][0]
    xdim = slice_1st.Columns
    ydim = slice_1st.Rows
    zdim = slice_1st.NumberOfSlices
    tdim = slice_1st.NumberOfTimeSlices
    sample_frame = frame_list[0]
    del frame_list

    # Read data = { 'subwindow': subwindow, 'ROI_No': ROI_No, 'bset': bset, 'ROIslice': ROIslice, 'name': name, 'SIs': SIs }
    img, dim, pmap_names, pmap_slices, pmap_subwindow, bset, name = resolve_ASCIIparamdata(data)

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
            pixel_array = np.divide(np.subtract(pixel_array, r_intercept), r_slope)
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
    
from argparse import ArgumentParser
import sys
import os
import DicomIO
import conversions as conv
import time
import numpy as np
import bfitASCII_IO
import glob

if __name__ == "__main__":
#    parser = ArgumentParser()
#    parser.add_argument("--dicomdir", dest="dicomdir", help="dicomdir", required=True)
#    args = parser.parse_args()

    errors = 0

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filenames_raw = glob.glob('*ASCII.txt')
    filenames_orig = os.listdir(original_path)
    filenames_corr = os.listdir(corrected_path)

    ASCIIio = bfitASCII_IO.bfitASCII_IO()
    for fname_raw in filenames_raw:
        splitted_raw = fname_raw.split('_')
        splitted_raw = splitted_raw[:-2]
        # Reach for non-corrected fitted file
        found_fname_orig = ''
        for fname_orig in filenames_orig:
            splitted_orig = fname_orig.split('_')
            splitted_orig = splitted_orig[:-4]
            if splitted_orig == splitted_raw:
                found_fname_orig = fname_orig
                break
        # Reach for corrected fitted file
        found_fname_corr = ''                
        for fname_corr in filenames_corr:
            splitted_corr = fname_corr.split('_')
            splitted_corr = splitted_corr[:-4]
            if splitted_corr == splitted_raw:
                found_fname_corr = fname_corr
                break
        # Collect data for plotting                    
        data_orig = ASCIIio.Read((original_path + os.sep + found_fname_orig), False)
        data_orig_dim = resolve_ASCIIdimdata(data_orig)
        data_orig_slice_numbers = data_orig['ROIslice']
        data_orig_xy_bounds = data_orig['subwindow']
        data_bounds = [data_orig_xy_bounds[0], data_orig_xy_bounds[1], data_orig_xy_bounds[2], data_orig_xy_bounds[3], data_orig_slice_numbers[0], data_orig_slice_numbers[-1]]
        data_bounds = np.subtract(data_bounds, 1)
        print data_bounds
        data_corr = ASCIIio.Read((corrected_path + os.sep + found_fname_corr), False)
        patient_name = ''.join([item + '_' for item in splitted_raw])
        patient_name = patient_name[0:-1]
        print patient_name
        fname_raw_orig = original_path + os.sep + patient_name + '_Noncorrected_ASCII_Biexp_results.txt'
        fname_raw_corr = corrected_path + os.sep + patient_name + '_Motioncorrected_ASCII_Biexp_results.txt'
        data_raw_orig = ASCIIio.Read((fname_raw_orig), True)
        data_raw_corr = ASCIIio.Read((fname_raw_corr), True)

        # Read DICOM mask data        
        prefix = ''.join([item + '_' for item in splitted_raw])
        DICOMpath = resolve_DICOMpath(prefix)
        if not os.path.exists(DICOMpath):
            print (DICOMpath + " DOES NOT EXIST")
            continue
        ROIpixel_array_all, ROInames = resolve_DICOMROI_imgdata(DICOMpath, data_orig_xy_bounds, data_orig_slice_numbers)

        ROIpixel_array_all = [np.ones(shape=(data_orig_dim[0], data_orig_dim[1], data_orig_dim[2]))]
        ROInames = ['ALLvoxels']


        # Do plotting of ROI position, displacement-RMSE correlation, RMSE boxplot inside ROI
        plotdataROI(fname_orig, data_corr, data_orig, data_raw_orig, data_raw_corr, ROIpixel_array_all, ROInames)
        # Do plotting of voxelwise parameter change
        plot_data(fname_orig, data_corr, data_orig, data_raw_orig, data_raw_corr)
        # Write parametric map
        ASCII2DICOM(data_raw_orig, 'DICOMconverted', 'results_DICOM_pmaps', patient_name, data_bounds)
        
        break

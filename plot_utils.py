# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:52:00 2014

@author: merisaah
"""

import numpy as np

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
# vmin, vmax    - scaling min and max, default 0 and 0 to rescale according to data
#
def plot_add_img_pairs(fig, plt, img_left, img_right, dim, slice_Nos, pw, img_cmap, vmin=0, vmax=0):
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
        axis.set_xlabel('slice ' + ('%04d' % slice_Nos[axis_i]), labelpad=-3, fontsize=8)
        
        subax1 = add_subplot_axes(axis,subpos)
        plt.axis('off')
        subax2 = add_subplot_axes(axis,subpos2)
        plt.axis('off')
#        cbar_ax = add_subplot_axes(axis,cbar_pos)

        # Resolve image min/max so that they are all the same
        if vmin == 0 and vmax == 0:
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
        plt.title('slice ' + ('%04d' % slice_Nos[axis_i]), fontsize=8)
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
    fig.subplots_adjust(bottom=0.01)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(wspace=0.15)
    fig.subplots_adjust(hspace=.6)
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
    print (dim, len(axes))
    for axis_i in range(dim[2]):
        print (dim, len(axes))
        axis = axes[axis_i]
        [line.set_markersize(0) for line in axis.xaxis.get_ticklines()]
        axis.set_yticklabels([])
        [line.set_markersize(0) for line in axis.yaxis.get_ticklines()]
        axis.set_xlabel('slice ' + ('%04d' % axis_i), labelpad=-3, fontsize=6)

        stat_left = data_left[axis_i]
        stat_right = data_right[axis_i]
        
        # Display boxplots of values side-by-side
        subax1 = add_subplot_axes_with_ticks(axis,subpos)
        for t in subax1.get_yticklabels():
            t.set_fontsize(6)
        if len(stat_left) == 0:
            plt.title('No data', fontsize=4)
            continue
        
        #plt.axis('off')        
        bp = plt.boxplot([stat_left, stat_right], notch=True, sym='+', vert=1, whis=1.5)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        xtickNames = plt.setp(subax1, xticklabels=['orig', 'corrected'])
        plt.setp(xtickNames, rotation=0, fontsize=6)
#        ttest_results = sm.stats.ttest_ind(stat_left, stat_right)
#        plt.title('one-tailed t-test\np=' + ('(%04.3f)' % (ttest_results[1]/2)) + ' N=' + str(len(stat_left)), fontsize=8) 
        U_results = scipy.stats.mannwhitneyu(stat_left, stat_right)
        if len(stat_left) > 0:
            plt.title('Mann-Whitney U\np=' + ('(%04.3f)' % U_results[1]) + ' N=' + str(len(stat_left)), fontsize=4)
        else:
            plt.title('N=' + str(len(stat_left)), fontsize=4)

        # Display regression line side-by-side
        if len(stat_left) > 0:
            subax2 = add_subplot_axes(axis,subpos2)
            X = sm.add_constant(stat_left)
            est = sm.OLS(stat_right, X)
            est = est.fit()
            str_rsquared = ('R^2:%04.3f' % est.rsquared)
            str_int = ('int:%04.3f' % est.params[0]) + ' ' + ('p=(%04.3f)' % est.pvalues[0])
            str_slope = ('slope:%04.3f' % est.params[1]) + ' ' + ('p=(%04.3f)' % est.pvalues[1])
            plt.title(str_rsquared + '\n' + str_int + '\n' + str_slope, fontsize=4)
        else:
            plt.title('N/A' + '\n' + 'N/A' + '\n' + 'N/A', fontsize=4)
        plt.scatter(stat_left, stat_right,  color='black', s=1, alpha=0.8)
        plt.xlabel("orig RMSE", labelpad=-10, fontsize=4)
        plt.ylabel("corrected RMSE", labelpad=-12, fontsize=4)

        if len(stat_left) > 0:
            X.sort()
            y_hat = est.predict(X)
            plt.plot(X[:, 1], y_hat, 'b', alpha=1.0)
        else:
            plt.plot([0],[0], 'b', alpha=1.0)
    
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

    filename_splitted = filename.split(os.sep)
    filename_path = filename_splitted[:-2]
    filename_base = filename_splitted[-1]
    filename_base = filename_base.replace('.','_')
    filename_splitted = filename_base.split('_')
    print filename_splitted
    studyname = ''.join([item + '_' for item in filename_splitted[:-2]])
    studyname = studyname.replace('.','_')

    # Open pdf
    if not os.path.exists(output_path + os.sep + filename_base):
        os.makedirs(output_path + os.sep + filename_base)
    pdf = PdfPages(output_path + os.sep + filename_base + os.sep + studyname + 'ROI_QC.pdf')
    for ROI_i in range(len(ROInames)):
        pvals_img, dim, parameternames, slice_numbers, xy_bounds, bset, name = resolve_ASCIIparamdata(data, ROIimgs[ROI_i])
        pvals_img2 = resolve_ASCIIimgdata(data2, dim, ROIimgs[ROI_i])

        SI_orig_img = resolve_ASCIIimgdata(data_raw_orig, dim, ROIimgs[ROI_i])
        SI_b_0_orig = SI_orig_img[:, :, :, 0]
        SImax = np.max(SI_b_0_orig[:, :, :])
        SImin = np.min(SI_b_0_orig[:, :, :])

        SI_b_0_orig_ROI = np.ndarray(shape=(dim[0], dim[1], dim[2]))
        data_left = []
        data_right = []
        print SI_b_0_orig_ROI.shape
        print ROIimgs[ROI_i].shape
        ROI_voxels = 0
        for z in range(SI_b_0_orig_ROI.shape[2]):
            pval0_r = []
            pval1_r = []
            for y in range(SI_b_0_orig_ROI.shape[1]):
                for x in range(SI_b_0_orig_ROI.shape[0]):
                    if ROIimgs[ROI_i][x, y, z] != 0:
                        SI_b_0_orig_ROI[x, y, z] = SImax
                        pval0_r.append(pvals_img[x, y, z, -1])
                        pval1_r.append(pvals_img2[x, y, z, -1])
                        ROI_voxels += 1
                    else:
                        SI_b_0_orig_ROI[x, y, z] = SI_b_0_orig[x, y, z]                   
            data_left.append(pval0_r)
            data_right.append(pval1_r)
        print str(ROI_voxels) + ' voxels found' 
        SImaxROI = np.max(SI_b_0_orig_ROI[:, :, :])
        SIminROI = np.min(SI_b_0_orig_ROI[:, :, :])
        # Plot SI
#        fig = plot_create_fig('Signal Intensity b0 with ROI')
#        plot_add_img_pairs(fig, plt, SI_b_0_orig, SI_b_0_orig_ROI, dim, pw, cm.gist_gray)  
        # Ass statistics of values inside ROIs
        fig = plot_create_fig('RMSE comparison inside ROI')
        print len(data_left)
        print len(data_right)
        print dim
        pw = int(np.ceil(np.sqrt(dim[2])))
        plot_add_stat_pairs(fig, plt, data_left, data_right, dim, pw)
        pdf.savefig(fig)
        plt.close()
    pdf.close()

#
# Plots iteration info into pdf
#
# loginfo             - information file
# out_dir             - output directory
#
def plot_iterationinfo(loginfo, out_dir):
    import matplotlib.pyplot as plt
    import skimage.io
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(out_dir + os.sep + 'convergence_QC.pdf')

    # Go through resolutions
    no_resolutions = len(loginfo)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.75)
    fig.subplots_adjust(wspace=.01)
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.1)
    fig.suptitle('Deformation convergence', fontsize=12)
    print str(no_resolutions) + ' resolutions'
    for resolution_i in range(no_resolutions):
        resolution = loginfo[resolution_i]['data']
        ax = fig.add_subplot(no_resolutions, 1, resolution_i+1)
        x1 = np.array(range(1,resolution.shape[0]+1)).T
        ax.plot(x1, resolution[:,1].T, 'yo-')
        ax.set_xlabel((loginfo[resolution_i]['filename'] + '(' + loginfo[resolution_i]['metric'] + '):' + loginfo[resolution_i]['endcondition']), fontsize=8, labelpad=0)
        ax.set_ylabel('Metric', labelpad=0.1, fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    pdf.savefig(fig)
    pdf.close()

#
# Plot QC of one slice
#
# fig         - current figure where QC in plotted
# imslice     - slice image containing displacement values in x,y,z directions
# shape       - shape of original displacement field image z,x,y,d, where d is displacement [u,v,w]
# plot_i      - current plot index in QC subplots
# no_slices   - total number of slices that are going to be QC'd
# dwislice    - DWI slice for overlay purposes
#
def plot_slice(fig, imslice, shape, plot_i, no_slices, dwislice):
    import scipy.ndimage
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import warnings

    ax = fig.add_subplot(no_slices, 5, plot_i+1)
    X = np.linspace(0, shape[2], shape[2])
    Y = np.linspace(0, shape[1], shape[1])
    X, Y = np.meshgrid(X, Y)
    zfactor = 0.15

    # Catch user warning about changing result size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = scipy.ndimage.zoom(X, zfactor, order=0)
        Y = scipy.ndimage.zoom(Y, zfactor, order=0)
        imsliceRu = scipy.ndimage.zoom(imslice[:,:,0], zfactor, order=2)
        imsliceRv = scipy.ndimage.zoom(imslice[:,:,1], zfactor, order=2)
        imsliceRw = scipy.ndimage.zoom(imslice[:,:,2], zfactor, order=2)
    u = imsliceRu
    v = imsliceRv
    w = imsliceRw
    dir(ax.quiver)
    ax.quiver(X,Y,w,u,v,w)
    ax.set_aspect('equal')
    plt.axis('off')

    ydim = X.shape[0]
    xdim = X.shape[1]
    Xw = np.ndarray(shape=X.shape)*0
    Yw = np.ndarray(shape=X.shape)*0
    Zw = np.ndarray(shape=X.shape)*0
    for i in range(ydim):
        for j in range(xdim):
            Xw[i,j] = X[i,j] + imsliceRu[i,j]*(1/zfactor)
            Yw[i,j] = Y[i,j] + imsliceRv[i,j]*(1/zfactor)
            Zw[i,j] = 0 + imsliceRw[i,j]*0
    Xd = np.absolute(imslice[:,:,0])
    Yd = np.absolute(imslice[:,:,1])
    Zd = np.absolute(imslice[:,:,2])
    print 'Mean:' + str(np.mean(Xd)) + ' SD:' + str(np.std(Xd)) + ' Max:' + str(np.ma.max(Xd)) + ' Mean:' + str(np.mean(Yd)) + ' SD:' + str(np.std(Yd)) + ' Max:' + str(np.ma.max(Yd)) + ' Mean:' + str(np.mean(Zd)) + ' SD:' + str(np.std(Zd)) + ' Max:' + str(np.ma.max(Zd))
    ax = fig.add_subplot(no_slices, 5, plot_i+2)
    ax.imshow(dwislice, cmap = cm.Greys_r)
    ax.plot(Xw, Yw,'b-', linewidth=0.5)
    ax.plot(Xw.T, Yw.T,'b-', linewidth=0.5)
    ax.set_aspect('equal')
    plt.axis('off')

    plot_displacement_slice(fig, imslice[:,:,0], no_slices, plot_i + 3, 'X-Displacement in voxels', Xd)
    plot_displacement_slice(fig, imslice[:,:,1], no_slices, plot_i + 4, 'Y-Displacement in voxels', Yd)
    plot_displacement_slice(fig, imslice[:,:,2], no_slices, plot_i + 5, 'Z-Displacement in voxels', Zd)

    plot_i = plot_i + 5
    return plot_i

#
# Plot displacement slice for QC purposes
#
# fig          - current figure where QC is plotted
# slice_img    - slice image containing displacement values
# no_slices    - total number of slices to be QC'd
# plot_i       - current plot index in QC subplots
# title        - title printed as y-axis label
# Dvals        - displacement values for printing statistics on x-axis label
#
def plot_displacement_slice(fig, slice_img, no_slices, plot_i, title, Dvals):
    import matplotlib.pyplot as plt

    ax = fig.add_subplot(no_slices, 5, plot_i)
    surf = ax.imshow(slice_img)
    surf.set_clim(-1.0,1.0)
    plt.ylabel(title, fontsize=5, labelpad=-13.0)
    plt.xlabel('$\mu=$' + ('%.3f' % np.mean(Dvals)) + '\n$\sigma=$' + ('%.3f' % np.std(Dvals)) + '\n$max=$' + ('%.3f' % np.ma.max(Dvals)), fontsize=4, labelpad=0)
    cb = fig.colorbar(surf, shrink=0.65, aspect=10, orientation='vertical', pad=0.025)
    ax.set_xticklabels([])
    [line.set_markersize(3) for line in ax.xaxis.get_ticklines()]
    ax.set_yticklabels([])
    [line.set_markersize(3) for line in ax.yaxis.get_ticklines()]
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(4)

#
# Create coordinates file
#
# input_files         - elastix transformation tiff file with RGB channels for x,y,z directions
# parameter_files     - elastix co-registration parameters file
# output_prefix       - output prefix
# output_sub_prefixes - output subfolder prefix
# DWIimgs             - DWIimage for overlay purposes
#
def plot_deformation_vectors(input_files, parameter_files, output_prefix, output_sub_prefixes, DWIimgs, experiment_dir):
    import matplotlib.pyplot as plt
    import skimage.io
    import os
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = experiment_dir + '/' + output_prefix
    im = skimage.io.imread(input_files[0], plugin='tifffile')
    shape = im.shape
    no_slices = shape[0]
    used_slices = range(shape[0])
    no_slices_per_page = 3
    pdf = PdfPages(out_dir + os.sep + 'deformation_QC.pdf')
    print shape

    # Go through slices
    for used_slice_i in range(len(used_slices)):
        slice_i = int(used_slices[used_slice_i])
        # Go through b-values
        for b_value in range(len(input_files)):
            print str(slice_i) + ' frame ' + str(b_value)
            # Start writing data for new page
            if np.mod(b_value, 3) == 0:
                fig = plt.figure()
                fig.subplots_adjust(hspace=.01)
                fig.subplots_adjust(wspace=.1)
                fig.subplots_adjust(left=0.05)
                fig.subplots_adjust(right=0.95)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(bottom=0.01)
                plot_i = 0
                slice_str = ''
            im = skimage.io.imread(input_files[b_value], plugin='tifffile')
            imslice = im[slice_i,:,:,:]
            plot_i = plot_slice(fig, imslice, shape, plot_i, no_slices_per_page, DWIimgs[b_value][slice_i].pixel_array)
            slice_str = slice_str + ' ' + ('%.0f' % b_value)
            # Save data to figure
            if np.mod(b_value, 3) == 2:
                fig.suptitle('[' + output_prefix + '] Deformation QC for slice ' + str(slice_i) + ' frames ' + slice_str, fontsize=12)
                pdf.savefig(fig)
                #plt.show()
                plt.close()
        # Save data to figure if last round did not involve saving
        if np.mod(shape[3]-1, 3) != 2:
            fig.suptitle('[' + output_prefix + '] Deformation QC for slice ' + str(slice_i) + ' frames ' + slice_str, fontsize=12)
            pdf.savefig(fig)
            #plt.show()
            plt.close()
    pdf.close()



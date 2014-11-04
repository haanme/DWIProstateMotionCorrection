# -*- coding: utf-8 -*-
"""
    Created on Fri Apr 11 20:20:02 2014

    @author: merisaah
    """
import os
import numpy as np

class bfitASCIIError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class bfitASCIIReadError(bfitASCIIError):
    def __str__(self):
        return repr('Read error ' + self.value)


class bfitASCIIWriteError(bfitASCIIError):
    def __str__(self):
        return repr('Write error ' + self.value)

class bfitASCII_IO:

    def Write3D(self, path, data, SI_file=True):

        subwindow = data['subwindow']
        if data.has_key('ROI_no'):
            ROI_No = data['ROI_No']
        else:
            ROI_No = data['number']
        bset = data['bset']
        ROIslice = data['ROIslice']
        name = data['name']
        if data.has_key('SIs'):
            SIs = data['SIs']
        else:
            SIs = data['data']
        if not SI_file:
            executiontime = data['executiontime']
            description = data['description']
            parameters = data['parameters']

        f = open(path, 'w')
        # write header information
        f.write('subwindow: [%d %d %d %d]\n' % (subwindow[0], subwindow[1], subwindow[2], subwindow[3]))
        f.write('number: %d\n' % ROI_No)
        f.write('bset: [')
        for i in range(len(bset)):
            f.write('%d ' % bset[i])
        f.write(']\n')
        f.write('ROIslice: [')
        for i in range(len(ROIslice)):
            f.write('%d ' % ROIslice[i])
        f.write(']\n')
        f.write('name: %s\n' % name)
        if not SI_file:
            f.write('executiontime: %d seconds\n' % executiontime)
            f.write('description: %s\n' % description)
            f.write('parameters: %s\n' % parameters)
            data_length = len(SIs)
            print_step = data_length/10
            bset_length = SIs.shape[1]
            for i in range(len(SIs)):
                for j in range(bset_length):
                    f.write('%.15f ' % SIs[i][j])
                f.write('\n')
                if(np.mod(i,print_step) == 0):
                    print ('writing %d/%d' % (i+1, data_length))
            print ('writing %d/%d' % (data_length, data_length))
        else:
            f.write('SIs: \n')
            # write SI data
            data_length = len(SIs)
            print_step = data_length/10
            bset_length = len(bset)
            for i in range(len(SIs)):
                for j in range(bset_length):
                    f.write('%.15f ' % SIs[i][j])
                f.write('\n')
                if(np.mod(i,print_step) == 0):
                    print ('writing %d/%d' % (i+1, data_length))
            print ('writing %d/%d' % (data_length, data_length))
        f.close()

    def Read(self, path, SI_file):

        f = open(path)
        lines = f.readlines()
        f.close()
        outdata = {'data':[]}
        for line_i in range(len(lines)):
            line = lines[line_i]
            # resolve subwindow
            if line.find('subwindow') == 0:
                subs = line.split()
                subs = subs[1:]
                subs[0] = subs[0].lstrip('[')
                subs[-1] = subs[-1].rstrip(']')
                if len(subs[-1]) == 0:
                    subs = subs[:-1]
                outdata['subwindow'] = [int(float(subs[0])), int(float(subs[1])), int(float(subs[2])), int(float(subs[3]))]
                continue
            # resolve bset
            if line.find('bset') == 0:
                subs = line.split()
                subs = subs[1:]
                subs[0] = subs[0].lstrip('[')
                subs[-1] = subs[-1].rstrip(']')
                if len(subs[-1]) == 0:
                    subs = subs[:-1]
                bset = [int(float(s)) for s in subs]
                outdata['bset'] = bset
                continue
            # resolve parameters
            if line.find('parameters') == 0:
                subs = line.split()
                subs = subs[1:]
                outdata['parameters'] = subs
                continue
            # resolve ROi slice numbers
            if line.find('ROIslice') == 0:
                subs = line.split()
                subs = subs[1:]
                subs[0] = subs[0].lstrip('[')
                subs[-1] = subs[-1].rstrip(']')
                if len(subs[-1]) == 0:
                    subs = subs[:-1]
                ROIslice = [int(float(s)) for s in subs]
                outdata['ROIslice'] = ROIslice
                continue
            # resolve description
            if line.find('description') == 0:
                subs = line.split(':')
                subs = subs[1:]
                outdata['description'] = ':'.join(subs).strip()
                continue
            # resolve name
            if line.find('name') == 0:
                subs = line.split(':')
                subs = subs[1:]
                outdata['name'] = subs[0].strip()
                continue
            # resolve name
            if line.find('executiontime') == 0:
                subs = line.split(':')
                subs = subs[1:]
                outdata['executiontime'] = subs[0].strip()
                continue
            # resolve number
            if line.find('number') == 0:
                subs = line.split(':')
                subs = subs[1:]
                outdata['number'] = int(float(subs[0]))
                continue
            if (SI_file and line_i < 6) or (not SI_file and line_i < 8):
                #print "continue " + line
                continue
            #print "number str " + line
            # resolve parameter values
            outdata['data'].append([float(s) for s in line.split()])
        outdata['data'] = np.array(outdata['data'])
        return outdata

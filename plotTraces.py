#!/usr/bin/python

import sys
import analysisTools
import matplotlib.pyplot as plt
import numpy as np
import os
import npy2timeStamp
import glob

'''
UPDATED 10 November 2016

* if filename is NOT provided, this script will look for xyDataWithTimeStamps.npy
and plot that

* if filename(s) provided, it will estimate timestamps and plot
note: you can specify a single file (e.g. xy161018-151646)
or a range of files that share a common prefix (e.g. xy161018-15)
do NOT include the .npy in your filename

USAGE: 
python plotTraces.py  (<-- this looks for xyDataWithTimeStamps.npy)
	OR
python plotTraces.py fileStem

the fileStem argument is the name of the file to plot (WITHOUT the .npy suffix)
e.g. xy161018 or xy161018-15 or xy161018-152312


'''

#### Loading Data ###################################################################
if len(sys.argv) < 2:
    filenames = glob.glob('xyData*.npy')
    if len(filenames) > 0:
        print('loading %s . . . ' % filenames[0])
        data = analysisTools.loadData(filenames[0][:-4])
    else:
        exit('Usage: = %s nameOfFile (no .npy at end!)' % sys.argv[0])
else:
    fileStem = sys.argv[1]
    # add estimated timeStamps if needed
#     npy2timeStamp.main(fileStem)  
    data = analysisTools.loadData(fileStem)
#     os.remove('xyDataWithTimeStamps.npy') # need this if run npy2timeStamp


#### Plots for XY data ###############################################################

'''
show the path of objects in all ROI's
this is only useful for smaller time frames (~20 minutes)
'''
analysisTools.showAllTraces(data)



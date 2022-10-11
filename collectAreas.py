#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:43:09 2022

@author: iwoods

I want to compare areas as measured by the tracking program.
One issue now is that the area for each clip (within a longer video) is 
calculated by the median of the measured areas of that clip. 

But I want to find the median area of ALL clips for a particular video. 
These are all saved as filenames that look like this:
	AC_Sep29_tricaine_tardigrade2-2_areas.csv

The example above is Asenka (AC), tricaine treated, tardigrade 2, clip 2

This script compiles areas from each clip and then gets median of ALL 
measured areas for each video.

"""

import glob
import numpy as np

areaFiles = glob.glob('*areas.csv')

areaDict = {}

# collect all the areas
for file in sorted(areaFiles):
    # read in areas
    f = open(file,'r')
    lines = f.readlines()
    f.close()
    areas = np.array([float(x.rstrip()) for x in lines])
    
    # get name of video that contains this clip
    vidFile = file.split('_areas')[0][:-2]

    # add areas to a dictionary, with the key as the video name
    if vidFile in areaDict.keys():
        areaDict[vidFile] = np.hstack((areaDict[vidFile], areas))
    else:
        areaDict[vidFile] = areas
    
# print out medians
for video in sorted(areaDict.keys()):
    print(video + ',' + str(np.median(areaDict[video])))

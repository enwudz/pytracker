#!/usr/bin/env python3

import trackingTools
import sys

numROI = 24 # 1 or gridded, e.g. 6, 12, 24, 48, 96
roiShape = 'c' # 'r' or 'c' for rectangle or circle

'''
this script sets up and saves an ROI mask
as mask.png

User draws rectangular ROI
Can then choose to partition into a grid of rectangles or circles

Drawing tips:
start on upper left, drag to lower right
do not reverse the motion of your mouse (i.e. always move DOWN and RIGHT)
do not let rectangle go past the boundary of the screen
if you don't get the ROI you want, try again.
'''

vidSource, vidType = trackingTools.getVideoStream(sys.argv)
trackingTools.defineRectangleROI(vidSource,numROI,roiShape)

trackingTools.showROI()
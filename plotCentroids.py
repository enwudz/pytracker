#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:22:50 2022

@author: iwoods
"""

'''
Plot coordinates from trackCritter
smoothed?
linked to time (need fps) somehow?
'''

import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cv2

def main(centroid_file):
    
    # get height, width, fps from filestem
    movie_file = centroid_file.split('_centroids')[0] + '.mov'
    (vid_width, vid_height, vid_fps, vid_frames) = getVideoData(movie_file)
    print(vid_width, vid_height, vid_fps)
    
    xcoords = []
    ycoords = []
    
    # read in data
    df = pd.read_csv(centroid_file, names = ['x','y'], header=None)
    
    xcoords = df.x.values
    ycoords = df.y.values
    
    # smooth the data?
    
    # plot (scatter) colormap
    cmap_name = 'viridis'
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(1,0,len(xcoords)))
    
    figsize = (vid_height/200, vid_width/200)
    f, a = plt.subplots(1)
    sc = a.scatter(vid_width-xcoords,vid_height-ycoords, c = cols)
    a.set_xlim([0, vid_width])
    a.set_ylim([0, vid_height])
    a.set_xticks([])
    a.set_yticks([])
    # add legend for time
    norm = mpl.colors.Normalize(vmin=0, vmax=vid_frames / vid_fps)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label = 'Time (sec)')
    plt.show()
    
    
    
    return

def getVideoData(videoFile):
    if len(glob.glob(videoFile)) == 0:
        exit('Cannot find ' + videoFile)
    else:
        vid = cv2.VideoCapture(videoFile)
        vid_width  = int(vid.get(3))
        vid_height = int(vid.get(4))
        vid_fps = int(np.round(vid.get(5)))
        vid_frames = int(vid.get(7))
        vid.release()
    return (vid_width, vid_height, vid_fps, vid_frames)


if __name__ == "__main__":
    
    
    if len(sys.argv) > 1:
        centroid_file = sys.argv[1]
    else:
        centroid_list = glob.glob('*centroid*')
        centroid_file = centroid_list[0]
        
    print('Getting data from ' + centroid_file)

    main(centroid_file)

    
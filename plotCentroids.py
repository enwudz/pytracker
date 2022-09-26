#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:22:50 2022

@author: iwoods
"""

import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cv2
import scipy.signal

'''
WISH LIST
DONE Display the last frame of the video, with color-coded centroid path DONE

On side, add text with video name, object area, distance, speed
    ^^^ or as x axis label?

DONE So, need to code to calculate speed and distance. May want to SMOOTH before speed.

'''

def main(centroid_file):
    
    # get height, width, fps from filestem
    movie_file = centroid_file.split('_centroids')[0] + '.mov'
    (vid_width, vid_height, vid_fps, vid_frames) = getVideoData(movie_file)
    print(vid_width, vid_height, vid_fps)
    
    # get last frame of movie
    f, a = plt.subplots(1, figsize=(14,6))
    frame = getLastFrame(movie_file)
    a.imshow(frame)
    
    # read in coordinates
    df = pd.read_csv(centroid_file, names = ['x','y'], header=None) 
    xcoords = df.x.values
    ycoords = df.y.values
    
    # smooth the data?
    smoothedx = smoothFiltfilt(xcoords,3,0.05)
    smoothedy = smoothFiltfilt(ycoords,3,0.05)
    
    # just a basic line plot
    # plt.plot(xcoords,ycoords, linewidth=8, color = 'forestgreen', label = 'raw') # raw coordinates
    # plt.plot(smoothedx,smoothedy, linewidth=2, color = 'lightgreen', label = 'smoothed') # smoothed
    # plt.legend() 
    # print(len(xcoords),len(smoothedx))
    
    # calculate distance
    distance_traveled = cumulativeDistance(smoothedx, smoothedy)
    print('Distance = ' + str(distance_traveled))
    
    # plot (scatter) with colormap
    cmap_name = 'plasma'
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(0,1,len(xcoords)))
    a.scatter(xcoords,ycoords, c = cols, s=10)
    a.set_xticks([])
    a.set_yticks([])
    # add legend for time
    norm = mpl.colors.Normalize(vmin=0, vmax=vid_frames / vid_fps)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label = 'Time (sec)')


    plt.axis('off')
    plt.show()
    
    return

def cumulativeDistance(x,y):
    # adapted from https://stackoverflow.com/questions/65134556/compute-cumulative-euclidean-distances-between-subsequent-pairwise-coordinates
    XY = np.array((x, y)).T
    return np.linalg.norm(XY - np.roll(XY, -1, axis=0), axis=1)[:-1].sum()

def smoothFiltfilt(x, pole=3, freq=0.1):
    # adapted from https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
    # output length is same as input length
    # as freq increases, signal is smoothed LESS
    
    b, a = scipy.signal.butter(pole, freq)
    filtered = scipy.signal.filtfilt(b,a,x)
    return filtered

def smoothConvolve(x,window_len=11,window='hanning'):
    # adapted from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    # output from this is NOT the same size as input!
    # ... so not great when calculating distance
    
    if x.ndim != 1:
        exit("smooth only accepts 1 dimension arrays.")
    
    if x.size < window_len:
        exit("Input vector needs to be bigger than window size.")
    
    if window_len<3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        exit("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
    print(len(x), len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
        
    smoothed = np.convolve(w/w.sum(),s,mode='valid')
    return smoothed 

def getLastFrame(videoFile):
    vid = cv2.VideoCapture(videoFile)

    for i in range(50): # sometimes cannot get last frame!?
    
        last_frame_num = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - i
        vid.set(cv2.CAP_PROP_POS_FRAMES, last_frame_num)
        ret, frame = vid.read()
        
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print('Got last frame at movie end - ' + str(i+1) + ' frames')
            return frame
        else:
            print('... cannot get last frame in ' + str(i+1) + ' tries . . .')
        
    exit('Cannot get last frame in ' + str(i+1) + ' tries')
            

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

    
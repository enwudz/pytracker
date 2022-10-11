#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 06:46:13 2022

@author: iwoods

From centroid coordinates (with frame times) 
    show path of centroids along the video (time gradient color)
    show frame times within video (time gradient color)
    show timing of turns (with decreasing alpha each frame) 
        and stops as text on the movie frames. 
"""

import sys
import analyzePath
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import cv2
import glob


def main(centroid_file):
    
    # get movie file
    filestem = centroid_file.split('_centroids')[0]
    movie_file = filestem + '.mov'
    
    # read in times and coordinates
    df = pd.read_csv(centroid_file, names = ['frametime','x','y'], header=None)
    frametimes = df.frametime.values
    xcoords = df.x.values
    ycoords = df.y.values
    
    # get timing of turns and stops
    stop_times, turn_times = analyzePath.main(centroid_file)  
    stop_times = np.ravel(stop_times)
    turn_times = np.ravel(turn_times)
    
    # get colors for coordinates and times (coded for time)
    frames_in_video = getFrameCount(movie_file) 
    dot_colors = makeColorList('plasma', frames_in_video)
    
    # checking frame times (from centroid file) vs. what cv2 says . . .
    #print(len(frametimes), frames_in_video) # these are not the same sometimes
    #print(frametimes[-5:])
    
    # go through frames of video
    vid = cv2.VideoCapture(movie_file)
    frame_number = 0
    
    # plotting stuff to adjust
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    marker_size = 10
    text_size = 2
    
    while vid.isOpened():
        ret, frame = vid.read()
        
        if (ret != True):  # no frame!
            print('... video end!')
            break    
    
        frametime = np.round(frametimes[frame_number], decimals = 3)
        
        # TIMES (color coded)
        frame = cv2.putText(frame, str(frametime).ljust(5,'0'),
                            (100, 100), # position
                            font, text_size,
                            dot_colors[frame_number], # color
                            4, cv2.LINE_8)
        
        # COORDINATES (color-coded to time)
        # ==> SINGLE coordinate
        # x = xcoords[frame_number]
        # y = ycoords[frame_number]
        # cv2.circle(frame, (x, y), 5, dot_colors[frame_number], -1)
        # ==> add ALL coordinates so far
        frame  = addCoordinatesToFrame(frame, xcoords[:frame_number+1], ycoords[:frame_number+1], dot_colors, marker_size)
                
        # add text for turns (fade in before and out after by text alpha)
        # add text for stops
    
        cv2.imshow('press (q) to quit', frame) # frame or binary_frame
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break      
        
        frame_number += 1
     
    print(frame_number)    
    vid.release()
    cv2.destroyAllWindows()
    
    return

def labelTimes(frametimes, labeltimes, buffer):
    '''
    Goal is to add label to a movie that gradually appears and disappears
    
    Take a vector of times, and return a vector of alphas
        alpha = 1 during the times when an event is occurring
            alpha ranges from 0 to 1 during buffer before event
            alpha ranges from 1 to zero during buffer after event
        

    Parameters
    ----------
    frametimes : numpy array
        A vector of times for each frame during the video
    labeltimes : numpy array
        A vector of times that should be labeled 
    buffer : TYPE
        The number of time increments (video frames) during which
        the fade-in or fade-out occurs

    Returns
    -------
    alphas : numpy array
        A vector of alphas to show text

    '''
    
    alphas = np.zeros(len(frametimes))
    buffervals = np.linspace(0,1,buffer+2)[1:-1]
    
    # probably a very inefficient way to do this
    # go through each frame
    # what is current value?
    # is this frame in time list? set alpha to 1
    # is this frame within buffer size BEFORE a value in time list?
         # how many frames before? what would alpha value be
         # is this alpha value > current value? If so, current value = alpha value
    # is this frame within buffer size AFTER a value in the time list?
         # how many frames before? what would alpha value be
         # is this alpha value > current value? If so, current value = alpha value
    
    for i, frametime in enumerate(frametimes):
        
        current_alpha = alphas[i]
        
        if frametime in labeltimes:
            print(frametimes)
            alphas[i] = 1
        
        else:
            
            # look in frames AFTER this one (i.e. frame i) ... 
            for j, b in enumerate(np.arange(buffer)):
                if frametime[i + j+1] in labeltimes:
                    alpha_val = buffervals[-b]
                    if alpha_val > current_alpha:
                        alphas[i] = alpha_val
                        
            # look in frames BEFORE this one (i.e. before frame i)
            for j,b in enumerate(np.arange(buffer)):
                if frametime[i - (j+1)] in labeltimes:
                    alpha_val = buffervals[b]
                    if alpha_val > current_alpha:
                        alphas[i] = alpha_val
    
        return alphas 
        

def addCoordinatesToFrame(frame, xcoords, ycoords, colors, markersize=5):
    '''

    Parameters
    ----------
    frame : open CV image
    coordinates : list of tuples
        a list of tuples of coordinates within frame
    colors : list of colors
        a list of tuples of cv2 formatted colors ... can be longer than coordinates list

    Returns
    -------
    frame with a dot positioned at each coordinate

    '''
    for i, xcoord in enumerate(xcoords):
        cv2.circle(frame, (xcoord, ycoords[i]), markersize, colors[i], -1)

    return frame

def getFrameCount(videofile):
    """get the number of frames in a movie file"""
    cap = cv2.VideoCapture(videofile)
    num_frames = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
    print('... number of frames in ' + videofile + ' : ' + str(num_frames) )
    cap.release()
    return num_frames

def makeColorList(cmap_name, N):
     cmap = cm.get_cmap(cmap_name, N)
     cmap = cmap(np.arange(N))[:,0:3]
     cmap = np.fliplr(cmap)
     
     # format for cv2 = 255 is max pixel intensity, colors are BGR     
     cmap = cmap * 255 # for opencv colors
     # convert RGB to BGR ... apparently no need! 
     # cmap = [[color[0], color[1], color[2]] for i, color in enumerate(cmap)]

     return [tuple(i) for i in cmap]
    
if __name__ == "__main__":
    
    
    if len(sys.argv) > 1:
        centroid_file = sys.argv[1]
    else:
        centroid_list = glob.glob('*centroid*')
        centroid_file = centroid_list[0]
        
    #print('Getting data from ' + centroid_file)

    main(centroid_file)
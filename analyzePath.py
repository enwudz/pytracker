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
    filestem = centroid_file.split('_centroids')[0]
    movie_file = filestem + '.mov'
    (vid_width, vid_height, vid_fps, vid_frames, vid_length) = getVideoData(movie_file, True)
    
    # get last frame of movie
    f, a = plt.subplots(1, figsize=(14,6))
    frame = getLastFrame(movie_file)
    a.imshow(frame)
    
    # get median tardigrade size (i.e. area in pix^2)
    area = getAreas(filestem)
    
    # read in coordinates
    df = pd.read_csv(centroid_file, names = ['x','y'], header=None) 
    xcoords = df.x.values
    ycoords = df.y.values
    
    # smooth the data?
    smoothedx = smoothFiltfilt(xcoords,3,0.05)
    smoothedy = smoothFiltfilt(ycoords,3,0.05)
    
    # ==> a line plot to compare raw path with smoothed path
    # plt.plot(xcoords,ycoords, linewidth=8, color = 'forestgreen', label = 'raw') # raw coordinates
    # plt.plot(smoothedx,smoothedy, linewidth=2, color = 'lightgreen', label = 'smoothed') # smoothed
    # plt.legend() 
    # print(len(xcoords),len(smoothedx))
    
    # calculate distance from smoothed data
    distance = cumulativeDistance(smoothedx, smoothedy)
    #print('Distance = ' + str(distance_traveled))
    
    # plot (scatter) centroids with colormap that shows time
    cmap_name = 'plasma'
    cmap = mpl.cm.get_cmap(cmap_name)
    cols = cmap(np.linspace(0,1,len(xcoords)))
    a.scatter(xcoords,ycoords, c = cols, s=10)
    a.set_xticks([])
    a.set_yticks([])
    # add legend for time
    norm = mpl.colors.Normalize(vmin=0, vmax=vid_length)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label = 'Time (sec)')

    # calculate # of turns, # of speed changes (from smoothed data)
    time_increment = 1 # in seconds
    speed_changes, discrete_turns, angle_space = turnsStartsStops(smoothedx, smoothedy, vid_fps, time_increment)

    # ==> add labels from experiment:
    plt.title(filestem)
    a.set_xlabel(getDataLabel(area, distance, vid_length))
    
    # Show the plot
    plt.show()
    
    return

def turnsStartsStops(xcoords, ycoords, vid_fps, increment):
    '''
    From x and y coordinates of a path, group into increments of length binsize
    estimate the number of times there is a change in speed greater than a specified threshold
    estimate the number of discrete turns in the path (where angle of turn > threshold)
    estimate the total amount of angle space explored along the path

    Parameters
    ----------
    xcoords : numpy array
        x coordinates
    ycoords : numpy array
        y coordinates.
    vid_fps : integer
        frames (i.e. coordinates) per second in video
    increment : integer
        increment duration (in seconds) for path coordinate bins

    Returns
    -------
    speed_changes = integer amount of # of changes in speed
    discrete_turns = integer amount of # of turns
    angle_space = floating point number of cumulative changes in path angle

    '''
    
    # get the number of points per time increment
    points_in_bin = int(vid_fps * increment)
    
    # get duration of the video in seconds
    video_length = np.around(len(xcoords) / vid_fps, decimals = 2)

    # get distance traveled along path
    path_distance = cumulativeDistance(xcoords, ycoords)
    
    # get average speed
    average_speed = np.around(path_distance / video_length, decimals = 2)
    
    # get speeds and angles for each bin
    binned_x = binList(xcoords, points_in_bin)
    binned_y = binList(ycoords, points_in_bin)
    speeds = np.zeros(len(binned_x))
    bearings = np.zeros(len(binned_x))
    for i, xbin in enumerate(binned_x): # could probably do list comprehension
        start_coord = np.array([xbin[0], -binned_y[i][0]]) # we do -y because y=0 is the top of the image
        end_coord = np.array([xbin[-1], -binned_y[i][-1]])
        
        # calculate speed in this increment
        distance_for_bin = np.linalg.norm(start_coord - end_coord)
        time_in_bin = len(xbin) / vid_fps
        speeds[i] = np.around(distance_for_bin / time_in_bin, decimals=2)
        
        # calculate angle in this increment
        bearings[i] = getBearing(start_coord, end_coord)
        # print(start_coord, end_coord, angles[i])
        
    # ==> from speeds and angles, FIND speedChanges, discrete_turns, angle_space
    # DEFINE THRESHOLDS for changes in speed or direction
    # for speed, define a change in speed as a change that is greater than
    #     33%(?) of the average speed across the path
    # for turn, define a discrete turn as a turn that is greater than 
    #     30(?) degrees ... when SPEED is above a certain threshold?
    speed_change_percentage_threshold = 33 # percent of average speed
    turn_degree_threshold = 30 # degrees
    # what is the magnitude of a 'real' change in speed?
    speed_change_threshold = np.around(speed_change_percentage_threshold/100 * average_speed, decimals = 2)
   
    # set counters to zero 
    speed_changes = 0   # changes in speed that meet the thresholds above
    discrete_turns = 0  # changes in bearing that meet the thresholds above
    angle_space = 0     # cumulative total of changes in bearing
    
    print('speed change threshold: ', speed_change_threshold)
    for i, speed in enumerate(speeds[:-1]):
        
        # what was the change in speed?
        delta_speed = np.abs(speeds[i+1] - speeds[i])
        
        # was this a 'discrete' change in speed?
        if delta_speed >= speed_change_threshold:
            #print('change in speed: ', delta_speed)
            speed_changes += 1
            
        # what was the change in bearing?
        delta_bearing = np.abs(bearings[i+1]-bearings[i])
        angle_space += delta_bearing # cumulative total of changes in bearing
        
        # was this a 'discrete' change in bearing (i.e. a 'turn')?
        if speed >= speed_change_threshold and delta_bearing >= turn_degree_threshold:
            #print('A TURN!')
            discrete_turns += 1
    
    angle_space = np.around(angle_space, decimals=2)
    printMe = True
    if printMe == True:
        printString = 'Speed changes: ' + str(speed_changes)
        printString += ', Discrete turns: ' + str(discrete_turns)
        printString += ', Explored angles: ' + str(angle_space)
        print(printString)
    return speed_changes, discrete_turns, angle_space

def getBearing(p1, p2):
    deltaX = p2[0]-p1[0]
    deltaY = p2[1]-p1[1]
    degrees = np.arctan2(deltaX,deltaY) / np.pi * 180
    if degrees < 0:
        degrees = 360 + degrees
    return np.around(degrees, decimals = 2)

def binList(my_list, bin_size):
    binned_list = [my_list[x:x+bin_size] for x in range(0, len(my_list), bin_size)]
    return binned_list

def getDataLabel(area, distance, vid_length, pix1mm = None): # convert from pixels?
    speed = np.around(distance/vid_length, decimals = 2)
    data_label = 'Area : ' + str(area)
    data_label += ', Distance : ' + str(distance)
    data_label += ', Time: ' + str(vid_length)
    data_label += ', Speed: ' + str(speed)
    # turns
    # speed changes
    return data_label

def getAreas(filestem):
    area_file = filestem + '_areas.csv'
    try:
        f = open(area_file,'r')
        print('Getting areas from ' + area_file)
    except:
        exit('Cannot find ' + area_file)
    areas = [float(x.rstrip()) for x in f.readlines()]
    # histfig, histax = plt.subplots(1)
    # histax.hist(areas)
    # histax.set_xlabel('Tardigrade area')
    # histax.set_ylabel('Number of frames')
    # plt.show()
    return np.median(areas)

def cumulativeDistance(x,y):
    # adapted from https://stackoverflow.com/questions/65134556/compute-cumulative-euclidean-distances-between-subsequent-pairwise-coordinates
    XY = np.array((x, y)).T
    cumulative_distance = np.linalg.norm(XY - np.roll(XY, -1, axis=0), axis=1)[:-1].sum()
    return np.around(cumulative_distance, decimals = 2)

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
        
    exit('Cannot get last frame in ' + str(i+1) + ' tries')
            

def getVideoData(videoFile, printOut = True):
    if len(glob.glob(videoFile)) == 0:
        exit('Cannot find ' + videoFile)
    else:
        vid = cv2.VideoCapture(videoFile)
        vid_width  = int(vid.get(3))
        vid_height = int(vid.get(4))
        vid_fps = int(np.round(vid.get(5)))
        vid_frames = int(vid.get(7))
        vid.release()
        vid_length = np.around(vid_frames / vid_fps, decimals = 2)
    if printOut == True:
        printString = 'width: ' + str(vid_width)
        printString += ', height: ' + str(vid_height)
        printString += ', fps: ' + str(vid_fps)
        printString += ', #frames: ' + str(vid_frames)
        printString += ', duration: ' + str(vid_length)
        print(printString)
    return (vid_width, vid_height, vid_fps, vid_frames, vid_length)


if __name__ == "__main__":
    
    
    if len(sys.argv) > 1:
        centroid_file = sys.argv[1]
    else:
        centroid_list = glob.glob('*centroid*')
        centroid_file = centroid_list[0]
        
    print('Getting data from ' + centroid_file)

    main(centroid_file)

    
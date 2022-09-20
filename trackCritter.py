#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:02:29 2022

@author: iwoods

ffmpeg: 
    ffmpeg -f image2 -r 30 -pattern_type glob -i 'tardicrawler*_frames*.png' -vcodec mpeg4 kristen_object.mp4
        (removed -s 1280x720 from above)
    
"""

import cv2
import numpy as np
import matplotlib as plt
import glob
import sys
from scipy import stats

def main(movie_file):
    
    # make background image using N random frames of video
    background_image = backgroundFromRandomFrames(movie_file, 100)
    
    # run through video and compare each frame with background
    findCritter(movie_file, background_image, 25)
    
    # bw = cv2.imread('frame_binary_tester.png')
    # cv2.imshow('press (q) to quit', bw)
    # cv2.waitKey()
    # cv2.destroyAllWindows()   
    # getCentroid(bw, 100)
    
    return

def findCritter(video_file, background, pixThreshold = 25):
    
    # go through video
    print('... starting video ' + video_file)
    vid = cv2.VideoCapture(video_file)
    
    # print out video info
    printVidInfo(vid, True)

    print('... using pixel threshold ' + str(pixThreshold))
    
    fstem = video_file.split('.')[0]
    
    frame_number = 0
    
    centroid_coordinates = [] # container for (x,y) coordinates of centroid of target object at each frame
    areas = [] # container for calculated areas of target object at each frame
    
    while vid.isOpened():
        ret, frame = vid.read()
        frame_number += 1
        
        if (ret != True):  # no frame!
            print('... video end!')
            break    
        
        # find difference between frame and background, as thresholded binary image
        binary_frame = getBinaryFrame(frame, background, pixThreshold)
        
        # find contours on the thresholded binary image
        contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # draw contours on the original frame
        cv2.drawContours(frame, contours, -1, (0,255,0), 5)
        
        # if more than one contour, make a decision about which contour is the target object
        # decision can be based on area of target ... or maybe last known position?
        if len(contours) > 1:
            print(str(frame_number) + ' has more than one objected detected!')
            if len(areas) == 0:
                target_area = 100 # just a guess
            else:
                target_area = np.mean(areas)
            target = getTargetObject(contours, target_area)
        else:
            target = contours[0]
            
        # get centroid of target object
        # calculate moment of target object
        M = cv2.moments(target)
        # calculate x,y coordinate of center of M
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # store coordinates
        centroid_coordinates.append((cX,cY))
        # show centroid on frame (color coded??)
        cv2.circle(frame, (cX, cY), 5, (0, 255, 255), -1)
        
        # get area of target object
        target_area = cv2.contourArea(target)
        areas.append(target_area)
    
        # save frame to file
        # saveFrameToFile(fstem, frame_number, frame) # frame or binary_frame
        
        # show each frame as the movie plays
        # ADD A TIME STAMP?? COLOR CODED??
        # or add CENTROID? COLOR CODED?
        # cv2.imshow('press (q) to quit', frame) # frame or binary_frame
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break        
        
    vid.release()
    cv2.destroyAllWindows()
    
    writeData(fstem + '_centroids', centroid_coordinates)
    writeData(fstem + '_areas', areas)

def writeData(filestem, data):
    outfile = filestem + '.csv'
    o = open(outfile, 'w')
    for d in data:
        if type(d) == tuple:
            stuff = [str(thing) for thing in d]
            o.write(','.join(stuff) + '\n')
        else:
            o.write(str(d) + '\n')
            
    o.close()

def getTargetObject(contours, targetArea):
    '''
    Parameters
    ----------
    contours : list of contours from cv2.findContours
        object is white, background is black
    targetArea : integer
        expected or calculated area (in pixels) of target object.

    Returns
    -------
    the contour from within contours that is closest in area to targetArea

    '''
    
    areas = [cv2.contourArea(cnt) for cnt in contours]
    closest_index = np.argmin(np.abs(areas - targetArea))
    
    return contours[closest_index]

def getBinaryFrame(frame, background, pixThreshold):
    
    # convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # do NOT blur this frame. Blurring will create differences with the background.
    # but here's code to blur, just to keep it handy
    # gaussian kernel size
    # needs to be positive, and odd. Higher values = more blurry
    #kernelSize = (51,51)
    #blurred_frame =  cv2.GaussianBlur(gray_frame,kernelSize,0)
    #blurred_frame =  cv2.medianBlur(frame,kernelSize[0])
    
    # find difference from background
    diff = cv2.absdiff(background, gray_frame)
    
    # convert difference to binary image
    _,diffbw = cv2.threshold(diff, pixThreshold, 255,cv2.THRESH_BINARY)
    #_,diffbw = cv2.threshold(diff,pixThreshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # blur the binary image ... this can get rid of 'holes' within contours
    kernelSize = (21,21) # higher kernel = more blurry, slower
    blurred_diff = cv2.medianBlur(diffbw, kernelSize[0]) # medianBlur slow ... but seems to do better
    #blurred_diff =  cv2.GaussianBlur(diffbw,kernelSize,0) # gaussian faster than median
    
    # threshold again to omit pixels of intensity < 255 (from the blur)
    # ... or leave for later as part of contour picking?
    _,final_diff = cv2.threshold(blurred_diff, 120, 255,cv2.THRESH_BINARY)
    
    return final_diff

def saveFrameToFile(file_stem, frame_number, frame):
    # to make a movie from frames
    # conda install ffmpeg
    # ffmpeg -f image2 -r 10 -s 1080x1920 -pattern_type glob -i '*.png' -vcodec mpeg4 movie.mp4
    # -r is framerate of movie
    
    file_name = file_stem + '_frames_' + str(frame_number).zfill(8) + '.png'
    cv2.imwrite(file_name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
 
def checkForBackgroundImage(movie_file):
    # check to see if there is already an image
    background_imname = movie_file.split('.')[0] + '_background.png'
    print('... looking for background image: ' + background_imname)
    background_exists = False
    if len(glob.glob(background_imname)) > 0:
        background_exists = True
        print('... found it!')
    return background_imname, background_exists        

def backgroundFromRandomFrames(movie_file, num_background_frames):
    ''' Check if there is already a background image for this movie file
    if so, load it and return it
    if not, make one and return it'''
    
    # maybe try https://hostadvice.com/how-to/how-to-do-background-removal-in-a-video-using-opencv/ instead?
    
    background_imname, background_exists = checkForBackgroundImage(movie_file)
    if background_exists == False:
    
        # if no background image already, make one!
        print("... not there! Making background image now ... ")
        
        # Select n frames at random from the video
        frames_in_video = getFrameCount(movie_file) 
        background_frames, num_background_frames = getBackgroundFrames(frames_in_video, num_background_frames)
    
        # make an empty matrix to store stack of video frames
        img = getFirstFrame(movie_file) # get first frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = np.shape(gray)
        video_stack = np.zeros([img_height, img_width, num_background_frames])
        
        frame_counter = 0
        image_index = 0
        
        # go through video
        vid = cv2.VideoCapture(movie_file)
        while (vid.isOpened()):
            ret, frame = vid.read()
            if (ret != True):  # no frame!
                print('... video end!')
                break       
            
            #print('Looking at frame ' + str(frame_counter) + ' of ' + str(frames_in_video))
            if frame_counter in background_frames:
                print('getting frame ' + str(image_index+1) + ' of ' + str(num_background_frames) )
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video_stack[:,:,image_index] = gray_frame
                image_index += 1
            frame_counter += 1
        
        #print('... releasing video ...')
        vid.release() 
        
        # get mode of image stack for each pixel
        print("... calculating mode for background image (takes awhile) ...")
        background_image = stats.mode(video_stack, axis=2)[0][:,:] # this is SLOW!
    
        print('... saving background image as ' + background_imname)
        cv2.imwrite(background_imname, background_image)
        print('... finished making background image!')
        
    print('... Loading background ...')
    background_image = cv2.imread(background_imname)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    return background_image
        

def getBackgroundFrames(num_frames, num_background_frames):
    num_background_frames = min(num_background_frames, num_frames)

    return sorted(np.random.choice(num_frames, 
                                   num_background_frames, 
                                   replace = False)), num_background_frames

def getImageSize(img):
    """get the # rows x # cols for an image"""
    num_rows, num_cols = np.shape(img)[:2]
    return num_rows, num_cols

def getFrameCount(videofile):
    """get the number of frames in a movie file"""
    cap = cv2.VideoCapture(videofile)
    num_frames = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
    print('... number of frames in ' + videofile + ' : ' + str(num_frames) )
    return num_frames

def getFirstFrame(videofile):
    """get the first frame from a movie, and return it as an image"""
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        return image
    else:
        print('cannot get an image from ' + videofile)
        return None

    
def displayImage(img):
    """Show an image, press any key to quit"""
    cv2.imshow('press any key to exit', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def printVidInfo(vid, printIt = True):
    vid_width  = int(vid.get(3))
    vid_height = int(vid.get(4))
    vid_fps = int(np.round(vid.get(5)))
    vid_frames = int(vid.get(7))
    if printIt is True:
        print('  Video height: ' + str(vid_height))
        print('  Video width:  ' + str(vid_width))
        print('  Video fps:    ' + str(vid_fps))
        print('  Video frames: ' + str(vid_frames))
    return (vid_width, vid_height, vid_fps, vid_frames)
    
if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_list = glob.glob('*.mov')
        movie_file = movie_list[0]
        
    print('Movie is ' + movie_file)

    main(movie_file)


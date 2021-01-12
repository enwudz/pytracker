#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import sys
import os
from datetime import datetime
import sys
import analysisTools
import csv

'''
stuff to tweak to change tracking parameters:
	grayBlur: the degree of blurring
		(3,3) is low, used for clear Daphnia movies
		(7,7) is good for fish
	DiffImage: the pixel threshold for what is a real difference between frames.
		Typically 12
	smoothByAveraging: the buffer size (# of frames to average)
		Typically 4
'''

#### things to modify / tweak to change tracking parameters

def getPixThreshold(pixThreshold = 25):
	return pixThreshold

def getSmoothBufferSize(buffer = 4):
	return buffer

#### Image and Video tools

# retrieve first frame of movie
def getFirstFrame(mov):
	cap = cv2.VideoCapture(mov)
	ret, img = cap.read()
	cap.release()
	return img

# set up video stream
def getVideoStream(systemArguments):
	if len(systemArguments) < 2:
		exit('usage: python ' + systemArguments[0] + ' [movFile OR 0]')
	if systemArguments[1] == '0':
		print('Starting camera')
		videoStream = 0 # camera
		videoType = 0
	else:
		videoStream = systemArguments[1] # saved movie
		print('reading ' + videoStream)
		videoType = 1
	return videoStream, videoType

# find difference between current frame and stored frame
def diffImage(storedFrame,currentFrame,pixThreshold=0,showIt=0):
	if pixThreshold == 0:
		pixThreshold = getPixThreshold()
	diff = cv2.absdiff(storedFrame,currentFrame)
	_,diff = cv2.threshold(diff,pixThreshold,255,cv2.THRESH_BINARY)
	if showIt == 1:
		cv2.imshow('Press q to exit',diff) # check difference image
	diff = diff / 255
	return diff

# convert to grayscale and blur
def grayBlur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian kernel size
    # needs to be positive, and odd. Higher values = more blurry
    #kernelSize = (3,3) # for zebrafish, daphnia
    kernelSize = (11,11)
    return cv2.GaussianBlur(gray,kernelSize,0)

# quickly show an image in matplotlib
def quickShow(img):
	plt.imshow(img,cmap='gray')
	plt.xticks([]),plt.yticks([])
	plt.show()
	return

# load the saved first frame and the saved mask.
def loadImageAndMask():
	# Load the setup image from this experiment
	try:
		i = cv2.imread('frame1.png')
		m = cv2.imread('mask.png',0)
	except:
		exit('Cannot open image and/or mask file')
	return(i,m)

# return a subset of an image based on corner coordinates
def getImageSliceFromCorners(img,corner):
	lowerLeft = corner[0]
	upperRight = corner[1]
	imageSlice = img[upperRight[1]:lowerLeft[1],lowerLeft[0]:upperRight[0]]
	return imageSlice

# adjust camera resolution
def adjustResolution(cap,cameraType):
    if (cameraType == 1):
        xres=1280
        yres=720
    elif (cameraType == 2):
        xres=825
        yres=480
    else:
        print('Using default resolution')

    print ('Adjusting camera resolution to %s x %s' % (str(xres), str(yres)))

    cap.set(3,xres)
    cap.set(4,yres)

    return cap

# from a color map, make N evenly spaced colors
def make_N_colors(cmap_name, N):
     cmap = cm.get_cmap(cmap_name, N)
     cmap = cmap(np.arange(N))[:,0:3]
     return [tuple(i*255) for i in cmap]

# calculate or acquire elapsed time
def getElapsedTime(videoType,vidInfo,startTime,endTime):
	if videoType == 0:
		vidInfo['vidLength'] = (endTime-startTime).days * (24*60*60) + (endTime-startTime).seconds
		vidInfo['startTime'] = startTime
		vidInfo['endTime'] = endTime
	else:
		vidInfo['vidLength'] = float(input('Enter movie length (in seconds): '))
	return vidInfo

#### ROI tools

# find ROIs on the mask file
# this is here because I wanted to minimize instances of cv2.findContours
def findROIsOnMask(m):
	contours = cv2.findContours(m.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# Find OpenCV version
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	contours = contours[0]

	#contours = contours[0] if major_ver==2 else contours[1]
	return contours

# set non-ROI areas to BLACK
def removeNonROI(img,mask):
	img[np.where(mask==0)]=0
	return img

# examine a masked ROI, and guess where the object is by find contours (not great)
def guessTheObject(roiSlice,maskSlice):
	# invert image so that objects are light, background dark
	inverted = 255-roiSlice
	# set area outside of roi to black
	inverted = removeNonROI(inverted,maskSlice)
	# threshold the image
	_,th = cv2.threshold(inverted,170,255,cv2.THRESH_BINARY)
	# find contours
	contours = findROIsOnMask(th)
	if len(contours) > 0:
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)
		cnt=contours[max_index]
		# find center coordinates of largest contour
		M = cv2.moments(cnt)
		try:
			cX = int(M["m10"] / M["m00"])
		except:
			cX = 0
		try:
			cY = int(M["m01"] / M["m00"])
		except:
			cY = 0
	else:
		# can't find a good object. Set x,y as CENTER of roiSlice
		xspan,yspan = np.shape(roiSlice)
		cX = int(round(xspan/2))
		cY = int(round(yspan/2))
# 		cX = 0
# 		cY = 0
	return (cX,cY)

# make and save a SINGLE circular ROI
def makeCircularROIMask(gray,center,radius):
	# img should be grayscale
	blank = np.zeros(np.shape(gray))
	cv2.circle(blank, center, radius, (255), -1)
	cv2.imwrite('mask.png',blank.astype('uint8'))
	return blank

# setup a single, circular ROI on a video file
# wish: code to select (on an image) center and radius of circle
def singleCircle(vidFile, center=(45,45), radius = 35):
	fr = getFirstFrame(vidFile)
	fr = grayBlur(fr)
	# make single circular ROI mask for testing
	mask = makeCircularROIMask(fr,center,radius)
	# save mask
	cv2.imwrite('mask.png',mask)
	# subtract out nonROI
	subt = removeNonROI(fr,mask)
	# show the masked image
	quickShow(subt)
	return mask

# find all of the pixels that are selected by the user (GREEN rectangle(s))
def greenThreshImage(mask):
	# everything that is not GREEN in mask is set to black
	mask[mask != (0,255,0)] = 0
	# convert to thresholded image
	gmask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	_,gmask = cv2.threshold(gmask,12,255,0)
	return gmask

# return the # of ROIs in a mask
def countROIs():
	return len(findROIsOnMask(cv2.imread('mask.png',0)))

# find the ROIs in a thresholded image and assign each a different number
def findAndNumberROIs():

	# input needs to be grayscale, as uint8 ... read from file
	try:
		mask = cv2.imread('mask.png',0)
	except:
		exit('Cannot open mask file')

	# find shapes in mask
	contours = findROIsOnMask(mask)

	# assign pixel values to each ROI evenly across 255 (np.floor(x)) and save mask
	pixVals =  np.floor(np.linspace(10,255, len(contours) + 2))
	# or assign pixel values sequentially
	#pixVals = range(len(contours)+1)

	# draw the contours on the mask
	for h,cnt in enumerate(reversed(contours)):
		cv2.drawContours(mask,[cnt],0,pixVals[h+1],-1)
		#cv2.drawContours(mask,cnt,0,pixVals[h+1],-1)

	# save the new mask that is numbered by ROI
	cv2.imwrite('mask.png',mask)

	# return mask, number of contours
	return mask, len(contours)

# assign an integer value to each ROI in a mask
def convertMaskToWeights(mask):
	vals = np.unique(mask)
	for i in range(len(vals)):
		mask[mask==vals[i]]=i
	mask = mask.astype(int)
	w = mask.ravel() # convert to single row for weights in bincount
	return mask,w

# load an ROI mask and get a list of corner coordinates
# of inscribing rectangle for reach ROI
def getROICornersFromMask():

	corners=[]
	m = cv2.imread('mask.png',0)

	contours = findROIsOnMask(m)

	for cnt in range(len(contours)):

		coords = contours[cnt]

		xPoints = [c[0][0] for c in coords]
		yPoints = [c[0][1] for c in coords]

		minX = np.min(xPoints)
		maxX = np.max(xPoints)

		minY = np.min(yPoints)
		maxY = np.max(yPoints)

		lowerLeft = (minX,maxY)
		upperRight = (maxX,minY)

		corners.append([lowerLeft,upperRight])

	return list(reversed(corners))

# Generate a figure showing the ROImask on image
def showROI():

	(img,m) = loadImageAndMask()

	f = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
	pic = plt.subplot(1,1,1)

	pic.imshow(img) # show video image
	#pic.hold(True) # deprecated

	# need to find and number ROI
	map = pic.imshow(m,alpha=0.3) # superimpose ROI mask

	numROI = str(len(np.unique(m)) - 1)
	plt.title('Number of ROI = ' + numROI)
	plt.xticks([]),plt.yticks([]) # clear axis ticks and labels

	# color Bar
	labs = ['background']
	for a in range(1,len(np.unique(m))):
		labs.append(str(a))

	# labels for color bar; wishlist: reduce if > 16
	cbar = f.colorbar(map,ticks=np.unique(m))
	cbar.ax.get_yaxis().labelpad=20
	cbar.ax.set_yticklabels(labs)
	cbar.ax.invert_yaxis()
	cbar.set_label('ROI #', rotation = 270, size = 16)

	deleteData('roiMask*.png')
	savedRoiMask = 'roiMask' + numROI + '.png'
	plt.savefig(savedRoiMask)
	plt.show()
	return

# find the corners of the user-selected GREEN rectangle
# as [x y] where 0 0 = lower left of image]
def getCornersOfRectangle(m):

	ret,thresh = cv2.threshold(m,1,255,0)
	contours = findROIsOnMask(m)

	lowerLeft = tuple(contours[0][1][0]) # as (x,y) where 0 0 = lower left of image]
	upperRight = tuple(contours[0][3][0])

	return (lowerLeft, upperRight)

# given a number of wells in a plate, return the # columns and # rows
def getRowsColsFromNumWells(numWells):
	# first get a list of the factors of the number
	factors = []
	for i in np.arange(numWells)+1:
		if numWells % i == 0:
			factors.append(i)
	# if factor list is even, get the numbers around middle
	if len(factors) % 2 == 0:
		mid = int(len(factors)/2)
		rows = factors[mid-1]
		cols = factors[mid]
	else:
		# if factor list is odd, get median
		cols = int(np.median(factors))
		rows = cols
	return rows, cols

def getXspanYspanFromRectangle(lowerLeft,upperRight):
	xspan = upperRight[0] - lowerLeft[0]
	yspan = lowerLeft[1] - upperRight[1]
	return (xspan,yspan)

# make mask of CIRCULAR ROI, loading a mask and input well#
def gridCircles(numWells):

	numRows,numCols = getRowsColsFromNumWells(numWells)
	m = cv2.imread('mask.png',0)

	vals = np.unique(m)
	if len(vals) > 2:
		sys.exit('Mask has more than one shape!')

	# identify enclosing rectangle for entire plate
	(lowerLeft, upperRight) = getCornersOfRectangle(m)
	(xspan,yspan) = getXspanYspanFromRectangle(lowerLeft,upperRight)

	# find radius
	wallSize = 0.1 # expressed as proportion of fullRadius
	fullRadius = int(np.round(xspan / (numCols * 2)))
	radius = int(np.round((1-wallSize)*fullRadius))

	# find coordinate of first circle
	firstX = int(np.round(fullRadius)) + lowerLeft[0]
	firstY = int(np.round(fullRadius)) + upperRight[1]

	# find centers
	xPoints = [firstX + (2*fullRadius*i) for i in np.arange(numCols)]
	yPoints = [firstY + (2*fullRadius*i) for i in np.arange(numRows)]
	centers = [(x,y) for y in yPoints for x in xPoints]

	# draw circles on new blank image
	b = np.zeros(np.shape(m))
	[cv2.circle(b, i, radius, (255), -1) for i in centers]
	return b

# make mask of rectangular/square ROI, loading a mask and input well#
def gridRectangles(numWells):

	numRows,numCols = getRowsColsFromNumWells(numWells)
	#numRows,numCols = 1,3
	m = cv2.imread('mask.png',0)

	vals = np.unique(m)
	if len(vals) > 2:
		sys.exit('Mask has more than one shape!')

	(lowerLeft, upperRight) = getCornersOfRectangle(m)
	(xspan,yspan) = getXspanYspanFromRectangle(lowerLeft,upperRight)

	innerRect = 0 # 1 if want inner + outer rectangles
	wallSize = 0.04 # usually 0.1

# 	print ('xspan = ' + str(xspan))
# 	print ('yspan = ' + str(yspan))
# 	print ('numCols = ' + str(numCols))
# 	print ('numRows = ' + str(numRows))

	# Find dimensions of ROI's and walls
	xWidth = int(round (xspan / (numCols + (numCols * wallSize) - wallSize))) # algebra!
	yWidth = int(round (yspan / (numRows + (numRows * wallSize) - wallSize)))
	xWallWidth = int(np.floor(wallSize * xWidth))
	yWallWidth = int(np.floor(wallSize * yWidth))

	# these will print coordinates of large rectangle
	#print upperLeft
	#print lowerRight
	#print 'x width = ' + str(xWidth) + '; wallSize = ' + str(xWallWidth)
	#print 'y width = ' + str(yWidth) + '; wallSize = ' + str(yWallWidth)

	# make NEW MASK
	#   starting in upper left, find coordinates of each ROI
	#   and replace in plateMask current ROI number
	#   then move on to the next row and to the same thing

	gridMask = np.zeros(m.shape)
	roiNumber = 1
	xStart = int(lowerLeft[0])
	currentX = xStart
	currentY = int(upperRight[1])

	for r in range(numRows):

		for c in range(numCols):

			#print (roiNumber,currentX,currentY) # prints coordinates of upperLeft corner

			gridMask[currentY:currentY+yWidth,currentX:currentX+xWidth] = roiNumber
			roiNumber += 1

			''' replace region in plateMask with current ROI number
			when indexing, array origin is in upper left
			but image origin is lower left, so it's a bit confusing'''

			# now do an inner rectangle if necessary
			if innerRect == 1:
				(gridMask, roiNumber) = addInnerRectangle(gridMask,roiNumber,currentX,currentY,xWidth,yWidth)

			# done with this column, update x
			currentX = currentX + xWidth + xWallWidth

		# done with columns
		# now ready to start new row, update y and reset X
		currentX = xStart
		currentY = currentY + yWidth + yWallWidth

	# done generating ROIs
	gridMask = gridMask.astype(int)
	cv2.imwrite('mask.png',gridMask)

	gridMask, numROI = findAndNumberROIs()

	return gridMask

# function for selecting ROIS with mouse
def drawRectangle(event,x,y,flags,params):

	global ix,iy,drawing, img

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			cv2.rectangle(img,(ix,iy),(x,y),(0,255,0,0.5),-1) #GREEN

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False

# find a circle that fits inside a rectangle
def findCircleInsideRectangle(upperLeft,lowerRight):
	(xspan,yspan) = getXspanYspanFromRectangle(upperLeft,lowerRight)
	xcenter = upperLeft[0] + int(np.round(xspan/2))
	ycenter = lowerRight[1] + int(np.round(yspan/2))
	radius = int(np.round(np.mean([xspan/2,yspan/2])))
	center = (xcenter,ycenter)
	return (center,radius)

# setup a mask from the selected rectangles.
# within this function, can select grid of (r)ectangles or grid of (c)ircles
def defineRectangleROI(videoStream,numWells,roiShape='r'):

	global ix, iy, drawing, img

	# Setup drawing of ROI's
	drawing = False
	ix,iy = -1,-1

	imageName = 'frame1.png'
	maskName = 'mask.png'

	# remove old images and masks, if present
	removeFiles([imageName,maskName])

	cap = cv2.VideoCapture(videoStream)

	# adjust resolution if needed on attached camera
	if videoStream == 0:
		cap = adjustResolution(cap,1) # 1 for logitech camera

	# Grab a frame from the video
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			img = frame
			img[img>250]=250 # remove saturated pixels
			cv2.imwrite(imageName,img)
			break
		else:
			sys.exit("Failure to capture image")
			break

	cap.release()
	cv2.destroyAllWindows()

	## add ROIs onto the image
	roiWindowName = 'Select ROI, press Esc when finished'
	cv2.namedWindow(roiWindowName)
	cv2.setMouseCallback(roiWindowName,drawRectangle)

	while(1):
		cv2.imshow(roiWindowName,img)
		k = cv2.waitKey(1) & 0xFF
		if  k == 27:
			# save current image with green rectangles
			#cv2.imwrite('rects.png',img)
			mask = img;
			break
	cv2.destroyAllWindows()

	# make a binary image of the ROI mask based on GREEN rectangle
	gmask = greenThreshImage(mask)
	cv2.imwrite(maskName,gmask)

	# find the shapes (ROIs) in the image, count them
	(mask, numROI) = findAndNumberROIs()

	if numWells != 1:
		# can do rectangular
		if roiShape == 'r':
			mask = gridRectangles(numWells)
		# or circular
		else:
			mask = gridCircles(numWells)
			# find and number??

	elif roiShape == 'c':
		# find center and radius of rectangle
		# and use it to make a circular ROI!
		(upperLeft, lowerRight) = getCornersOfRectangle(mask)
		(center,radius) = findCircleInsideRectangle(upperLeft,lowerRight)
		mask = makeCircularROIMask(mask,center,radius)

	# save the updated mask
	cv2.imwrite(maskName,mask)

	findAndNumberROIs()

#### Motion tools

# main algorithm for finding objects
def findCenterOfBlob(thresholdedImage):
	xes = np.sum(thresholdedImage,axis=0)
	yes = np.sum(thresholdedImage,axis=1)
	xav = int(np.round(np.sum((xes * (np.arange(len(xes)) + 1))) / np.sum(xes)))
	yav = int(np.round(np.sum((yes * (np.arange(len(yes)) + 1))) / np.sum(yes)))
	return (xav,yav) # this is a tuple

# smooth x,y data by averaging last N positions
def smoothByAveraging(newPosition,storedPositions):
	# newPosition is a tuple
	# want to return a tuple too!
	bufferSize = 4 # 4 for daphnia, zebrafish; 15 for mice
	if len(storedPositions) == 0:
		storedPositions = newPosition * np.ones([bufferSize,2])
	else:
		storedPositions = storedPositions[1:] # remove first element
		coordsAsList = [ newPosition[0],newPosition[1] ]
		storedPositions = np.reshape(np.append(storedPositions,coordsAsList),[bufferSize,2])
	avg = np.mean(storedPositions,axis=0)
	return ( int(np.round(avg[0])), int(np.round(avg[1]))), storedPositions

#### Thigmotaxis

# get the center coordinate of a rectangle, given its corners
def getCenterFromCorner(corner):
	xspan = corner[1][0]-corner[0][0]
	yspan = corner[0][1]-corner[1][1]
	xCenter = corner[0][0]+int(round(xspan/2))
	yCenter = corner[1][1]+int(round(yspan/2))
	return (xCenter,yCenter)

# given a fraction of total length or width of ROI,
# determine threshold distance for what is inside vs. outside
def thigmoDistanceThreshold(fractionOfLength,cornersOfRoi):
	xspan = cornersOfRoi[1][0]-cornersOfRoi[0][0]
	yspan = cornersOfRoi[0][1]-cornersOfRoi[1][1]

	xthresh = int(round(xspan/2 * fractionOfLength))
	ythresh = int(round(yspan/2 * fractionOfLength))

	return (xthresh,ythresh)

def getInnerRegionOfSquare(thresholdDistFromEdge,cornerOfRoi):
	xspan = cornerOfRoi[1][0]-cornerOfRoi[0][0]
	yspan = cornerOfRoi[0][1]-cornerOfRoi[1][1]

	xBuffer = int(thresholdDistFromEdge*xspan)
	yBuffer = int(thresholdDistFromEdge*yspan)

	xInner = range(cornerOfRoi[0][0] + xBuffer, cornerOfRoi[1][0] - xBuffer)
	yInner = range(cornerOfRoi[1][1] + yBuffer, cornerOfRoi[0][1] - yBuffer)

	return(xInner,yInner)

def findDistanceBetweenCoordinates(p1,p2):
	import math
	return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def determineIfOuter(coords,innerRegions):
	xCoord = coords[0]
	yCoord = coords[1]
	if xCoord not in innerRegions[0] or yCoord not in innerRegions[1]:
		isOuter = 1
	else:
		isOuter = 0
	return isOuter

def xyToThigmoSquares(data,thresholdDistFromEdge=0.1464466):

	# get information from data
	numFrames = np.shape(data)[0]
	roiData = data[:,1:]

	# find corners of each ROI
	corners = getROICornersFromMask()
	numRoi = len(corners)

	# for each ROI, get list of INNER xCoordinates and yCoordinates
	innerRegions = [getInnerRegionOfSquare(thresholdDistFromEdge,corners[c]) for c in range(numRoi)]

	# initialize empty matrix for thigmotaxis data
	thigmo = np.zeros([numFrames,numRoi])

	# each line of xy data, determine if ROI object coordinate is within INNER region
	for f in range(numFrames):

		# for each ROI, get (x,y) coordinate for this frame
		xyCoords = [(roiData[f,i],roiData[f,i+numRoi]) for i in range(numRoi)]

		isOut = [determineIfOuter(xyCoords[i],innerRegions[i]) for i in range(numRoi)]

		# add data to thigmo
		thigmo[f,:] = isOut

	# insert time vector into thigmo
	thigmo = np.insert(thigmo,0,data[:,0],axis=1)

	return thigmo

# take x,y coordinate data and return matrix of inner vs. outer given a threshold
def xyToThigmoCircles(data,thresholdDistFromCenter=0.71):
	# 0.71 = circle with half area in and half area out

	# find center coordinates of each ROI
	corners = getROICornersFromMask()
	centers = [getCenterFromCorner(c) for c in corners]
	numRoi = len(centers)

	# get information from data
	numFrames = np.shape(data)[0]
	roiData = data[:,1:]

	# define threshold of what is inner vs. outer
	distanceThresholds = [thigmoDistanceThreshold(thresholdDistFromCenter,corners[c]) for c in range(numRoi)]
	distanceThresholds = [np.mean(x) for x in distanceThresholds]

	# initialize empty matrix for thigmotaxis data
	thigmo = np.zeros([numFrames,numRoi])

	# each line of xy data, determine if ROI object coordinate is outside threshold distance
	for f in range(numFrames):

		# for each ROI, get (x,y) coordinate for this frame
		xyCoords = [(roiData[f,i],roiData[f,i+numRoi]) for i in range(numRoi)]

		# for Circles, find distance between coordinates and centers
		distances = [findDistanceBetweenCoordinates(xyCoords[xy],centers[xy]) for xy in range(numRoi)]

		# set OUT to 1 if distance greater than the threshold
		isOut = [1 if distances[xy] > distanceThresholds[xy] else 0 for xy in range(numRoi)]

		# for Squares or Rectangles ... treat x and y thresholds separately?
		# for each ROI get absolute value of difference between x-center and x-coordinate
		# and test if this difference is above the specified threshold
		# if difference is above threshold, set to 1
# 		xOuters = [1 if (abs(centers[i][0] - xyCoords[i][0]) > distanceThresholds[i][0]) == True
# 			else 0 for i in range(numRoi)]
		# do the same thing for y
# 		yOuters = [1 if (abs(centers[i][1] - xyCoords[i][1]) > distanceThresholds[i][1]) == True
# 			else 0 for i in range(numRoi)]
		# add the xTrues and the yTrues
# 		sumOuters = np.sum((xOuters,yOuters),axis=0)
		# set everything that is greater than 0 to 1#
# 		isOut = [1 if i > 0 else 0 for i in sumOuters]

		# add data to thigmo
		thigmo[f,:] = isOut

	# insert time vector into thigmo
	thigmo = np.insert(thigmo,0,data[:,0],axis=1)

	return thigmo

#### SocialBox

def xyDataToAboveBelow(xyData):
	# find center coordinates of each ROI
	corners = getROICornersFromMask()
	centers = [getCenterFromCorner(c) for c in corners]
	numRoi = len(centers)
	numFrames = np.shape(xyData)[0]
	yCenters = [c[1] for c in centers]
	aboveOrBelow = np.zeros([numFrames,numRoi])

	for r in range(numRoi):
		aboveOrBelow[:,r] = xyData[:,r+1+numRoi] < yCenters[r]

	# insert time vector back into aboveOrBelow
	aboveOrBelow = np.insert(aboveOrBelow,0,xyData[:,0],axis=1)
	return aboveOrBelow

#### File Management

def setupVidInfo():
	vidInfo = {}
	vidInfo['vidType'] = ''
	vidInfo['fps'] = ''
	vidInfo['movieFile'] = ''
	vidInfo['numFrames'] = ''
	vidInfo['startTime'] = ''
	vidInfo['endTime'] = ''
	vidInfo['vidLength'] = ''
	return vidInfo

def saveVidInfo(vidInfo):
	w = csv.writer(open('vidInfo.csv','w'))
	for key,val in vidInfo.items():
		w.writerow([key,val])

def loadVidInfo():
	vidInfo = {}
	for key,val in csv.reader(open('vidInfo.csv')):
		vidInfo[key] = val
	return vidInfo

def saveData(data,dataType):
	timeStamp = datetime.now()
	fname = dataType + timeStamp.strftime('%y%m%d-%H%M%S') + '.npy'
	np.save(fname,data)

def checkForPreviousData(searchType):
	f = glob.glob(searchType)
	numFiles = len(f)
	if numFiles > 0:
		exit('There are %i .npy files in this directory already\nPlease remove them or make a new directory!' % numFiles)

def deleteData(searchType):
    filenames = glob.glob(searchType)
    for f in filenames:
        os.remove(f)

def calculateAndSaveFPS(data, vidInfo):

	# calculate fps
	print('Estimating fps . . . ')
	numFrames = np.shape(data)[0]
	print('Number of frames: ' + str(numFrames))
	vidInfo['numFrames'] = str(numFrames)

	print('Movie length is ' + str(vidInfo['vidLength']) + ' seconds')
	elapsed = float(vidInfo['vidLength'])

	fps = numFrames/elapsed
	fps = int(round(fps))
	vidInfo['fps'] = str(fps)
	print('frames per second: ' + str(fps))

	# update vidInfo
	saveVidInfo(vidInfo)

	return vidInfo

def addTimeStampsToData(data):

	# load video info
	vidInfo = loadVidInfo()

	fps = float(vidInfo['fps'])

	print('\nAdding time vector to data at fps = %1.0f ...' % fps)
	timeStamps = list(range(0,np.shape(data)[0]))
	timeStamps = [x/fps for x in timeStamps]

	# add time stamps to data
	data = np.insert(data,0,timeStamps,axis=1)

	# remove old data
	#deleteData('xy*npy')

	# save new data with time stamp
	print('saving data ...')
	np.save('xyDataWithTimeStamps.npy',data)

def removeFiles(fileList):
	for fileName in fileList:
		try:
			os.remove(fileName)
		except OSError:
			pass

def keepOrAppend(clearData,dataType):
	# check to see if clear existing data
	# or append this data to existing data
	if clearData == 'n':
		deleteData(dataType+'*.npy')
	else:
		print("Keeping existing data ...")

## This is the NEWEST version (Aug 24,2020). It accommodates time "gaps" among the npy files that result from stopping and starting data collection.

# Until Dec26,2020, its filename was "testing.py"

import numpy as np
import sys
import analysisTools
from matplotlib import dates

binSize = 1

# choose whether to get all of the data, or just those after a certain time
if len(sys.argv) > 1:
    startData = sys.argv[1]
    csvFile = 'distances_' + str(binSize) + '_' + startData + '.csv'
    timeStampFile = 'distances_' + str(binSize) + '_' + startData + '.csv'

    data = analysisTools.loadDataFromStartTime(startData,'xy')

else:
    csvFile = 'distances_' + str(binSize) + '.csv'
    timeStampFile = 'timeStamps_' + str(binSize) + '.csv'
    # load all xy data
    data = analysisTools.loadData('xy')


# split data at time gaps, if any
dataChunks = analysisTools.getMatrixSubsets(data,analysisTools.checkDataForTimeGaps(data))

# initialize empty containers for binnedData and binnedTime
binnedData = []
binnedTime = []

for i,data in enumerate(dataChunks):

    print('--> looking at chunk ' + str(i+1) + ' of ' + str(len(dataChunks)))
    # convert coordinates to distances
    d = analysisTools.convertXYDataToDistances(data)

    '''
    bin data
    analysisTools.binData takes two arguments:
    (1) the data to be binned (including timestamps)
    (2) the binSize in seconds
    for 1 minute bins, binSize = 60
    for 10 minutes bins, binSize = 600

    analysisTools.binData returns two outputs
    (1) binned data, (2) binned time vector
    '''

    bd,bt = analysisTools.binData(d,binSize)
    print(' data in this chunk',np.shape(bd)) # could comment these out
    print(' times in this chunk',np.shape(bt)) # could comment these out

    if len(binnedData) == 0:
        binnedData = bd
        binnedTime = bt
    else:
        binnedData = np.vstack((binnedData,bd))
        binnedTime = np.hstack((binnedTime,bt))

# save the binned distances data to a .csv file
print('    saving data to %s . . . ' % csvFile)
np.savetxt(csvFile, binnedData, delimiter=',', fmt = '%1.2f' ,newline='\n')

# convert time to yyyy-mm-dd hh:mm:ss format
binnedTime = [dates.num2date(x).strftime('%Y-%m-%d %H:%M:%S') for x in binnedTime]

# save the binned time vector to a .csv file
print('    saving timestamps to %s . . . ' % timeStampFile)
with open(timeStampFile,'w') as f:
	for t in binnedTime:
		f.write('%s\n' % t)

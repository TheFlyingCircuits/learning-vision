# get some math utils
import math

# OpenCV
import numpy as np
import cv2 as cv

# wpiLib
from cscore import CameraServer
from ntcore import NetworkTableInstance

# Let's give it a shot, maybe OpenCV releases the GIL!
# from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# tracking stats
from collections import defaultdict

def getVectorFieldViz(gradMags, gradAngles):
    positiveAngles = np.copy(gradAngles)
    positiveAngles[positiveAngles < 0] += 2*np.pi
    normalizedAngles = positiveAngles / (2*np.pi)
    
    h = (179 * normalizedAngles).astype(np.uint8)
    s = np.full_like(h, 255)
    v = (255 * gradMags).astype(np.uint8)
    hsv = cv.merge([h, s, v])
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


# image derivative functions:
def getGradient(greyscaleImage, threadPool=None):
    # init kernels
    kernelX = np.array([-0.5, 0, 0.5], dtype=greyscaleImage.dtype).reshape(1, 3)
    kernelY = np.array([-0.5, 0, 0.5], dtype=greyscaleImage.dtype).reshape(3, 1)

    if (threadPool is None):
        # calculate each component of the gradient one after the other
        gradientX = cv.filter2D(greyscaleImage, -1, kernelX, borderType=cv.BORDER_REPLICATE)
        gradientY = cv.filter2D(greyscaleImage, -1, kernelY, borderType=cv.BORDER_REPLICATE)
        return gradientX, gradientY
    else:
        # send work to threads in pre-allocated threadpool so they can be done in parallel
        gradX_inProgress = threadPool.submit(cv.filter2D, greyscaleImage, -1, kernelX, borderType=cv.BORDER_REPLICATE)
        gradY_inProgress = threadPool.submit(cv.filter2D, greyscaleImage, -1, kernelY, borderType=cv.BORDER_REPLICATE)
        
        # grab the result from each thread when it's ready
        gradientX = gradX_inProgress.result()
        gradientY = gradY_inProgress.result()
        
        return gradientX, gradientY


    # it would be nice if I could just have the threadpool be local, so that the calling funciton doesn't
    # have to worry about passing it in and "it just works", but that puts a noticable damper on performance
    # because the threadpool has to be constructed on every funciton call, and destructed on every function exit.
    # # send work to threads
    # with ThreadPoolExecutor(max_workers=2) as myThreadPool:
    #     myFutureX = myThreadPool.submit(cv.filter2D, greyscaleImage, -1, kernelX, borderType=cv.BORDER_REPLICATE)
    #     myFutureY = myThreadPool.submit(cv.filter2D, greyscaleImage, -1, kernelY, borderType=cv.BORDER_REPLICATE)


def getDirectionalDerivative(surface, directionX, directionY, threadPool=None):
    gradX, gradY = getGradient(surface, threadPool)
    return cv.blendLinear(src1=gradX, src2=gradY, weights1=directionX, weights2=directionY)
    # return gradX * directionX + gradY * directionY


def helpfulDivide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))
    





def shiftUp(img):
    # originially implemented with numpy roll,
    # but that's pretty slow so now I'm trying
    # an opencv function.
    # upon testing, opencv may be slower?
    # I'll come back to this later.
    bottomRow = img[-1,:]
    shiftedUp = np.roll(img, -1, axis=0)
    shiftedUp[-1,:] = bottomRow
    # height, width = img.shape

    # translationX = 0
    # translationY = -1
    # translationMatrix = np.array([[1, 0, translationX], [0, 1, translationY]], dtype=np.float32)

    # shiftedUp = cv.warpAffine(img, translationMatrix, (width, height), borderMode=cv.BORDER_REPLICATE)
    return shiftedUp

def shiftDown(img):
    topRow = img[0,:]
    shiftedDown = np.roll(img, 1, axis=0)
    shiftedDown[0,:] = topRow
    # height, width = img.shape

    # translationX = 0
    # translationY = 1
    # translationMatrix = np.array([[1, 0, translationX], [0, 1, translationY]], dtype=np.float32)

    # shiftedDown = cv.warpAffine(img, translationMatrix, (width, height), borderMode=cv.BORDER_REPLICATE)
    return shiftedDown

def shiftRight(img):
    leftColumn = img[:,0]
    shiftedRight = np.roll(img, 1, axis=1)
    shiftedRight[:,0] = leftColumn
    # height, width = img.shape

    # translationX = 1
    # translationY = 0
    # translationMatrix = np.array([[1, 0, translationX], [0, 1, translationY]], dtype=np.float32)

    # shiftedRight = cv.warpAffine(img, translationMatrix, (width, height), borderMode=cv.BORDER_REPLICATE)
    return shiftedRight

def shiftLeft(img):
    rightColumn = img[:,-1]
    shiftedLeft = np.roll(img, -1, axis=1)
    shiftedLeft[:,-1] = rightColumn
    # height, width = img.shape

    # translationX = -1
    # translationY = 0
    # translationMatrix = np.array([[1, 0, translationX], [0, 1, translationY]], dtype=np.float32)

    # shiftedLeft = cv.warpAffine(img, translationMatrix, (width, height), borderMode=cv.BORDER_REPLICATE)
    return shiftedLeft

def getLevelSets(img, isoValue):
    # TODO: maybe optimize with filter2D
    #       and see if opencv has a faster sign() function
    signs = np.sign(img - isoValue).astype(np.int8)
    signAbove = shiftDown(signs)
    signBelow = shiftUp(signs)
    signLeft = shiftRight(signs)
    signRight = shiftLeft(signs)

    crossesNorth = ((signs + signAbove) == 0)
    crossesSouth = ((signs + signBelow) == 0)
    crossesEast = ((signs + signRight) == 0)
    crossesWest = ((signs + signLeft) == 0)
    isZero = (signs == 0) # unlikely, but here for correctness

    return crossesNorth | crossesSouth | crossesEast | crossesWest | isZero


def threadedLevelSets(surface, isoValue, threadPool):
    signs = np.sign(surface - isoValue).astype(np.int8)

    futureNorth = threadPool.submit(crossesNorth, signs)
    futureSouth = threadPool.submit(crossesSouth, signs)
    futureEast = threadPool.submit(crossesEast, signs)
    futureWest = threadPool.submit(crossesWest, signs)

    # An iterator that returns futures in the order they are completed
    futuresInOrder = concurrent.futures.as_completed((futureNorth, futureSouth, futureEast, futureWest))

    firstHalf = threadPool.submit(np.logical_or, next(futuresInOrder).result(), next(futuresInOrder).result())
    secondHalf = threadPool.submit(np.logical_or, next(futuresInOrder).result(), next(futuresInOrder).result())

    futuresInOrder = concurrent.futures.as_completed((firstHalf, secondHalf))


    output = (signs == 0)
    output |= next(futuresInOrder).result()
    output |= next(futuresInOrder).result()
    return output

def crossesNorth(signs):
    signAbove = shiftDown(signs)
    return ((signs + signAbove) == 0)

def crossesSouth(signs):
    signBelow = shiftUp(signs)
    return ((signs + signBelow) == 0)

def crossesEast(signs):
    signRight = shiftLeft(signs)
    return ((signs + signRight) == 0)

def crossesWest(signs):
    signLeft = shiftRight(signs)
    return ((signs + signLeft) == 0)

def numericLevelSets(surface, isoValue):
    signs = np.sign(surface - isoValue).astype(np.int8)
    n = signs + shiftDown(signs)
    s = signs + shiftUp(signs)
    e = signs + shiftLeft(signs)
    w = signs + shiftRight(signs)

    product = signs
    for d in [n, s, e, w]:
        product *= d
    return (product == 0)

def threadedNumericLevelSets(surface, isoValue, threadPool):
    if (threadPool is None):
        return numericLevelSets(surface, isoValue)
    signs = np.sign(surface - isoValue).astype(np.int8)

    sumNorth = threadPool.submit(crossesNorthNumeric, signs)
    sumSouth = threadPool.submit(crossesSouthNumeric, signs)
    sumEast = threadPool.submit(crossesEastNumeric, signs)
    sumWest = threadPool.submit(crossesWestNumeric, signs)

    # An iterator that returns futures in the order they are completed
    futuresInOrder = concurrent.futures.as_completed((sumNorth, sumSouth, sumEast, sumWest))

    firstHalf = threadPool.submit(np.multiply, next(futuresInOrder).result(), next(futuresInOrder).result())
    secondHalf = threadPool.submit(np.multiply, next(futuresInOrder).result(), next(futuresInOrder).result())

    futuresInOrder = concurrent.futures.as_completed((firstHalf, secondHalf))

    product = signs
    for future in futuresInOrder:
        product *= future.result()
    return (product == 0)



def crossesNorthNumeric(signs):
    signsAbove = shiftDown(signs)
    return signs + signsAbove
def crossesSouthNumeric(signs):
    signsBelow = shiftUp(signs)
    return signs + signsBelow
def crossesEastNumeric(signs):
    signsRight = shiftLeft(signs)
    return signs + signsRight
def crossesWestNumeric(signs):
    signsLeft = shiftRight(signs)
    return signs + signsLeft

def crossesLeftRight(signs):
    return crossesEastNumeric(signs) * crossesWestNumeric(signs)

def crossesUpDown(signs):
    return crossesNorthNumeric(signs) * crossesSouthNumeric(signs)

def lessThreadedNumericLevelSets(surface, isoValue, threadPool):
    signs = np.sign(surface - isoValue).astype(np.int8)

    leftRight = threadPool.submit(crossesLeftRight, signs)
    upDown = threadPool.submit(crossesUpDown, signs)

    futuresInOrder = concurrent.futures.as_completed((leftRight, upDown))

    output = signs
    output *= next(futuresInOrder).result()
    output *= next(futuresInOrder).result()
    return (output == 0)

def shortCircuitLevelSets(surface, isoValue):
    # TODO: maybe optimize with filter2D
    #       and see if opencv has a faster sign() function
    signs = np.sign(surface - isoValue).astype(np.int8)
    signAbove = shiftDown(signs)
    signBelow = shiftUp(signs)
    signLeft = shiftRight(signs)
    signRight = shiftLeft(signs)

    crossesNorth = ((signs + signAbove) == 0)
    crossesSouth = ((signs + signBelow) == 0)
    crossesEast = ((signs + signRight) == 0)
    crossesWest = ((signs + signLeft) == 0)
    isZero = (signs == 0) # unlikely, but here for correctness

    # manual short circuiting
    crossesZero = isZero
    crossesZero[crossesZero == False] |= crossesNorth[crossesZero == False]
    crossesZero[crossesZero == False] |= crossesSouth[crossesZero == False]
    crossesZero[crossesZero == False] |= crossesEast[crossesZero == False]
    crossesZero[crossesZero == False] |= crossesWest[crossesZero == False]

    return crossesZero




def simonVision(originalImage, statsDict, threadPool=None,):
    # extract info from image
    height, width, channels = originalImage.shape
    totalPixels = width * height

    # Normalization, Greycale, Blur
    t0 = cv.getTickCount()
    # normalizedImage = originalImage.astype(np.float32) / 255
    # greyscale = cv.cvtColor(normalizedImage, cv.COLOR_BGR2GRAY)
    greyscale = originalImage[:,:,0].astype(np.float32) / 255
    t1 = cv.getTickCount()
    statsDict["pre-processing"] += (t1-t0)/cv.getTickFrequency()


    # Calculate gradient, angles, and magnitudes.
    # grad has dimensions of [intensity / distance]
    t0 = cv.getTickCount()
    gradX, gradY = getGradient(greyscale, threadPool)
    t1 = cv.getTickCount()
    statsDict["grad"] += (t1-t0)/cv.getTickFrequency()


    t0 = cv.getTickCount()
    gradMags, gradAngles = cv.cartToPolar(gradX, gradY)
    t1 = cv.getTickCount()
    statsDict["mags and angles"] += (t1-t0)/cv.getTickFrequency()


    t0 = cv.getTickCount()
    if (threadPool is None):
        gradDirectionX = helpfulDivide(gradX, gradMags)
        gradDirectionY = helpfulDivide(gradY, gradMags)
    else:
        gradDirectionX_inProgress = threadPool.submit(helpfulDivide, gradX, gradMags)
        gradDirectionY_inProgress = threadPool.submit(helpfulDivide, gradY, gradMags)
        gradDirectionX = gradDirectionX_inProgress.result()
        gradDirectionY = gradDirectionY_inProgress.result()
    t1 = cv.getTickCount()
    statsDict["division"] += (t1-t0)/cv.getTickFrequency()

    # now directional derivative tests
    t0 = cv.getTickCount()
    directionalDerivative = getDirectionalDerivative(gradMags, gradDirectionX, gradDirectionY, threadPool)
    t1 = cv.getTickCount()
    statsDict["1st Direcitonal Derivative"] += (t1-t0)/cv.getTickFrequency()

    # t0 = cv.getTickCount()
    # directionalConcavity = getDirectionalDerivative(directionalDerivative, gradDirectionX, gradDirectionY, threadPool)
    # t1 = cv.getTickCount()
    # statsDict["2nd Direcitonal Derivative THIS ONE"] += (t1-t0)/cv.getTickFrequency()

    # level sets
    # t0 = cv.getTickCount()
    # peaksTroughsFlatlines = getLevelSets(directionalDerivative, 0)
    # t1 = cv.getTickCount()
    # print("og level sets:", (t1-t0)/cv.getTickFrequency())

    # t0 = cv.getTickCount()
    # peaksTroughsFlatlines = threadedLevelSets(directionalDerivative, 0, threadPool)
    # t1 = cv.getTickCount()
    # print("threaded level sets:", (t1-t0)/cv.getTickFrequency())

    # t0 = cv.getTickCount()
    # peaksTroughsFlatlines = numericLevelSets(directionalDerivative, 0)
    # t1 = cv.getTickCount()
    # print("numeric level sets:", (t1-t0)/cv.getTickFrequency())


    # t2 = cv.getTickCount()
    # peaksTroughsFlatlines = threadedNumericLevelSets(directionalDerivative, 0, threadPool)
    # t3 = cv.getTickCount()
    # statsDict["threaded numeric level sets THIS ONE"] += (t3-t2)/cv.getTickFrequency()

    # statsDict["total"] += (t3-t0)/cv.getTickFrequency()

    t0 = cv.getTickCount()
    if (threadPool == None):
        directionalConcavity = getDirectionalDerivative(directionalDerivative, gradDirectionX, gradDirectionY, threadPool)
        peaksTroughsFlatlines = numericLevelSets(directionalDerivative, 0)
    else:
        directionalFuture = threadPool.submit(getDirectionalDerivative, directionalDerivative, gradDirectionX, gradDirectionY, threadPool)
        levelSetFuture = threadPool.submit(threadedNumericLevelSets, directionalDerivative, 0, threadPool)
        # levelSetFuture = threadPool.submit(numericLevelSets, directionalDerivative, 0)
        directionalConcavity = directionalFuture.result()
        peaksTroughsFlatlines = levelSetFuture.result()
    t1 = cv.getTickCount()
    statsDict["combo meal"] += (t1-t0)/cv.getTickFrequency()

    # t0 = cv.getTickCount()
    # peaksTroughsFlatlines = lessThreadedNumericLevelSets(directionalDerivative, 0, threadPool)
    # t1 = cv.getTickCount()
    # print("less threaded numeric level sets:", (t1-t0)/cv.getTickFrequency())

    # t0 = cv.getTickCount()
    # peaksTroughsFlatlines = shortCircuitLevelSets(directionalDerivative, 0)
    # t1 = cv.getTickCount()
    # print("short circuit level sets:", (t1-t0)/cv.getTickFrequency())

    # level sets can happen at the same time as concavity!!!!
    # gotta check about adding a task from within a thread though...

    


    # t0 = cv.getTickCount()
    # directionalDerivative = threadedDirectionalDerivative(gradMags, gradDirectionX, gradDirectionY)
    # directionalConcavity = threadedDirectionalDerivative(directionalDerivative, gradDirectionX, gradDirectionY)
    # t1 = cv.getTickCount()
    # print("threadPooledDirectional:", (t1-t0)/cv.getTickFrequency())

    # t0 = cv.getTickCount()
    # directionalDerivative = dumbDirectionalDerivative(gradMags, gradDirectionX, gradDirectionY)
    # directionalConcavity = dumbDirectionalDerivative(directionalDerivative, gradDirectionX, gradDirectionY)
    # t1 = cv.getTickCount()
    # print("dumbDirectional:", (t1-t0)/cv.getTickFrequency())

    # t0 = cv.getTickCount()
    # showMe = getVectorFieldViz(gradMags / math.sqrt(0.5), gradAngles)
    # t1 = cv.getTickCount()
    # statsDict["viz"] += (t1-t0)/cv.getTickFrequency()
    showMe = greyscale
    return showMe


def main():
    # start the camera
    camera = cv.VideoCapture(0, cv.CAP_V4L2)

    # set picture size and fps
    # 1280 x 800 (MJPG 100/120, YUYV 10)
    # 1280 x 720 (MJPG 100/120, YUYV 10)
    #  800 x 600 (MJPG 100/120)
    #  640 x 480 (MJPG 100/120)
    #  320 x 240 (MJPG 100/120)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
    camera.set(cv.CAP_PROP_FPS, 120)

    # request MJPG format for video
    desiredFourcc = cv.VideoWriter.fourcc('M','J','P','G')
    camera.set(cv.CAP_PROP_FOURCC, desiredFourcc)

    # TODO: other configs
    #       exposure, whiteBalance, gain, etc.

    frameWidth = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    targetFPS = int(camera.get(cv.CAP_PROP_FPS))
    frameSize = (frameHeight, frameWidth)

    fourccInt = int(camera.get(cv.CAP_PROP_FOURCC))
    fourccBytes = fourccInt.to_bytes(length=4, byteorder='little')
    fourccString = fourccBytes.decode()

    # read back configs to confirmt they were set correctly
    print("Starting CircuitVision on "+str(frameWidth)+"x"+str(frameHeight)+" frames @ "+str(targetFPS)+" FPS")


    # networkTables = NetworkTableInstance.getDefault()
    # networkTables.startServer()
    # framePutter = CameraServer.putVideo("My Stream", frameWidth, frameHeight)
    # framePutter.putFrame(np.ones(frameSize, dtype=np.float32))

    print("openCV Verison:", cv.__version__)
    print("numpyVersion:", np.__version__)
    print("CV Build:", cv.getBuildInformation())

    with ThreadPoolExecutor(max_workers=8) as myThreadPool:
        stats = defaultdict(lambda:0)
        iterationCount = 0
        warmupPeriod = 10
        programStart = cv.getTickCount()
        while (True):
            # get the next frame from the camera
            _, frame = camera.read()

            pipelineStart = cv.getTickCount()
            showMe = simonVision(frame, stats, myThreadPool)
            # testOpenCVImpl(frame)
            pipelineEnd = cv.getTickCount()

            # record stats for display
            pipelineDeltaT = (pipelineEnd - pipelineStart) / cv.getTickFrequency() # seconds
            pipelineFrequency = 1/pipelineDeltaT
            
            iterationCount += 1
            totalRuntime = (cv.getTickCount() - programStart) / cv.getTickFrequency()
            if (int(totalRuntime) % 2 != 0):
                continue
            
            print("total runtime (s):", totalRuntime)
            avgCumulativeTime = 0
            for nameOfStep, totalRuntimeOfStep in stats.items():
                avgRuntimeOfStep = totalRuntimeOfStep / iterationCount
                avgCumulativeTime += avgRuntimeOfStep

                totalCharacters = 30
                labelLength = len(nameOfStep)
                leftoverCharacters = totalCharacters - labelLength
                print((" "*leftoverCharacters)+nameOfStep+" (ms):", round(avgRuntimeOfStep*1000, 3), "    cum (ms):", round(avgCumulativeTime*1000, 3))
            
            avgLoopTime = avgCumulativeTime
            avgFPS = 1/avgLoopTime
            print("current runtime (ms):", round(pipelineDeltaT*1000, 3), " avg runtime (ms):", round(avgLoopTime*1000, 3))
            print("current FPS:", round(pipelineFrequency, 3), " avg FPS:", round(avgFPS, 3))
            print()
            # framePutter.putFrame(showMe)

            if ((totalRuntime >= warmupPeriod) and (warmupPeriod > 0)):
                # reset stats after warmup
                iterationCount = 0
                warmupPeriod = -1
                stats = defaultdict(lambda:0)




def testOpenCVImpl(image):
    tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # detectorParameters = 
    # detector = cv.aruco.ArucoDetector(dictionary=tagFamily)
    corners, ids, _ = cv.aruco.detectMarkers(image, tagFamily)

    # drawOnMe = image.copy()
    # drawOnMe = cv.aruco.drawDetectedMarkers(drawOnMe, corners, ids)
    # cv.imshow("OpenCV Detector", drawOnMe)

main()
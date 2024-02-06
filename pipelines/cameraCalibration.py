# OpenCV
import numpy as np
import cv2 as cv

# wpiLib
from cscore import CameraServer
from ntcore import NetworkTableInstance
# import wpilib

framePutter = None


def main():
    # start the camera
    camera = cv.VideoCapture(0, cv.CAP_V4L2)

    # request MJPG format for video
    desiredFourcc = cv.VideoWriter.fourcc('M','J','P','G')
    camera.set(cv.CAP_PROP_FOURCC, desiredFourcc)

    # set picture size and fps
    # 1280 x 800 (MJPG 100/120, YUYV 10)
    # 1280 x 720 (MJPG 100/120, YUYV 10)
    #  800 x 600 (MJPG 100/120)
    #  640 x 480 (MJPG 100/120)
    #  320 x 240 (MJPG 100/120)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
    camera.set(cv.CAP_PROP_FPS, 120)

    # TODO: other configs
    #       exposure, whiteBalance, gain, etc.

    # read back configs to confirm they were set correctly
    frameWidth = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    targetFPS = int(camera.get(cv.CAP_PROP_FPS))

    fourccInt = int(camera.get(cv.CAP_PROP_FOURCC))
    fourccBytes = fourccInt.to_bytes(length=4, byteorder='little')
    fourccString = fourccBytes.decode()

    global framePutter
    networkTables = NetworkTableInstance.getDefault()
    # start NetworkTables
    onRobot = False
    if (not(onRobot)):
        print("Starting CircuitVision in Desktop Mode")
        networkTables.startServer()
    else:
        print("Starting CircuitVision in Robot Mode")
        networkTables.startClient4("wpilibpi")
        networkTables.setServerTeam(1787)
        #networkTables.startDSClient()
    framePutter = CameraServer.putVideo("My Stream", frameWidth, frameHeight)
    framePutter.putFrame(np.full((frameHeight, frameWidth), 255, dtype=np.uint8))


    print("Using "+str(frameWidth)+"x"+str(frameHeight)+" frames @ "+str(targetFPS)+" FPS")
    print("openCV Verison:", cv.__version__)
    print("numpyVersion:", np.__version__)
    # print("CV Build:", cv.getBuildInformation())


    # networktables interface for capturing data
    networkTable = networkTables.getTable("cameraCalibration")
    subscriber = networkTable.getDoubleTopic("frameCapture").subscribe(0)
    publisher = networkTable.getDoubleTopic("frameCapture").publish()
    publisher.set(50) # put the topic on the dashboard
    subscriber.readQueue() # read the value I just put so count doesn't trigger on first sight
    captureCountPub = networkTable.getDoubleTopic("captureCount").publish()

    # publishers for calibration output
    centerXPub = networkTable.getDoubleTopic("centerX").publish()
    centerYPub = networkTable.getDoubleTopic("centerY").publish()
    focalXPub = networkTable.getDoubleTopic("focalX").publish()
    focalYPub = networkTable.getDoubleTopic("focalY").publish()
    rmsErrorPub = networkTable.getDoubleTopic("rmsReprojectError").publish()
    




    #
    # START CALIBRATION
    #

    # A "chessboard point" is a place where two black squares touch
    chessboardPointsPerRow = 9
    chessboardPointsPerCol = 6
    patternCols = chessboardPointsPerRow # each row has all columns
    patternRows = chessboardPointsPerCol # each col as all rows
    patternShape = (patternCols, patternRows)
    distanceBetweenCornersMeters = 1 #3/100
    localChessBoardCoordinates = []
    for r in range(patternRows):
        for c in range(patternCols):
            x = c * distanceBetweenCornersMeters
            y = r * distanceBetweenCornersMeters
            z = 0
            localChessBoardCoordinates.append((x, y, z))
    localChessBoardCoordinates = np.array(localChessBoardCoordinates, dtype=np.float32)
    # print("mine:", localChessBoardCoordinates)
    # objp = np.zeros((9*14,3), np.float32)
    # objp[:,:2] = np.mgrid[0:9,0:14].T.reshape(-1,2)
    # print("theirs:", objp)

    while(True):
        # init arrays to keep track of point correspondeneces
        imageCordsOfCorners = []
        correspondingWorldPoints = []
        captureCount = 0
        publisher.set(50)
        subscriber.readQueue()

        while (True):

            # get the next frame from the camera, and convert to greyscale
            _, frame = camera.read()
            greyscale = frame[:,:,0]

            # look for chessboard in the frame
            patternFound, corners = cv.findChessboardCorners(greyscale, patternSize=patternShape)

            if (patternFound):
                # just copying docs
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refinedCorners = cv.cornerSubPix(greyscale, corners, (11,11), (-1,-1), criteria)

                canvas = np.copy(frame)
                cv.drawChessboardCorners(canvas, patternShape, refinedCorners, patternWasFound=True)
                framePutter.putFrame(canvas)
            else:
                canvas = np.copy(frame)
                cv.drawChessboardCorners(canvas, patternShape, corners, patternWasFound=False)
                framePutter.putFrame(canvas)

            # detect a change on the dashboard.
            if (len(subscriber.readQueue()) > 0 and patternFound and subscriber.get() > 0):
                imageCordsOfCorners.append(refinedCorners)
                correspondingWorldPoints.append(localChessBoardCoordinates)
                captureCount += 1

            captureCountPub.set(captureCount)
            if (subscriber.get() < 0):
                framePutter.putFrame(np.full_like(frame, 255))
                print("Done with captures!")
                print("working on camera matrix now (may take a minute...)")
                break
            # testOpenCVImpl(frame)
    
        rmsReprojectError, cameraMatrix, distCoeffs, rotationVecs, translationVecs = cv.calibrateCamera(correspondingWorldPoints, imageCordsOfCorners, greyscale.shape[::-1], None, None)
        # print("camera matrix (units are pixels):", cameraMatrix)
        # print("rmsReprojectError:", rmsReprojectError)

        centerXPub.set(cameraMatrix[0, 2])
        centerYPub.set(cameraMatrix[1, 2])
        focalXPub.set(cameraMatrix[0, 0])
        focalYPub.set(cameraMatrix[1, 1])
        rmsErrorPub.set(rmsReprojectError)
        # finish inner loop, then start new calibration in outer loop




def testOpenCVImpl(image):
    tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # detectorParameters = 
    # detector = cv.aruco.ArucoDetector(dictionary=tagFamily)
    corners, ids, _ = cv.aruco.detectMarkers(image, tagFamily)

    drawOnMe = image.copy()
    drawOnMe = cv.aruco.drawDetectedMarkers(drawOnMe, corners, ids)
    framePutter.putFrame(drawOnMe)

main()

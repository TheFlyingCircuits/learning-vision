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
    onRobot = True
    if (not(onRobot)):
        print("Starting CircuitVision in Desktp Mode")
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

    table = networkTables.getTable("myTable")
    subscriber = table.getDoubleTopic("myTopic").subscribe(0)
    publisher = table.getDoubleTopic("myTopic").publish()
    publisher.set(50) # put the topic on the dashboard
    subscriber.readQueue() # read the value I just put so count doesn't trigger on first sight
    captureCountPub = table.getDoubleTopic("captureCount").publish()



    #
    # START CALIBRATION
    #

    # A "chessboard point" is a place where two black squares touch
    chessboardPointsPerRow = 7
    chessboardPointsPerCol = 5
    distanceBetweenCornersMeters = 1 #3/100
    localChessBoardCoordinates = []
    for r in range(chessboardPointsPerRow):
        for c in range(chessboardPointsPerCol):
            x = r * distanceBetweenCornersMeters
            y = c * distanceBetweenCornersMeters
            z = 0
            localChessBoardCoordinates.append((x, y, z))
    localChessBoardCoordinates = np.array(localChessBoardCoordinates, dtype=np.float32)

    # init arrays to keep track of point correspondences
    imageCordsOfCorners = []
    correspondingWorldPoints = []

    captureCount = 0
    while(True):
        # get the next frame from the camera, and convert to greyscale
        _, frame = camera.read()
        greyscale = frame[:,:,0]

        # look for chessboard in the frame
        patternFound, corners = cv.findChessboardCorners(greyscale, patternSize=(chessboardPointsPerRow, chessboardPointsPerCol))

        if (patternFound):
            # just copying docs
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refinedCorners = cv.cornerSubPix(greyscale, corners, (11,11), (-1,-1), criteria)

            canvas = np.copy(frame)
            cv.drawChessboardCorners(canvas, (chessboardPointsPerRow, chessboardPointsPerCol), refinedCorners, patternWasFound=True)
            framePutter.putFrame(cv.flip(canvas, 1))
        else:
            canvas = np.copy(frame)
            cv.drawChessboardCorners(canvas, (chessboardPointsPerRow, chessboardPointsPerCol), corners, patternWasFound=False)
            framePutter.putFrame(cv.flip(canvas, 1))

        # detect a change on the dashboard.
        if (len(subscriber.readQueue()) > 0 and patternFound):
            imageCordsOfCorners.append(refinedCorners)
            correspondingWorldPoints.append(localChessBoardCoordinates)
            captureCount += 1

        captureCountPub.set(captureCount)
        if (subscriber.get() < 0):
            print("Done with captures!")
            print("working on camera matrix now (may take a minute...)")
            break
        # testOpenCVImpl(frame)
    
    rmsReprojectError, cameraMatrix, distCoeffs, rotationVecs, translationVecs = cv.calibrateCamera(correspondingWorldPoints, imageCordsOfCorners, greyscale.shape[::-1], None, None)
    print("camera matrix (units are pixels):", cameraMatrix)
    print("rmsReprojectError:", rmsReprojectError)



def testOpenCVImpl(image):
    tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # detectorParameters = 
    # detector = cv.aruco.ArucoDetector(dictionary=tagFamily)
    corners, ids, _ = cv.aruco.detectMarkers(image, tagFamily)

    drawOnMe = image.copy()
    drawOnMe = cv.aruco.drawDetectedMarkers(drawOnMe, corners, ids)
    framePutter.putFrame(drawOnMe)

main()

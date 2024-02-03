# OpenCV
import numpy as np
import cv2 as cv

# wpiLib
from cscore import CameraServer
from ntcore import NetworkTableInstance

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

    frameWidth = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    targetFPS = int(camera.get(cv.CAP_PROP_FPS))

    fourccInt = int(camera.get(cv.CAP_PROP_FOURCC))
    fourccBytes = fourccInt.to_bytes(length=4, byteorder='little')
    fourccString = fourccBytes.decode()

    # read back configs to confirm they were set correctly
    print("Starting CircuitVision on "+str(frameWidth)+"x"+str(frameHeight)+" frames @ "+str(targetFPS)+" FPS")

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
        networkTables.startDSClient()
    framePutter = CameraServer.putVideo("My Stream", frameWidth, frameHeight)
    framePutter.putFrame(np.full((frameHeight, frameWidth), 255, dtype=np.uint8))

    print("openCV Verison:", cv.__version__)
    print("numpyVersion:", np.__version__)
    # print("CV Build:", cv.getBuildInformation())

    imageCordsOfCorners = []
    worldCordsOfCorners = []

    chessboardSize = (7, 5) # internal corners
    # TODO: SHOLD BE REAL WORLD UNITS????
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    planarCords = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    planarCords[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    captureCount = 0

    while (captureCount < 20):
        # get the next frame from the camera
        _, frame = camera.read()

        greyscale = frame[:,:,0]

        patternFound, corners = cv.findChessboardCorners(greyscale, patternSize=chessboardSize)

        if (patternFound):
            # just copying docs
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refinedCorners = cv.cornerSubPix(greyscale, corners, (11,11), (-1,-1), criteria)

            imageCordsOfCorners.append(refinedCorners)
            worldCordsOfCorners.append(planarCords)

            canvas = np.copy(frame)
            cv.drawChessboardCorners(canvas, chessboardSize, refinedCorners, patternWasFound=True)
            framePutter.putFrame(canvas)
            captureCount += 1
        else:
            canvas = np.copy(frame)
            cv.drawChessboardCorners(canvas, chessboardSize, corners, patternWasFound=False)
            framePutter.putFrame(canvas)


        # testOpenCVImpl(frame)
    
    returnVal, cameraMatrix, distCoeffs, rotationVecs, translationVecs = cv.calibrateCamera(worldCordsOfCorners, imageCordsOfCorners, greyscale.shape[::-1], None, None)
    print("camera matrix:", cameraMatrix)
    print("WHAT ARE THE UNITS?")




def testOpenCVImpl(image):
    tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # detectorParameters = 
    # detector = cv.aruco.ArucoDetector(dictionary=tagFamily)
    corners, ids, _ = cv.aruco.detectMarkers(image, tagFamily)

    drawOnMe = image.copy()
    drawOnMe = cv.aruco.drawDetectedMarkers(drawOnMe, corners, ids)
    framePutter.putFrame(drawOnMe)

main()

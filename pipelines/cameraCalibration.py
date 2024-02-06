# OpenCV
import numpy as np
import cv2 as cv

# wpiLib
from cscore import CameraServer
from ntcore import NetworkTableInstance

def initCamera():
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

    return camera, frameWidth, frameHeight, targetFPS

def initNetworkTables():
    networkTables = NetworkTableInstance.getDefault()

    onRobot = False
    if (onRobot):
        print("Starting CircuitVision in Robot Mode")
        networkTables.startClient4("wpilibpi")
        networkTables.setServerTeam(1787)
        # networkTables.startDSCLient()
    else:
        print("Starting CircuitVision in Desktop Mode")
        networkTables.startServer()
    
    return networkTables


framePutter = None

def main():
    networkTables = initNetworkTables()
    camera, frameWidth, frameHeight, targetFPS = initCamera()

    # setup button to switch between calibration mode and AR mode
    networkTable = networkTables.getTable("cameraCalibration")
    modeButtonPub = networkTable.getBooleanTopic("captureMode").publish()
    modeButtonSub = networkTable.getBooleanTopic("captureMode").subscribe(True)
    modeButtonPub.set(True) # True = is calibrating, False = visualizing (not calibrating)

    # setup button to capture calibration data
    captureButtonPub = networkTable.getBooleanTopic("capture").publish()
    captureButtonSub = networkTable.getBooleanTopic("capture").subscribe(False)
    captureButtonPub.set(False)

    # init camera stream for viz
    global framePutter
    framePutter = CameraServer.putVideo("My Stream", frameWidth, frameHeight)
    framePutter.putFrame(np.full((frameHeight, frameWidth), 255, dtype=np.uint8))

    print("Using "+str(frameWidth)+"x"+str(frameHeight)+" frames @ "+str(targetFPS)+" FPS")
    print("openCV Verison:", cv.__version__)
    print("numpyVersion:", np.__version__)
    # print("CV Build:", cv.getBuildInformation())

    # init publishers for calibration info:
    captureCountPub = networkTable.getDoubleTopic("captureCount").publish()
    centerXPub = networkTable.getDoubleTopic("centerX").publish()
    centerYPub = networkTable.getDoubleTopic("centerY").publish()
    focalXPub = networkTable.getDoubleTopic("focalX").publish()
    focalYPub = networkTable.getDoubleTopic("focalY").publish()
    rmsErrorPub = networkTable.getDoubleTopic("rmsReprojectError").publish()

    # init variables for storing calibration info
    cameraMatrix = None
    distortionFactors = None
    imageCordsOfCorners = []
    correspondingWorldPoints = []

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
    
    # start grabbing frames
    while (True):
        # get the next frame from the camera, and convert to greyscale
        _, frame = camera.read()
        greyscale = frame[:,:,0]

        # read dashboard buttons
        modeButtonChanged = len(modeButtonSub.readQueue()) > 0
        isCaptureMode = modeButtonSub.get()
        captureButtonClicked = (captureButtonSub.get() == True)
        if (captureButtonClicked):
            # auto toggle off after click
            captureButtonPub.set(False)

        if (modeButtonChanged and isCaptureMode):
            imageCordsOfCorners = []
            correspondingWorldPoints = []
        elif (isCaptureMode): # in the middle of calibration mode
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

            # detect a click of the capture button on the dashboard.
            if (captureButtonClicked and patternFound):
                imageCordsOfCorners.append(refinedCorners)
                correspondingWorldPoints.append(localChessBoardCoordinates)
        elif (modeButtonChanged and not(isCaptureMode)):
            # transition into AR mode. Update camera calibration info
            rmsReprojectError, cameraMatrix, distortionFactors, rotationVecs, translationVecs = cv.calibrateCamera(correspondingWorldPoints, imageCordsOfCorners, greyscale.shape[::-1], None, None)

            centerXPub.set(cameraMatrix[0, 2])
            centerYPub.set(cameraMatrix[1, 2])
            focalXPub.set(cameraMatrix[0, 0])
            focalYPub.set(cameraMatrix[1, 1])
            rmsErrorPub.set(rmsReprojectError)
        elif (not(isCaptureMode)):
            testOpenCVImpl(frame, cameraMatrix)


        captureCountPub.set(len(correspondingWorldPoints))
        # captureButtonPub.set(False)





def testOpenCVImpl(image, cameraMatrix):
    # https://docs.opencv.org/4.6.0/d5/dae/tutorial_aruco_detection.html
    tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16h5)
    # detectorParameters = 
    # detector = cv.aruco.ArucoDetector(dictionary=tagFamily)

    # corners is a list of lists
    # within each sub list, there are 4 corners, ordered clockwise
    # starting corner is unknown?
    corners, ids, _ = cv.aruco.detectMarkers(image, tagFamily)
    if ((ids is None) or len(ids) == 0):
        framePutter.putFrame(image)
        return

    tagWidthInches = 6.5
    metersPerInch = 0.0254
    tagWidthMeters = tagWidthInches * metersPerInch
    distortionCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32) # assume negligible lens distoriton
    rotationVectors, translationVectors, _ = cv.aruco.estimatePoseSingleMarkers(corners, tagWidthMeters, cameraMatrix, distortionCoeffs, )

    drawOnMe = image.copy()
    drawOnMe = cv.aruco.drawDetectedMarkers(drawOnMe, corners, ids)
    howLongToDrawAxesMeters = tagWidthMeters
    drawOnMe = cv.drawFrameAxes(drawOnMe, cameraMatrix, distortionCoeffs, rotationVectors, translationVectors, howLongToDrawAxesMeters, thickness=2)
    # (x,y,z) -> (r,g,b)
    framePutter.putFrame(drawOnMe)

main()

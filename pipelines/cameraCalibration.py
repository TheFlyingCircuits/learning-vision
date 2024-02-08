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

    onRobot = True
    if (onRobot):
        print("Starting CircuitVision in Robot Mode")
        networkTables.startClient4("wpilibpi")
        networkTables.setServerTeam(1787)
        # networkTables.startDSCLient()
    else:
        print("Starting CircuitVision in Desktop Mode")
        networkTables.startServer()
    
    return networkTables


class CameraCalibrator:

    def __init__(self, imageShape, chessboardPointsPerRow, chessboardPointsPerCol, distanceBetweenCorners=1):
        # record image dimensions for later
        # we us numpy convention shape = (rows, cols) = (height, width)
        self.imageShape = imageShape

        # A "chessboard point" is a place where two black squares touch
        patternCols = chessboardPointsPerRow # each row has all columns
        patternRows = chessboardPointsPerCol # each column has all rows
        self.patternShape = (patternCols, patternRows)
        self.distanceBetweenCorners = distanceBetweenCorners

        # generate the coordinates of each chessboard corner
        # as viewed from the chessboard's frame of reference
        # (top left corner as origin, x axis along the rows, y axis along the columns)
        # TODO: understand why the distance between corners isn't really necessary.
        localChessBoardCoordinates = []
        for r in range(patternRows):
            for c in range(patternCols):
                x = c * distanceBetweenCorners
                y = r * distanceBetweenCorners
                z = 0
                localChessBoardCoordinates.append((x, y, z))
        self.localChessBoardCoordinates = np.array(localChessBoardCoordinates, dtype=np.float32)

        # init arrays for keeping track of point correspondences
        self.imageCordsOfCorners = []
        self.correspondingWorldPoints = []

        # init the outputs (default to some already taken values)
        self.cameraMatrix = None
        self.distortionCoefficients = None
        self.rmsReprojectionError = None

    
    def findCorners(self, greyscaleImage):
        patternFound, cornerCords = cv.findChessboardCorners(greyscaleImage, patternSize=self.patternShape)
        if (patternFound):
            return cornerCords
        else:
            return None


    def addMeasurement(self, greyscaleImage):
        # refine corners, then add the correspondences
        cornerCords = self.findCorners(greyscaleImage)
        
        # just copying docs, idk what any of this means
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refinedCornerCords = cv.cornerSubPix(greyscaleImage, cornerCords, (11,11), (-1,-1), criteria)

        self.imageCordsOfCorners.append(refinedCornerCords)
        self.correspondingWorldPoints.append(self.localChessBoardCoordinates)

        return refinedCornerCords

    def updateCalibration(self):
        if (len(self.correspondingWorldPoints) == 0):
            self.setDefaultCalibration()
            return
        
        threeDeePoints = self.correspondingWorldPoints
        twoDeeProjections = self.imageCordsOfCorners
        calibrationOutputs = cv.calibrateCamera(threeDeePoints, twoDeeProjections, self.imageShape[::-1], None, None)

        # unpack calibration outputs
        self.rmsReprojectionError = calibrationOutputs[0]
        self.cameraMatrix = calibrationOutputs[1]
        self.distortionCoefficients = calibrationOutputs[2]

        # not used, but here incase I want them later.
        rotationVecs = calibrationOutputs[3]
        translationVecs = calibrationOutputs[4]

    def setDefaultCalibration(self):
        # default values based on previously taken measurements
        fx = 922.404080591854 # focal length measured in terms of the width of a pixel
        fy = 917.761755502239 # focal length measured in terms of the height of a pixel
        cx = 605.0506860423618 # the column in the image that's aligned with the optical axis
        cy = 405.32112366927157 # the row in the image that's aligned with the optical axis
        cameraMatrix = np.array([ [fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]
                                ], dtype=np.float32)
        
        rmsReprojectionError = 0.3495341197583
        distortionCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        self.cameraMatrix = cameraMatrix
        self.rmsReprojectionError = rmsReprojectionError
        self.distortionCoefficients = distortionCoeffs
    
    def publishToNetworkTables(self):

        if not(hasattr(self, 'networkTable')):
            networkTables = NetworkTableInstance.getDefault()
            self.networkTable = networkTables.getTable("cameraCalibration")

            # maybe a calibration object is itself an input to a networking class instead?
            # that feels like bettern seperation of concerns, but this feels like better
            # encapsulaiton.
            self.captureCountPub = self.networkTable.getDoubleTopic("captureCount").publish()
            self.centerXPub = self.networkTable.getDoubleTopic("centerX").publish()
            self.centerYPub = self.networkTable.getDoubleTopic("centerY").publish()
            self.focalXPub = self.networkTable.getDoubleTopic("focalX").publish()
            self.focalYPub = self.networkTable.getDoubleTopic("focalY").publish()
            self.rmsErrorPub = self.networkTable.getDoubleTopic("rmsReprojectError").publish()

        self.captureCountPub.set(len(self.correspondingWorldPoints))

        if (self.cameraMatrix is None):
            self.centerXPub.set(0)
            self.centerYPub.set(0)
            self.focalXPub.set(0)
            self.focalYPub.set(0)
            self.rmsErrorPub.set(0)
            return
        
        
        self.centerXPub.set(self.cameraMatrix[0, 2])
        self.centerYPub.set(self.cameraMatrix[1, 2])
        self.focalXPub.set(self.cameraMatrix[0, 0])
        self.focalYPub.set(self.cameraMatrix[1, 1])
        self.rmsErrorPub.set(self.rmsReprojectionError)



framePutter = None

def main():
    networkTables = initNetworkTables()
    camera, frameWidth, frameHeight, targetFPS = initCamera()

    # create a button to switch between calibration mode and AR mode
    networkTable = networkTables.getTable("cameraCalibration")
    modeButtonPub = networkTable.getBooleanTopic("captureMode").publish()
    modeButtonSub = networkTable.getBooleanTopic("captureMode").subscribe(True)
    modeButtonPub.set(True) # True = is calibrating, False = visualizing (not calibrating)
    modeButtonSub.readQueue() # read back the value we just published so we don't detect a change in mode on the first loop

    # create a button to capture calibration data
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

    # init publishers for tag info
    tagXPub = networkTable.getDoubleTopic("tagXmeters").publish()
    tagYPub = networkTable.getDoubleTopic("tagYmeters").publish()
    tagZPub = networkTable.getDoubleTopic("tagZmeters").publish()
    freedomXPub = networkTable.getDoubleTopic("tagXinches").publish()
    freedomYPub = networkTable.getDoubleTopic("tagYinches").publish()
    freedomZPub = networkTable.getDoubleTopic("tagZinches").publish()

    rotationXPub = networkTable.getDoubleArrayTopic("xHatCords").publish()
    rotationYPub = networkTable.getDoubleArrayTopic("yHatCords").publish()
    rotationZPub = networkTable.getDoubleArrayTopic("zHatCords").publish()

    
    # start grabbing frames
    cameraCalibrator = CameraCalibrator((frameHeight, frameWidth), 9, 6)
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



        # reset calibration data when switching into calibration mode
        if (modeButtonChanged and isCaptureMode):
            cameraCalibrator = CameraCalibrator((frameHeight, frameWidth), 9, 6)
        # add calibration data while in calibration mode
        elif (isCaptureMode):
            corners = cameraCalibrator.findCorners(greyscale)

            if (corners is None):
                framePutter.putFrame(frame)
            elif (captureButtonClicked):
                cameraCalibrator.addMeasurement(greyscale)
            else:
                canvas = np.copy(frame)
                cv.drawChessboardCorners(canvas, cameraCalibrator.patternShape, corners, True)
        # update intrinsics when switching out of calibration mode
        elif (modeButtonChanged and not(isCaptureMode)):
            cameraCalibrator.updateCalibration()
        elif (not(isCaptureMode)):
            translationVector, rotationMatrix = testOpenCVImpl(frame, cameraCalibrator.cameraMatrix)
            x, y, z = translationVector

            # only show out to millimeters/thousandths of an inch.
            # anything more than that is probably overkill / just noise.
            tagXPub.set(round(x, 3))
            tagYPub.set(round(y, 3))
            tagZPub.set(round(z, 3))
            inchesPerMeter = 1/0.0254
            freedomXPub.set(round(inchesPerMeter * x, 3))
            freedomYPub.set(round(inchesPerMeter * y, 3))
            freedomZPub.set(round(inchesPerMeter * z, 3))

            rotationXPub.set(rotationMatrix[:, 0])
            rotationYPub.set(rotationMatrix[:, 1])
            rotationZPub.set(rotationMatrix[:, 2])

        cameraCalibrator.publishToNetworkTables()


def findTagCorners(image):
    # https://docs.opencv.org/4.6.0/d5/dae/tutorial_aruco_detection.html
    tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    # tagFamily = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16h5)

    # don't use the default of no corner refinement! we want precision!
    # https://docs.opencv.org/4.6.0/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
    detectorParameters = cv.aruco.DetectorParameters.create()
    detectorParameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG

    # corners is a list of 4x2 numpy arrays
    # within each numpy array, there are 4 corners (xCord(column) followed by yCord(row)), ordered clockwise
    # [topLeft, topRight, bottomRight, bottomLeft] (source is 1st link, not mentioned in function docs (ugh))
    # 
    # experiment differs from documentation.
    # experiment says corner order is [bottomRight, bottomLeft, topLeft, topRight]
    # will look into fixing this later. for now, it all works.
    stupidCorners, stupidIds, cornersOfRejectedCandidates = cv.aruco.detectMarkers(image, tagFamily, parameters=detectorParameters)


    corners = []
    ids = []
    if (stupidIds is None):
        return corners, ids, image
    
    # unpack data in a format that actually makes sense
    # for some reason, they keep the actual values we want
    # within a list of size 1, so we pull them out here
    # so we don't have an annoying redundant index.
    for i in range(len(stupidIds)):
        actualId = stupidIds[i][0]
        actualCorners = stupidCorners[i][0]
        ids.append(actualId)
        corners.append(actualCorners)

    for i in range(len(corners)):
        fixedCorners = np.copy(corners[i])

        fixedCorners[0] = corners[i][2]
        fixedCorners[1] = corners[i][3]
        fixedCorners[2] = corners[i][0]
        fixedCorners[3] = corners[i][1]

        corners[i] = fixedCorners

        # put corners into right order convention
        # so that axes are correct?


    drawOnMe = image.copy()
    annotatedTags = cv.aruco.drawDetectedMarkers(drawOnMe, stupidCorners, stupidIds)
    return corners, ids, annotatedTags
        







def testOpenCVImpl(image, cameraMatrix):
    tagWidthInches = 6.5 # 6.5 for 36h11, 6 for 16h5
    metersPerInch = 0.0254
    tagWidthMeters = tagWidthInches * metersPerInch

    # corners is a list of 4x2 numpy arrays
    corners, ids, drawOnMe = findTagCorners(image)
    if (len(ids) == 0):
        framePutter.putFrame(image)
        return (0, 0, 0), np.zeros((3, 3))
    
    rotationVectors = [] # direction = rotation axis, magnitude = how many radians to rotate about that axis
    translationVectors = [] # units of meters
    localX = tagWidthMeters / 2
    localY = tagWidthMeters / 2
    localTagCords = np.array([(-localX, localY, 0), (localX, localY, 0), (localX, -localY, 0), (-localX, -localY, 0)], dtype=np.float32) # match order convention from above

    for setOfFourCorners in corners:
        distortionCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32) # assume negligible lens distoriton
        undocumentedReturnVal, stupidRotationVector, stupidTranslationVector = cv.solvePnP(localTagCords, setOfFourCorners, cameraMatrix, distortionCoeffs, flags=cv.SOLVEPNP_IPPE_SQUARE)

        # don't want a list of single element lists, we just want the cords
        # why do they do it like that????
        rotationVector = np.array([stupidRotationVector[0][0], stupidRotationVector[1][0], stupidRotationVector[2][0]])
        translationVector = np.array([stupidTranslationVector[0][0], stupidTranslationVector[1][0], stupidTranslationVector[2][0]])

        rotationVectors.append(rotationVector)
        translationVectors.append(translationVector)


    # create viz
    howLongToDrawAxesMeters = tagWidthMeters
    for i in range(len(ids)):
        rotationVector = rotationVectors[i]
        translationVector = translationVectors[i]
        drawOnMe = cv.drawFrameAxes(drawOnMe, cameraMatrix, distortionCoeffs, rotationVector, translationVector, howLongToDrawAxesMeters, thickness=3)
        # (x,y,z) -> (r,g,b)
    framePutter.putFrame(drawOnMe)

    print("\nrotationVectors:", rotationVectors)
    rotationMatrix, _ = cv.Rodrigues(rotationVectors[0])
    print("rotationMatrix:", rotationMatrix)
    return translationVectors[0], rotationMatrix

main()

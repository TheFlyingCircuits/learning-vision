import numpy as np
import cv2 as cv
 
# mouse callback globals
imageTranslationX = 0
imageTranslationY = 0
prevMouseX = -1
prevMouseY = -1
mouseClickX = -1
mouseClickY = -1
zoomFactor = 1

hasCallback = {}
 
def fancyZoomMouseCallback(mouseEvent, x, y, eventFlags, customParam):
    # Declare that we're using some globals
    global imageTranslationX, imageTranslationY, prevMouseX, prevMouseY, zoomFactor
 
    # track changes in mouse movement
    deltaMouseX = x - prevMouseX
    deltaMouseY = y - prevMouseY
    if (prevMouseX < 0 or prevMouseY < 0):
        # set delta to 0 if there isn't a valid prev value
        deltaMouseX = 0
        deltaMouseY = 0
 
    # Save the current mouse location so we can
    # reference it the next time the mouse moves
    prevMouseX = x
    prevMouseY = y
 
    # unpack flags to determine which event
    # we're dealing with.
    leftJustClicked = (mouseEvent == cv.EVENT_LBUTTONDOWN)
    rightJustClicked = (mouseEvent == cv.EVENT_RBUTTONDOWN)
    mouseBeingDragged = (mouseEvent == cv.EVENT_MOUSEMOVE)
    leftBeingHeld = (eventFlags & cv.EVENT_FLAG_LBUTTON) > 0
    rightBeingHeld = (eventFlags & cv.EVENT_FLAG_RBUTTON) > 0
    shiftBeingHeld = (eventFlags & cv.EVENT_FLAG_SHIFTKEY)
    controlBeingHeld = (eventFlags & cv.EVENT_FLAG_CTRLKEY) > 0
 
    if (mouseBeingDragged and leftBeingHeld and (not shiftBeingHeld)):
        # left click and drag to translate the image
        # (dragging 5 pixels on a 5x zoomed image should cause
        #  the original image to shift by 1 pixel)
        # In other words, (1/zoomFactor) has units of [imagePixels/windowPixel]
        imageTranslationX += deltaMouseX / zoomFactor
        imageTranslationY += deltaMouseY / zoomFactor
 
        # get info about window size
        windowName = customParam
        windowRect = cv.getWindowImageRect(windowName)
        width = windowRect[2]
        height = windowRect[3]
 
        # constrain the translation so we don't get lost
        imageTranslationX = np.clip(imageTranslationX, int(-width/2), int(width/2))
        imageTranslationY = np.clip(imageTranslationY, int(-height/2), int(height/2))
 
    elif (mouseBeingDragged and leftBeingHeld and shiftBeingHeld):
        # left click and drag while holding shift to zoom in and out of the image
        # (mouse wheel events only work on windows, and I'm on linux)
        zoomRate = 0.03
        zoomFactor += zoomRate * (-deltaMouseY)
        if (zoomFactor < 1):
            zoomFactor = 1
 
    elif ((leftJustClicked or rightJustClicked) and controlBeingHeld):
        # go back home and reset everything on a shift click
        imageTranslationX = 0
        imageTranslationY = 0
        zoomFactor = 1
        prevMouseX = -1
        prevMouseY = -1
        global mouseClickY,mouseClickX
        mouseClickX = x
        mouseClickY = y
 
 
def showZoomedImage(windowName, preZoomImage):
    # cv.imshow(windowName, preZoomImage)
    # return
    global imageTranslationX, imageTranslationY, zoomFactor
    width, height = preZoomImage.shape[1], preZoomImage.shape[0]
    centerX = width/2
    centerY = height/2
 
    # # adjust the zoom factor based on image size
    # windowRect = cv.getWindowImageRect(windowName)
    # windowWidth = windowRect[2]
    # windowHeight = windowRect[3]
 
    # total translation accounts for the fact that zooming happens from the top left corner
    # by default, but we want it to happen from the center of the window.
    totalTranslationX = (1-zoomFactor) * centerX + (imageTranslationX * zoomFactor)
    totalTranslationY = (1-zoomFactor) * centerY + (imageTranslationY * zoomFactor)
 
    # adapted from the openCV geometric transformations tutorial
    scaleThenTranslateMatrix = np.array([[zoomFactor, 0, totalTranslationX], [0, zoomFactor, totalTranslationY]], dtype=np.float32)
    movedAndZoomedImage = cv.warpAffine(preZoomImage, scaleThenTranslateMatrix, (width, height), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=(128, 128, 128))
 
    # shrink image to fit more on screen?
    # newWidth = int(width / 2)
    # newHeight = int(height / 2)
    # finalImage = cv.resize(movedAndZoomedImage, (newWidth, newHeight))
    # cv.imshow(windowName, finalImage)
    cv.imshow(windowName, movedAndZoomedImage)
 
 
def readBrightnessTrackbars(windowName):
    # helper function for visualizeVectorField()
    maxTrackbarValue = 10000
 
    relativeCeiling=1#  = cv.getTrackbarPos("Magnitude Ceiling", windowName) / maxTrackbarValue
    relativeFloor =0#-= cv.getTrackbarPos("Magnitude Floor", windowName) / maxTrackbarValue
 
    if ((windowName in hasCallback.keys()) and (hasCallback[windowName] == False)):
        hasCallback[windowName] = True
        initialCeiling = maxTrackbarValue
        initialFloor = 0
        # cv.createTrackbar("Magnitude Ceiling", windowName, initialCeiling, maxTrackbarValue, lambda dummyVariable : None)
        # cv.createTrackbar("Magnitude Floor", windowName, initialFloor, maxTrackbarValue, lambda dummyVariable : None)
 
        cv.setMouseCallback(windowName, fancyZoomMouseCallback, windowName)
 
        return (initialFloor / maxTrackbarValue), (initialCeiling / maxTrackbarValue)
    else:
        if (not(windowName in hasCallback.keys())):
            hasCallback[windowName] = False
        return relativeFloor, relativeCeiling
 
 
def getTrackbarAdjustedWeights(weights, windowName):
    minWeight, maxWeight = readBrightnessTrackbars(windowName)
    adjustedWeights = np.copy(weights)
    if (minWeight == maxWeight):
        # # if the min and max are the same,,set that value to be 0
        # # and all other values to be 1. This "negative highlighting"
        # # let's you see all the pixels with that magnitude in the context
        # # of all the other pixels, which is harder if I just kept the
        # # requested magnitude and zeroed everything else.
        # # This implementaiton also makes it easier to see all non-zero values
        # # which I do frequently.
        # adjustedWeights[adjustedWeights == maxWeight] = 0
        # adjustedWeights[adjustedWeights != maxWeight] = 1
 
        # actually, just round up everything that's above zero,
        # its easier to keep track of, and I never used the other option anyway.
        adjustedWeights[adjustedWeights > 0] = 1
    else:
        # normal rescaling
        # everything above max gets clipped to 1,
        # and everything below min gets clipped to 0.
        # everything in between gets scaled to fill the full [0, 1]
        # range of brightnesses
        # TODO: draw helpful ascii diagram in comments.
        # TODO: figure out if (max < min) is usefule somehow?
        adjustedWeights = (adjustedWeights - minWeight) / (maxWeight - minWeight)
        adjustedWeights[adjustedWeights > 1] = 1
        adjustedWeights[adjustedWeights < 0] = 0
 
    return adjustedWeights
 
 
def showVectorField(xComponents, yComponents,  windowName, externalWeights=None, precomputedAngles=None):
    """Displays a visualization of a given vector field, where
    color will indicate vector direction, and brightness will
    indicate vector magnitude.
 
    It's assumed that the vector field has been pre scaled,
    such that a magnitude of 0 will correspond to min brightess,
    and a magnitude of 1 or greater will correspond to max brightness.
    However, the funciton will add some trackbars that allow
    for some manual re-scaling of the magnitudes for
    visualization purposes.
    TODO: Clarify this, I feel like it's not super clear.
    """
 
    # Calculate the angle of each gradient vector, which will be
    # reflected in the color (hue) of each pixel in the output.
    # 
    # arctan2(y, x) gives angles in the range [-pi, pi]
    # For the purpose of visualization, we shift that into
    # the range [0, 2pi], then normalize to the range [0, 1]
    # (i.e. use units of rotations rather than radians)
    if (precomputedAngles is None):
        angles = np.arctan2(yComponents, xComponents)
    else:
        angles = np.copy(precomputedAngles)
    angles[angles < 0] += 2 * np.pi
    normalizedAngles = angles / (2 * np.pi)
 
    # Calculate the magnitude of each gradient vector, which will be
    # reflected in the brigthness (value) of each pixel in the ouput.
    # 
    # Note that we assume the vector magnitudes have already been
    # scaled to the range [0, 1] before being passed to this function,
    # but we also allow for re-normalization if desired.
    # This is mainly useful for switching back and forth between
    # viewing vector magnitudes relative to the maximum that could ever be achieved,
    # vs relative to the whatever the maximum happens to be on the current frame.
    # It also allows you to effectively specify what the maximum that could ever be achieved
    # actually is. For example, the max of my method is sqrt(0.5), but that's
    # not the same for Sobel or other derivative kernels (or even second/third/etc. derivatives with my method!)
    # It could even be other vector fields, like curvature or acceleration!
    # TODO: improve explanation?
    if (externalWeights is None):
        gradMags = np.sqrt(xComponents**2 + yComponents**2)
    else:
        gradMags = np.copy(externalWeights)
    gradMags = getTrackbarAdjustedWeights(gradMags, windowName)    
 
    # Construct the visualization in hsv colorspace,
    # then convert to bgr colorspace for easy visualization.
    noAngle = (xComponents == 0) & (yComponents == 0)
    h = (179 * normalizedAngles).astype(np.uint8)
    s = np.where(noAngle, 0, 255).astype(np.uint8)
    v = (255 * gradMags).astype(np.uint8)
    hsv = cv.merge([h, s, v])
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    showZoomedImage(windowName, bgr)
 
 
def annotateCorners(cornerScores, sourceImage, windowName, cornerColor=(0, 0, 255), dilateCorners=True):
    # For each pixel in the source image, we linearly interpolate
    # between its original BGR color and the given BGR cornerColor,
    # based on the strength of the corner (i.e. the cornerScore) at that pixel.
    # In other words, the new pixel is a weighted average of cornerColor
    # and the original pixel in the source image, where the weight
    # for cornerColor is given by the (normalized) corner score,
    # and the weight for the original pixel is (1-normalizedCornerScore).
    #
    # We also dilate the corner scores a bit just to make them
    # easier to see. That way you'll get a small cluster of points
    # around each corner instead of just a single pixel.
    # This is just to aid the visualization, this step wouldn't necessarily be
    # done on any further corner processing.
 
    # Normalize and dilate corner scores
    # (scores should already be globally normalized because maxCornerScore == 1?
    #  Will come back to this if I decide that typical corner scores are just too
    #  weak for global normalizaiton.)
    #
    # We don't need to renormalize after the dilate, because each pixel is just set
    # to the maximum of its neighbors.
 
    # Init the trackbar if it's not already on the window
    adjustedCornerScores = getTrackbarAdjustedWeights(cornerScores, windowName)
    if (dilateCorners):
        adjustedCornerScores = cv.dilate(adjustedCornerScores, None)
 
    # Reshape corner scores and color for proper broadcasting
    width = sourceImage.shape[1]
    height = sourceImage.shape[0]
    adjustedCornerScores = adjustedCornerScores.reshape((height, width, 1))
    broadcastableColor = np.array(cornerColor, dtype=np.float32) / 255
 
    # Perform the lerp, then display the image
    # annotatedCorners = simonLerp(adjustedCornerScores, broadcastableColor, sourceImage)
    annotatedCorners = (adjustedCornerScores * broadcastableColor) + (1-adjustedCornerScores) * sourceImage
    showZoomedImage(windowName, annotatedCorners)
 
    # alternative visualization idea that didn't work so well but that I might want to come back to:
    # use the negative of the original pixel as the corner color to guarantee contrast?
    #cornersOverlayRed = (normalizedCornerScores * (255 - originalImage[:,:,2])) + (1-normalizedCornerScores) * originalImage[:,:,2]
    #cornersOverlayGreen = (normalizedCornerScores * (255 - originalImage[:,:,1])) + (1-normalizedCornerScores) * originalImage[:,:,1]
    #cornersOverlayBlue = (normalizedCornerScores * (255 - originalImage[:,:,0])) + (1-normalizedCornerScores) * originalImage[:,:,0]
 
 
def showPositiveNegative(img, windowName):
    justNegatives = np.where(img < 0, np.abs(img), 0)
    justPositives = np.where(img > 0, img, 0)
 
    red = getTrackbarAdjustedWeights(justNegatives, windowName)
    green = np.zeros_like(img)
    blue = getTrackbarAdjustedWeights(justPositives, windowName)
    bgr = cv.merge([blue, green, red])
    showZoomedImage(windowName, bgr)
    # cv.imshow(windowName, bgr)
 
 
def displayStats(statStrings):
    width = 400
    spaceBetweenStrings = 20
    totalHeight = spaceBetweenStrings * (1 + len(statStrings))
    statsWindow = np.zeros((totalHeight, width, 3), dtype=np.uint8)
    # Note: full speed, just streaming with no processing is only 10 fps cause virtaul machine.
    #       Keep this in mind for benchmarks!
    # Followup: lol jk, frame rate was down becasue exposure was way up cause it was dark.
    #           actual fps is around 30 when there's sufficient lighting
 
    font = cv.FONT_HERSHEY_SIMPLEX
    color = (128, 128, 128)
    scale = 0.5
    xCordinate = 10
    yCordinate = spaceBetweenStrings
    for statString in statStrings:
        cv.putText(statsWindow, statString, (xCordinate, yCordinate), font, scale, color)
        yCordinate += spaceBetweenStrings
    cv.imshow("Stats", statsWindow)
 
    # print("width: ", camera.get(cv.CAP_PROP_FRAME_WIDTH))
    # print("height: ", camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    # print("frame rate: ", camera.get(cv.CAP_PROP_FPS))
    # print("backend api: ", camera.getBackendName())
 
    # which codec is used, not sure it really makes a difference?
    #
    # fourccInt = int(camera.get(cv.CAP_PROP_FOURCC))
    # fourccBytes = fourccInt.to_bytes(length=4, byteorder='little')
    # fourccString = fourccBytes.decode()
    # print("codec: ", fourccString)
 
    # documentation for "Mode" seems to indicate it should
    # specify the color space of returned frames
    # (e.g. BGR (default), RGB, Grey, YuYv)
    # and that it is supported by the V4L backend,
    # but checking this value at runtime returns 0,
    # which indicates it actually isn't supported.
    #
    # Format also seems similar, but honestly, I don't want to change these.
    # the default is fine.
    # print("mode: ", camera.get(cv.CAP_PROP_MODE))
    # print("format: ", camera.get(cv.CAP_PROP_FORMAT))
 
    # It's too much work to worry about really interpreting these.
    # I'll just deal with the popcorn myself
    # print("brightness:", camera.get(cv.CAP_PROP_BRIGHTNESS))
    # print("contrast:", camera.get(cv.CAP_PROP_CONTRAST))
    # print("saturation:", camera.get(cv.CAP_PROP_SATURATION))
    # print("hue:", camera.get(cv.CAP_PROP_HUE))
    # print("gain:", camera.get(cv.CAP_PROP_GAIN))
    # print("exposure:", camera.get(cv.CAP_PROP_EXPOSURE))
 
 
def displayHistogram(values, windowName, numBins=500):
    # Assume the values in the input have already been scaled to the range [0, 1]
    # (could be global or local min depending on user wants)
    # TODO: better documentation?
 
    # Init the trackbar if it's not already on the window
    xAxisUpperBound = cv.getTrackbarPos("frequencies to graph (as % of max magnitude)", windowName)
    yAxisUpperBound = cv.getTrackbarPos("height (as % of total num of datapoints)", windowName)
    maxBound = 1000
    if (xAxisUpperBound == -1 or yAxisUpperBound == -1):
        xAxisUpperBound = maxBound
        yAxisUpperBound = maxBound
        cv.createTrackbar("frequencies to graph (as % of max magnitude)", windowName, xAxisUpperBound, maxBound, lambda dummyVariable : None)
        cv.createTrackbar("height (as % of total num of datapoints)", windowName, yAxisUpperBound, maxBound, lambda dummyVariable : None)
 
    if (xAxisUpperBound < 1):
        xAxisUpperBound = 1
    if (yAxisUpperBound < 1):
        yAxisUpperBound = 1
 
    xAxisBounds = (0, xAxisUpperBound / maxBound)
    counts, binEdges = np.histogram(values, bins=numBins, range=xAxisBounds)
    maxValueToShow = values.size * (yAxisUpperBound / maxBound)
    normalizedCounts = counts / maxValueToShow # each count will be a percent of the max value to show
    normalizedCounts[normalizedCounts > 1] = 1
 
    plotWidth = numBins
    plotHeight = numBins
 
    # a single row indicating the height of each column (in pixels)
    heights = (normalizedCounts * plotHeight).astype(int)
 
    # a single column indicating the height of each row in pixels
    # (i.e. a y axis)
    yAxisValues = np.arange(plotHeight).reshape(plotHeight, 1)
    yAxisValues = np.flipud(yAxisValues) # put origin in bottom left of image like a normal graph.
    plot = (yAxisValues < heights).astype(np.float32)
 
    # The broadcasting stuff above is concice but hard to understand
    # Not super important for this function, but the below should
    # be equivalent and maybe easier to understand???
    # for i in range(plotWidth):
    #     col = np.zeros(plotHeight)
    #     col[0:heights[i]] = 1
    #     plot[:, i] = np.flip(col)
 
    cv.imshow(windowName, plot)
 
 
def showConnectedComponents(labeledImage, componentStats, windowName):
 
    # filter components based on area
    width = labeledImage.shape[1]
    height = labeledImage.shape[0]
    imgArea = width * height
    lowerBoundRatio, upperBoundRatio = readBrightnessTrackbars(windowName)
    minArea = 0#lowerBoundRatio * imgArea
    maxArea = 1000000#upperBoundRatio * imgArea
 
    areas = componentStats[:, cv.CC_STAT_AREA]
    componentsToShowList = np.argwhere((minArea <= areas) & (areas <= maxArea)).flatten()
    componentsToShow = np.where(np.isin(labeledImage, componentsToShowList), labeledImage, 0)
 
    # now color the components
    # Stolen from stack overflow 46441893 (thx dude, good idea)
    goldenRatio = (1 + np.sqrt(5))/2
    normalizedHues = (goldenRatio * componentsToShow.astype(np.float32)) % 1.0
    h = (179 * normalizedHues).astype(np.uint8)
    s = np.full_like(h, 255)
    v = np.where(componentsToShow == 0, 0, 255).astype(np.uint8)
    hsv = cv.merge([h, s, v])
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    showZoomedImage(windowName, bgr)
    # cv.imshow(windowTitle, bgr)
 
    # let the user click on a component to highlight
    global mouseClickY,mouseClickX
    return mouseClickX, mouseClickY, componentsToShowList
 
 
def drawCenterOfCurvature(cordX, cordY, radiusX, radiusY, imageToDrawOn, windowName):
 
    minRadiusToDraw = cv.getTrackbarPos("minRadius", windowName)
    radiusRange = cv.getTrackbarPos("radiusRange", windowName)
    if (minRadiusToDraw < 0):
        initialMinRadius = 1000
        initialRadiusRange = 1
        maxTrackbarValue = 1000
        cv.createTrackbar("minRadius", windowName, initialMinRadius, maxTrackbarValue, lambda dummyVariable : None)
        cv.createTrackbar("radiusRange", windowName, initialRadiusRange, maxTrackbarValue, lambda dummyVariable : None)
 
        cv.setMouseCallback(windowName, fancyZoomMouseCallback, windowName)
 
        minRadiusToDraw = initialMinRadius
        radiusRange = initialRadiusRange
 
    # unpack everything into python lists so I can loop
    radiusSize = np.sqrt(radiusX**2 + radiusY**2)
    radiusInRange = (minRadiusToDraw <= radiusSize) & (radiusSize <= (minRadiusToDraw + radiusRange))
    xList = cordX[radiusInRange].flatten().tolist()
    yList = cordY[radiusInRange].flatten().tolist()
    radiusXList = radiusX[radiusInRange].flatten().tolist()
    radiusYList = radiusY[radiusInRange].flatten().tolist()
    canvas = (255 * imageToDrawOn).astype(np.uint8) # i think drawing only works on uint8 images.
 
    for i in range(len(xList)):
        x = xList[i]
        y = yList[i]
        rx = radiusXList[i]
        ry = radiusYList[i]
        cx = int(x + rx)
        cy = int(y + ry)
        cv.arrowedLine(canvas, (x,y), (cx, cy), color=(0,0,255))
 
    showZoomedImage(windowName, canvas)
 
 
def showHeatMap(values, windowName):
    adjustedValues = getTrackbarAdjustedWeights(values, windowName)
    # cv.imshow("Adjused Values", adjustedValues)
 
    # h = (179 * normalizedAngles).astype(np.uint8)
    h = (60 * (1-adjustedValues)).astype(np.uint8) # colors go from mid red to mid green
    s = np.where(adjustedValues == 1, 0, 255).astype(np.uint8) #above thresh gets white, below gets black.
    v = np.where(adjustedValues == 0, 0, 255).astype(np.uint8)
    hsv = cv.merge([h, s, v])
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    showZoomedImage(windowName, bgr)
 
 
def getGaussianBlob(shape, row, col, height=1, std=0):
    blankImg = np.zeros(shape, dtype=np.float32)
    blankImg[row, col] = height
    if (std == 0):
        autoStd = np.sqrt(2 * np.pi / height)
        return cv.GaussianBlur(blankImg, ksize=(1281,1281), sigmaX=autoStd, borderType=cv.BORDER_REPLICATE)
    else:
        return cv.GaussianBlur(blankImg, ksize=(1281,1281), sigmaX=std, borderType=cv.BORDER_ISOLATED)
 
 
def curvatureComparison():
    pass
    # # Curvature from Velcity and Accel Experiments
    # velocityX = -gradY
    # velocityY =  gradX
    # speed = gradMags
    # oneOverSpeed = np.where(speed > 0, 1/speed, 0)
    # vHatX = velocityX * oneOverSpeed
    # vHatY = velocityY * oneOverSpeed
    # tangentX = vHatX
    # tangentY = vHatY
 
    # # First attempt. The idea is that my function yeilds the derivative w.r.t. time of 
    # # whatever I throw in there, and that dT/ds = dT/dt * dt/ds = (1/speed) * dT/dt.
    # #
    # # Where I went wrong here is in understanding what my function computes. It doesn't
    # # actually take the derivative of whatever I throw in there w.r.t. time.
    # # It first interprets the thing I throw in as a velocity vector field, and then finds
    # # the acceleration of that velocity vector field. If I was just taking the derivative
    # # of whatever was thrown in w.r.t. time, and the thing being thrown in wasn't a velocity
    # # vector field, I would have no way to compute dx/dt or dy/dt in the chain rule part. 
    # #
    # # So when I throw in the tangent vectors, it doesn't see them as unitless tangent vectors.
    # # it sees it as a velocity field where the magnitude of the velocity everywhere just
    # # happens to be 1. I have to forget the original velocity field, and think of these
    # # tangnets as being their own independent velocity field. In that case,
    # # 1/speed becomes 1 everywhere, and pluggin that into our formula for dT/ds,
    # # we get dT/ds = (1/speed) * dT/dt -> dT/ds = dT/dt (with an invisible factor of 1 taking care of the units)
    # # Therefore, the curvature is just numerically equivalent to dT/dt in the case where all speeds are 1.
    # # But furthermore, when all speeds are 1, we have T = V * (1/speed) -> T = V (with another invisible 1 for units).
    # # As a result, when the magnitude of all your velocies is 1 (as the getVectorFieldAccel() function believes),
    # # then computing dT/ds is numerically equivalent to computing dV/dt = acceleration!
    # #
    # # Alternatively, beccause of my insight I now have, I can say that because
    # # v^2 / R = A_perp, that 1 / R = A_perp / v^2. However, when v is always 1,
    # # all acceleration must be perpendicular! Therfore, in that case, we have:
    # # 1 / R = curvature = A_perp (where the v^2 became an invisible division by 1 which handles the units).
    # #
    # # TODO: Review original notes on this confusion, because they may still have an unanswered question:
    # # If something that's actually a normalized tangent is re-interpreted as a velcity vector,
    # # then computing it's acceleration is computationally identical to computing the curvature vector
    # # because when you go to make the units match by dividing by the speed, you just divide by 1 because
    # # that was your velocity? you devide by the velocity of the field, which in the second case has to be 1
    # # function doesn't tell the diff between velocity and tangent.
    # #
    # # why does this trick not work in real other situations though? If this was always the case
    # # then you just first find T(t), then reinterpret that as being velocity, then find accel
    # # without dividing out by the speed????
    # alreadyCurvatureX, alreadyCurvatureY = imageDerivatives.getVectorFieldAcceleration(tangentX, tangentY)
    # tooMuchCurvatureX = alreadyCurvatureX * oneOverSpeed
    # tooMuchCurvatureY = alreadyCurvatureY * oneOverSpeed
    # tooMuchCurvature = np.sqrt(tooMuchCurvatureX**2 + tooMuchCurvatureY**2)
    # cv.imshow("Too Much Curvature (Previously Confused Curvature)", tooMuchCurvature / np.max(tooMuchCurvature))
    # wrongUnitsButRightValueCurvature = np.sqrt(alreadyCurvatureX**2 + alreadyCurvatureY**2)
    # wubrvc = wrongUnitsButRightValueCurvature
    # cv.imshow("Wrong Units But Right Value Curvature", wubrvc / np.max(wubrvc))
 
    # # Now, to avoid having to use all those comments to explain the improper units ordeal
    # # see if I can come up with a relatively simple expression that is dimensionally consistent.
    # accelX, accelY = imageDerivatives.getVectorFieldAcceleration(velocityX, velocityY)
    # accelMag = np.sqrt(accelX**2 + accelY**2)
    # cv.imshow("Acceleration (Corners)", accelMag / np.max(accelMag))
 
    # # formula 1 curvature
    # vDotA = velocityX * accelX + velocityY * accelY
    # gotItCurvatureX = (velocityX * -vDotA) * oneOverSpeed**4 + accelX * oneOverSpeed**2
    # gotItCurvatureY = (velocityY * -vDotA) * oneOverSpeed**4 + accelY * oneOverSpeed**2
    # gotItCurvature = np.sqrt(gotItCurvatureX**2 + gotItCurvatureY**2)
    # cv.imshow("Formula 1 Curvature (No Insight)", gotItCurvature / np.max(gotItCurvature))
    # cv.imshow("Rescaled Formula 1", gotItCurvature / np.sqrt(2))
 
    # # formula 2 curvature
    # aDotVHat = accelX * vHatX + accelY * vHatY
    # aPerpX = accelX - vHatX * aDotVHat
    # aPerpY = accelY - vHatY * aDotVHat
    # aPerp = np.sqrt(aPerpX**2 + aPerpY**2)
    # perpCurvature = aPerp * oneOverSpeed**2
    # cv.imshow("Formula 2 Curvature (MY INSIGHT!)", perpCurvature / np.max(perpCurvature))
    # cv.imshow("Rescaled Formula 2", perpCurvature / np.sqrt(2))
 
    # # formula 3 curvature
    # crossMag = np.abs(accelX * velocityY - velocityX * accelY)
    # crossCurvature = crossMag * oneOverSpeed**3
    # cv.imshow("Formula 3 Curvature (Cross Product)", crossCurvature / np.max(crossCurvature))
    # cv.imshow("Rescaled Formula 3", crossCurvature / np.sqrt(2))
 
    # # see if it's just floating point weirdness
    # fixedPerpCurvature = np.where(perpCurvature <= np.sqrt(2), perpCurvature, 0)
    # cv.imshow("Outliers Removed Formula 2", fixedPerpCurvature / np.max(fixedPerpCurvature))
 
    # histogramMax = np.sqrt(2) #np.max(perpCurvature) # np.max(wubrvc) # np.max(perpCurvature) # np.sqrt(2)
    # visualizationUtils.displayHistogram(wubrvc / histogramMax, "Correct Histogram")
    # visualizationUtils.displayHistogram(perpCurvature / histogramMax, "Perp Histogram")
 
    # curvatureDiff = np.abs(perpCurvature - wubrvc)
    # cv.imshow("Curvature Diff", curvatureDiff / np.max(curvatureDiff))
    # visualizationUtils.displayHistogram(curvatureDiff / np.max(curvatureDiff), "Curvature Diff Histogram")
 
    # clipedCurvature = np.where(perpCurvature > 8, 0, perpCurvature)
    # cv.imshow("Clipped and Cleaned Curvature", clipedCurvature / np.max(clipedCurvature))
 
    # print("correctMax:", np.max(wubrvc))
    # print("max1:", np.max(gotItCurvature))
    # print("max2:", np.max(perpCurvature))
    # print("max3:", np.max(crossCurvature))
    # print("tooMuchMax:", np.max(tooMuchCurvature))
    # print("fixedMax:", np.max(fixedPerpCurvature))
    # print("type:", crossMag.dtype)
    # print("numOver:", (perpCurvature > np.sqrt(2)).astype(int).sum())
 
 
def drawHoughSpaceMap(radiusBinMidpoints, thetaBinMidpoints, sourceImage):
    # draw a template
    # print("radii:", radiusBinMidpoints)
    # print("theta:", thetaBinMidpoints*180/np.pi)
    height = sourceImage.shape[0]
    width = sourceImage.shape[1]
    template = np.zeros((height, width, 3))
    imageCenterX = width/2
    imageCenterY = height/2
    for radius in radiusBinMidpoints:
        radiusSpacing = 2
        if (radius >= 0):
            radius = int(radius+radiusSpacing)
            circleColor = (255,0,0)
            cv.circle(template, (int(imageCenterX),int(imageCenterY)), radius, color=circleColor)
        else:
            radius = int(abs(radius))
            if (radius > radiusSpacing):
                radius -= radiusSpacing
            circleColor = (0,0,255)
            cv.circle(template, (int(imageCenterX),int(imageCenterY)), radius, color=circleColor)
        for theta in thetaBinMidpoints:
 
            # Direction along line
            perpX, perpY = np.cos(theta), np.sin(theta)
            alongX, alongY = -perpY, perpX
            lineLength = 50
            t = lineLength/2
            x0, y0 = imageCenterX + (radius * perpX), imageCenterY + (radius * perpY)
            x1, y1 = x0 + t * alongX, y0 + t * alongY
            x2, y2 = x0 - t * alongX, y0 - t * alongY
 
            cv.line(template, (int(x1),int(y1)), (int(x2),int(y2)), color=(0,255,0))
 
    cv.imshow("houghSpaceDiscretization", template)
 
 
def drawHoughLines(strongLines, sourceImage, windowName, splitLinePoints=None):
    # unpack some data
    canvas = sourceImage.copy()
    radii = strongLines[:,0]
    angles = strongLines[:,1]
    scores = strongLines[:,2]
    lowerBoundRatio, upperBoundRatio = readBrightnessTrackbars(windowName)
 
    # just in case its not sorted cause I changed something while testing.
    maxScore = np.max(scores)
    minScoreToDraw = lowerBoundRatio * maxScore
    maxScoreToDraw = upperBoundRatio * maxScore
 
    drawMeMask = (minScoreToDraw <= scores) & (scores <= maxScoreToDraw) & (scores > 0)
    drawMeIndices = np.arange(len(scores))[drawMeMask]
 
    # do the drawing
    width = canvas.shape[1]
    height = canvas.shape[0]
    originX = 0#(width-1)/2
    originY = 0#(height-1)/2
 
    for i in drawMeIndices:
        theta = angles[i]
        radius = radii[i]
        # Direction along line
        perpX, perpY = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        alongX, alongY = -perpY, perpX
        lineLength = 3000
        t = lineLength/2
        x0, y0 = originX + (radius * perpX), originY + (radius * perpY)
        x1, y1 = x0 + t * alongX, y0 + t * alongY
        x2, y2 = x0 - t * alongX, y0 - t * alongY
 
        if (radius >= 0):
            lineColor = (255, 0, 0)
        else:
            lineColor = (0, 0, 255)
 
        cv.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), color=lineColor)
 
    if (not(splitLinePoints is None)):
        x1, y1 = splitLinePoints[0]
        x2, y2 = splitLinePoints[1]
        cv.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), color=(0,0,0))
 
    showZoomedImage(windowName, canvas)
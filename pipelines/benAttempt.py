import cv2
import numpy as np
import math
import visutils
import multiprocessing as mp
import itertools


kImgShape = (720, 1280)
imgGrey = np.zeros(kImgShape, dtype=np.float32)
canny = np.zeros(kImgShape, dtype=np.uint8)
magnitude = np.zeros(kImgShape, dtype=np.float32)
angle = np.zeros(kImgShape, dtype=np.float32)
gradientX = np.zeros(kImgShape, dtype=np.float32)
gradientY = np.zeros(kImgShape, dtype=np.float32)
inverseMagnitudes = np.zeros_like(magnitude)
labels = np.zeros_like(magnitude, dtype=np.int32)
xCoords, yCoords = np.indices(kImgShape, dtype=np.float32)
xCoords-=int((kImgShape[1]-1)/2)
yCoords-=int((kImgShape[0]-1)/2)
r = np.zeros(kImgShape, dtype=np.float32)

#applies scharr filter, returns arrays of magnitude and angle
#magnitude is normalized to 1, angle is normalized from -180 to 180
@profile
def findEdges(imgGrey):
    #optimal 3x3 kernel for scharr filter according to wikipedia
    scharrXKernel = np.array(
        [[-47, 0, 47],
        [-162, 0, 162],
        [-47, 0, 47]])
    
    scharrYKernel = np.array(
        [[-47, -162, -47],
        [0, 0, 0],
        [47, 162, 47]])
    
    global gradientX, gradientY, magnitude, angle, inverseMagnitudes
    gradientX = cv2.filter2D(imgGrey, -1, scharrXKernel, gradientX, borderType=cv2.BORDER_REPLICATE)
    gradientY = cv2.filter2D(imgGrey, -1, scharrYKernel, gradientY, borderType=cv2.BORDER_REPLICATE)

    magnitude, angle = cv2.cartToPolar(gradientX, gradientY, magnitude, angle, angleInDegrees=True)
    angle -= 180
    
    inverseMagnitudes = np.divide(1, magnitude, out=inverseMagnitudes, where=(magnitude > 0))
    
    gradientX *= inverseMagnitudes
    gradientY *= inverseMagnitudes

    return magnitude, angle, gradientX, gradientY

@profile
def componentHoughTransform(angles, gradientX, gradientY, canny):
    
    global labels, xCoords, yCoords, r
    nComponents, labels, stats, centroids = cv2.connectedComponentsWithStats(canny, labels=labels, connectivity=8)
    
    
    #gradientX is equivalent to cosine, likewise for sine
    r = np.multiply(xCoords, gradientX, r,)
    r += np.multiply(yCoords, gradientY) 
    
    areas = stats[:,cv2.CC_STAT_AREA]
    widths = stats[:,cv2.CC_STAT_WIDTH]
    heights = stats[:,cv2.CC_STAT_HEIGHT]
    
    #limits for valid components:
    #can't be too big or too small
    #can't be too wide/tall
    #area should be a lil less than ~4x width and height (there are 4 sides)
    validComponents = np.where((areas > 100) & (areas < 700) & 
                               ((widths/heights) < 1.3) & (heights/widths < 1.5) &
                               (areas/widths < 4.5) & (areas/widths > 3.) &
                               (areas/heights < 4.5) & (areas/heights > 3.))[0]
    
    

     
    labelsToShow = np.where(np.isin(labels, validComponents), labels, 0)
    clickX, clickY, _ = visutils.showConnectedComponents(labelsToShow, stats, "connectedComponents")
    # clickedComponent = labels[clickY, clickX]
    # print(clickedComponent)
    # validComponents = [clickedComponent]
    
    
    args=[(i, r, angles, labels) for i in validComponents]
    
    
    #multiprocessing
    if False:
        with mp.Pool() as p:
            results=p.map(houghSpacePerComponent, args)
            
    results = [houghSpacePerComponent(arg) for arg in args]
    
    return results


@profile
def houghSpacePerComponent(args):
    i, r, angles, labels = args
    
    
    kImgShape=angles.shape
    
    kMaxRPixels = int(math.sqrt(kImgShape[0]**2 + kImgShape[1]**2))
    rbins = int(kMaxRPixels/2) #should be divided by 2 for 1-pixel accuracy
    tbins = 1600
    
    print(np.max(r), np.max(angles))
    
    houghSpace = cv2.calcHist(
        [r, angles],
        [0, 1],
        (labels==i).astype(np.uint8, copy=False),
        [rbins, tbins], 
        [-kMaxRPixels,kMaxRPixels,-180,180]
    )

    
    
    rHistEdges = np.linspace(-kMaxRPixels, kMaxRPixels, rbins, True)
    tHistEdges = np.linspace(-180, 180, tbins, True)
    
    
    # houghSpace, rHistEdges, tHistEdges = np.histogram2d(
    #     r.flatten(), 
    #     angles.flatten(), 
    #     bins=[rbins,tbins],
    #     range=[[-kMaxRPixels, kMaxRPixels],[-180, 180]], 
    #     #weights=cMagnitudes.flatten()
    # )
    
    # houghSpace = cv2.GaussianBlur(houghSpace, (0, 0), 8)
    #visutils.showHeatMap(houghSpace/np.max(houghSpace), "houghSpace")
    
    
    
    #grabs only the nonzero elements of houghSpace, then argpartitions
    #topInds are the indices into houghSpace[houghSpace>0]
    #for example, if a value in topInds == 0, that refers to the first nonzero element in the hough space
    
    #print(houghSpace[houghSpace>0].shape[0])
    if houghSpace[houghSpace>0].shape[0] <= 30:
        return []
        
    
    topInds = np.argpartition(houghSpace[houghSpace>0], -30, axis=None)[-30:]

    
    #plug in topInds into indices of nonzero elements, going to coordinates in the hough space
    topRows, topCols = np.indices(houghSpace.shape)
    topRows = topRows[houghSpace > 0]
    topCols = topCols[houghSpace > 0]
    topRows = topRows[topInds]
    topCols = topCols[topInds]
    
    topR = np.flip(rHistEdges[topRows]).tolist()
    topT = np.flip(tHistEdges[topCols]).tolist()
    topVal = np.flip(houghSpace[topRows, topCols]).tolist()
    
    topLines = sorted(list(zip(topR, topT, topVal)), key=lambda line: line[2], reverse=True)
    
    #return topLines
    return cullDuplicateLines(topLines, 4, 100, 20)


def cullDuplicateLines(topLines, targetNumLines, rThreshold, tThreshold):
    
    strongLines = [topLines[0]]
    
    for line in topLines:
        
        if len(strongLines) >= targetNumLines:
            break
        
        isStrongLine = True
        
        for strongLine in strongLines:
            
    
            deltaR = abs(line[0] - strongLine[0])
            deltaT = abs(line[1] - strongLine[1])
            
            # if deltaT < tThreshold:
                
            #     a=math.cos(math.radians(line[1]))
            #     b=math.sin(math.radians(line[1]))
                
            #     d=math.cos(math.radians(strongLine[1]))
            #     e=math.sin(math.radians(strongLine[1]))
            
            #     c=line[0]
            #     f=strongLine[0]
            
            #     #lines are parallel
            #     if abs(a*e-b*d) < 0.01:
            #         #TODO: check r
            #         if deltaR < rThreshold:
            #             isStrongLine = False
            #             break
            #         continue
                
            #     intersectX = (c*e-b*f)/(a*e-b*d)
            #     intersectY = (a*f-c*d)/(a*e-b*d)
                
            #     if (intersectX < imgGrey.shape[1] and intersectX > 0 and intersectY < imgGrey.shape[0] and intersectY > 0):
            #         #intersection is on the screen and thetas are similar, so eliminate this line
            #         isStrongLine = False
            #         break

            if (deltaR < rThreshold) and (deltaT < tThreshold):
                #this line matches with a previous line
                if line[2] > strongLine[2]:
                    #if this line is stronger than the one in the list, replace it
                    strongLines.remove(strongLine) #technically remove is slow, but strongLines is max length 4 anyways so
                else:
                    #otherwise, break out of the loop and don't add the line
                    isStrongLine = False
                break
            
        if isStrongLine:
            strongLines.append(line)
    
    return strongLines

def findIntersectionsPerComponent(houghLines, kImgShape):
    lineIntersections = []
    for line1 in houghLines:
        for line2 in houghLines:
            a=math.cos(math.radians(line1[1]))
            b=math.sin(math.radians(line1[1]))
            
            d=math.cos(math.radians(line2[1]))
            e=math.sin(math.radians(line2[1]))
            
            c=line1[0]
            f=line2[0]
            
            if abs(a*e-b*d) == 0.00:
                continue

            intersectX = (c*e-b*f)/(a*e-b*d)
            intersectY = (a*f-c*d)/(a*e-b*d)
            
            #only checks for intersections on the screen
            if intersectX > kImgShape[1] or intersectX < -1 or intersectY > kImgShape[0] or intersectY < -1:
                continue
            
            lineIntersections.append((intersectX, intersectY))

def findIntersectionsOld(strongLines, kImgShape):
    #list of lists of intersections, each sublist represents one line
    lineIntersections = []


    #generates lineIntersections
    cnt = -1
    for line1 in strongLines:
        cnt += 1
        lineIntersections.append([])
        for line2 in strongLines:
            #to find intersection, solve system ax+by=c, dx+ey=f given 2 lines
            #a=cos(t1), b=sin(t1), d=cos(t2), e=sin(t2), c=r1, f=r2
            
            a=math.cos(math.radians(line1[1]))
            b=math.sin(math.radians(line1[1]))
            
            d=math.cos(math.radians(line2[1]))
            e=math.sin(math.radians(line2[1]))
            
            c=line1[0]
            f=line2[0]
            
            #x=(ce-bf)/(ae-bd), y=(af-cd)/(ae-bd)
            
            if abs(a*e-b*d) == 0.00:
                continue

            intersectX = (c*e-b*f)/(a*e-b*d)
            intersectY = (a*f-c*d)/(a*e-b*d)
            
            #only checks for intersections on the screen
            if intersectX > kImgShape[1] or intersectX < -1 or intersectY > kImgShape[0] or intersectY < -1:
                continue
            
            lineIntersections[cnt].append((intersectX, intersectY))
    
    return lineIntersections

def generateLineSegments(lineIntersections, magnitudes):
    blurredMagnitudes = cv2.GaussianBlur(magnitudes, (3, 3), 0.)
    
    lineSegments = []
    
    #scans along a line, checking at midpoints between intersections to generate line segments
    #ensures that two mini-line segments next to each other will be registered as a single line segment
    for line in lineIntersections:
        line.sort()
        
        if len(line) < 2:
            continue
        
        startX = None
        startY = None
        for i in range(len(line)-1):
            x1 = line[i][0]
            y1 = line[i][1]
            x2 = line[i+1][0]
            y2 = line[i+1][1]
            
            xMid = (x1+x2)/2
            yMid = (y1+y2)/2
        
            
            if (blurredMagnitudes[int(yMid)][int(xMid)] > 0.005):
                #current mini-segment is valid
                
                if startX == None and startY == None:
                    #if the previous segment was invalid, start a new segment
                    startX = x1
                    startY = y1
                
                if i == len(line)-2:
                    #if reached the last mini-segment, end the line segment here
                    lineSegments.append((startX, startY, x2, y2))
                    
            else:
                #current mini-segment is invalid
                
                if startX != None and startY != None:
                    #if there was an ongoing segment, end it here
                    lineSegments.append((startX, startY, x1, y1))
                    
                startX = None
                startY = None
                
    return lineSegments

def runPipeline(image, llrobot):
    global imgGrey, canny
    
    imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, imgGrey).astype(np.float32)/255
    kImgShape = imgGrey.shape
    
    magnitudes, angles, gradientX, gradientY = findEdges(imgGrey)

    #uses canny filter to get rid of extra points
    #old constants 110, 225, aperturesize=5 idk why it changed
    cv2.Canny(image, 27000, 42000, canny, 7, L2gradient=True)
    

    strongLines = list(itertools.chain.from_iterable(
                    componentHoughTransform(angles, gradientX, gradientY, canny)))

    #TODO: iterate through components, find intersections, and then pnp and then you are done
    
    if (len(strongLines) > 0):
        visutils.drawHoughLines(np.array(strongLines), image, "houghLines")

    #OUTPUTS AND VISUALIZATION

    """
    #normalize magnitude and angle into units that cv2 likes
    magnitudes = np.uint8(255*magnitudes)
    angles = np.uint8(angles/2 + 90)
    hsvImage = cv2.merge([angles, np.full(imgGrey.shape, 255, dtype=np.uint8), magnitudes])
    rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
    
    image[canny > 0] = (0, 0, 255)
    
    for coord in strongLines:

        maxR = coord[0]
        maxTheta = coord[1]

        #print(maxR, maxTheta)

        a = np.cos(math.radians(maxTheta))
        b = np.sin(math.radians(maxTheta))
        x0 = a*maxR
        y0 = b*maxR
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))



        print(maxR, maxTheta)
        if maxR > 0:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    
    for segment in lineSegments:
        
        x1, y1, x2, y2 = segment
        
        #cv2.line(image, np.int16((x1, y1)), np.int16((x2, y2)), (255, 255, 0), 2)
    
        
    for line in lineIntersections:
        for point in line:
            pass
            #cv2.circle(image, np.int16(point), radius=5, color = (0,255,0), thickness=-1)

    

    houghSpace = np.uint8(255*houghSpace)
    houghSpace = cv2.cvtColor(houghSpace, cv2.COLOR_GRAY2RGB)
    #houghSpace = cv2.circle(houghSpace, houghMaxCoords, radius=2, color=(0, 255, 0), thickness=-1)

    """
    
    
    return np.array([[]]), image, llrobot

    
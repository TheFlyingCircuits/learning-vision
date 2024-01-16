import cv2
import numpy as np
import math


#applies scharr filter, returns arrays of magnitude and angle
#magnitude is normalized to 1, angle is normalized from -180 to 180
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

    gradientX = cv2.filter2D(imgGrey, -1, scharrXKernel, borderType=cv2.BORDER_REPLICATE)
    gradientY = cv2.filter2D(imgGrey, -1, scharrYKernel, borderType=cv2.BORDER_REPLICATE)

    magnitude = np.sqrt(gradientX*gradientX+gradientY*gradientY)

    maxMagnitude = magnitude.max()#math.sqrt(2*256*256)
    magnitude /= maxMagnitude

    angle = np.arctan2(gradientY, gradientX)*180/math.pi

    return magnitude, angle




def runPipeline(image, llrobot):
    


    imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #imgGrey = cv2.GaussianBlur(imgGrey, (7, 7), 0)

    #imgGrey = cv2.adaptiveThreshold(imgGrey, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    #imgGrey = cv2.erode(imgGrey, np.ones((3, 3), np.uint8))
    
    canny = cv2.Canny(image, 110, 225, 5, L2gradient=True)

    imgGrey = np.float32(imgGrey/255)

    magnitudes, angles = findEdges(imgGrey)

    #uses canny filter to get rid of extra points
    magnitudes[canny==0] = 0



    indices = np.indices(imgGrey.shape)
    xCoords = indices[1]
    yCoords = indices[0]

    r = xCoords*np.cos(np.radians(angles)) + yCoords*np.sin(np.radians(angles))

    #you can pull this out of runPipeline to save a lil time?
    kMaxRPixels = int(math.sqrt(imgGrey.shape[0]**2 + imgGrey.shape[1]**2))
    rbins = kMaxRPixels
    tbins = 3600
    
    
    # rHistEdges = np.histogram_bin_edges(r.flatten(), bins='sqrt')
    # tHistEdges = np.histogram_bin_edges(angles.flatten(), bins='sqrt')
    
    houghSpace, rHistEdges, tHistEdges = np.histogram2d(
        r.flatten(), 
        angles.flatten(), 
        bins=[rbins,tbins],
        range=[[-kMaxRPixels, kMaxRPixels],[-180, 180]], 
        weights=magnitudes.flatten()
    )

    #non-max suppression: gets rid of all points that aren't maximums by using cv2.dilate()
    #houghSpace[cv2.dilate(houghSpace, np.ones((int(rbins/8), int(tbins/70)), np.float32), borderType=cv2.BORDER_WRAP)-houghSpace!=0] = 0
    
    #get arrays of indices of maximum points in hough space
    sortedRows, sortedCols = np.unravel_index(np.argsort(houghSpace, axis=None), houghSpace.shape)
    sortedVals = np.sort(houghSpace, axis=None)/np.max(houghSpace)

    #convert from histogram indices into actual r and theta
    sortedRows = rHistEdges[sortedRows]
    sortedCols = tHistEdges[sortedCols]

    #convert two separate x and y arrays into array of individual coordinates as tuples
    topRows = np.flip(sortedRows).tolist()
    topCols = np.flip(sortedCols).tolist()
    topVals = np.flip(sortedVals).tolist()
    topLines = list(zip(topRows, topCols, topVals))
    
    #strongLines = cv2.HoughLines(canny, 1, math.pi/180, 10)
    strongLines = topLines
    
    
    strongLines = [topLines[0]]
    
    rThreshold = 50
    tThreshold = 8
    
    for line in topLines:
        
        if len(strongLines) >= 50:
            break
        
        if line[2] > 3:
            continue
        
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

            if deltaR < rThreshold and deltaT < tThreshold:
                isStrongLine = False
                break
            
        if isStrongLine:
            strongLines.append(line)    
                
    
    
    
    

    #list of lists of intersections, each sublist represents one line
    lineIntersections = []

    #add border lines so that line intersection also includes intersections with border
    # strongLines.append((0, 0))
    # strongLines.append((0, 90))
    # strongLines.append((540, 0))
    # strongLines.append((200, 90))


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
            if intersectX > imgGrey.shape[1] or intersectX < -1 or intersectY > imgGrey.shape[0] or intersectY < -1:
                continue
            
            lineIntersections[cnt].append((intersectX, intersectY))

    
    
    
    
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
                    lineSegments.append((startX, startY, x2, y2))
                    
            else:
                #current mini-segment is invalid
                
                if startX != None and startY != None:
                    #if there was an ongoing segment, end it here
                    lineSegments.append((startX, startY, x1, y1))
                    
                startX = None
                startY = None
        
        

    



    #normalize magnitude and angle into units that cv2 likes
    magnitudes = np.uint8(255*magnitudes)
    angles = angles/2 + 90
    angles = np.uint8(angles)
    saturation = np.full(imgGrey.shape, 255, dtype=np.uint8)
    hsvImage = cv2.merge([angles, saturation, magnitudes])
    rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

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



        #print(maxR, maxTheta)
        if maxR > 0:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255*coord[2]), 2)
        else:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255*coord[2], 0), 2)
        
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

    
    image[canny > 0] = (0, 0, 255)
    return np.array([[]]), image, llrobot

    
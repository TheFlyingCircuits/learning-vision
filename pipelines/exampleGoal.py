import cv2
import numpy as np

# runPipeline() is called every frame by Limelight's backend.


def runPipeline(image, llrobot):

    # image is input as a uint8, we want floating point for processing
    normalizedGreyscaleImage = (1.0 / 255.0) * \
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradientX, gradientY = calculateGradient(normalizedGreyscaleImage)
    mag = (gradientX**2 + gradientY**2)**(0.5)

    maxGrad = (1.0 - 0) / 2.0
    # np.max(mag) #(maxGrad**2 + maxGrad**2)**(0.5) # theoretical max is sqrt(0.5) ~= 0.7
    maxMag = 0.5**0.5 * 0.25
    # print("max:", maxMag)
    normalizedMag = mag / maxMag
    normalizedMag[normalizedMag > 1] = 1
    # avgMag = np.mean(normalizedMag) * 4
    # normalizedMag[normalizedMag >= avgMag] = 1
    # normalizedMag[normalizedMag < avgMag] = 0

    scaledMag = normalizedMag * 255 * 2
    intMag = scaledMag.astype(np.uint8)

    angles = np.arctan2(gradientY, gradientX)
    normalizedAngles = (angles - np.pi) / (2.0 * np.pi)
    scaledAngles = normalizedAngles * 179
    intAngles = scaledAngles.astype(np.uint8)

    saturation = np.ones_like(intAngles)
    saturation *= 255
    outputImage = cv2.merge((intAngles, saturation, intMag))

    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_HSV2BGR)

    # print("yo")
    # print(normalizedGreyscaleImage.dtype)
    # print(image.shape)
    # print("hello1")

    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    largestContour = np.array([[]])  # can't just put null here
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
    # outputImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cv2.cvtColor(normalizedGreyscaleImage, cv2.COLOR_GRAY2BGR)
    # print("hello!")
    # print(outputImage.dtype)
    # ouptut image must be a BRGu8 type
    greyDisplay = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    greyDisplay = cv2.merge((greyDisplay, greyDisplay, greyDisplay))
    return largestContour, outputImage, llpython


def calculateGradient(image):
    """
    calculates the gradient of an image
    assumed input type is a single channel
    floating point image (i.e. greyscale)
    """
    width = image.shape[1]
    height = image.shape[0]

    # init output
    # gradient = np.zeros((height, width, 2))

    # easy to understand, but way too slow,
    # the limelight grinds to a halt
    # for y in range(height):
    #     print("row:", y)
    #     for x in range(width):

    #         if (y < 1 or y > height-2):
    #             # no neighbors above and below
    #             continue
    #         if (x < 1 or x > width-2):
    #             # no neighbors left or right
    #             continue

    #         # calculate df/dx by taking average
    #         # between neighbors
    #         stepSize = 1.0
    #         slopeLeft = (image[y, x] - image[y, x-1]) / stepSize
    #         slopeRight = (image[y, x+1] - image[y, x]) / stepSize
    #         avgSlopeX = (slopeLeft + slopeRight) / 2.0
    #         gradient[y, x, 0] = avgSlopeX # if x is 2nd index, shoult this be at channel 1 instead of 0?

    #         # calculate df/dy by taking average
    #         # between neighbors
    #         slopeAbove = (image[y, x] - image[y-1, x]) / stepSize
    #         slopeBelow = (image[y+1, x] - image[y, x]) / stepSize
    #         avgSlopeY = (slopeAbove + slopeBelow) / 2.0
    #         gradient[y, x, 1] = avgSlopeY

    kernelX = np.array([-0.5, 0, 0.5]).reshape(1, 3)
    kernelY = np.array([-0.5, 0, 0.5]).reshape(3, 1)
    gradientX = cv2.filter2D(image, -1, kernelX)
    gradientY = cv2.filter2D(image, -1, kernelY)

    # remove border artifacts?
    gradientX[:, 0] = 0
    gradientX[:, width-1] = 0
    gradientX[0, :] = 0
    gradientX[height-1, :] = 0

    gradientY[:, 0] = 0
    gradientY[:, width-1] = 0
    gradientY[0, :] = 0
    gradientY[height-1, :] = 0

    return gradientX, gradientY

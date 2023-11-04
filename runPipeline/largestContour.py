import cv2
import numpy as np


def runPipeline(image, llrobot):
    # Convert the image to grayscale
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Sobel filters for x and y directions
    # The Sobel operator is used for edge detection,
    # and it works by computing the gradient magnitude
    # of the image intensity function
    sobel_x_filter = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)

    sobel_y_filter = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=np.float32)

    # Apply Sobel filters
    sobel_x = cv2.filter2D(img_grey, cv2.CV_64F, sobel_x_filter)
    sobel_y = cv2.filter2D(img_grey, cv2.CV_64F, sobel_y_filter)

    # Calculate the magnitude of the gradient at each pixel in the image.
    # The gradient magnitude is a measure of how quickly the intensity of the image is changing at that pixel,
    # which is high at the edges in the image.
    # The Sobel operator gives us two images: sobel_x and sobel_y.
    # sobel_x contains the rate of change of the image intensity in the x-direction,
    # and sobel_y contains the rate of change in the y-direction.
    # To find the overall rate of change at each pixel (the gradient magnitude),
    # we combine these two images using the Pythagorean theorem:
    # sqrt(sobel_x^2 + sobel_y^2).
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient magnitude image to be in the range [0, 255].
    # This is done so that the image can be properly visualized and used for further processing.
    # The normalization is performed by dividing each pixel in the gradient magnitude image
    # by the maximum value in that image, bringing all the values within the range [0, 1].
    # Then, the values are scaled up to the range [0, 255] by multiplying by 255.
    # Finally, the image is converted to 8-bit unsigned integer format, which is a standard image format.
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

    # Apply a threshold operation to the gradient magnitude image.
    # All pixel values greater than 50 are set to 255 (white), while all others are set to 0 (black),
    # effectively creating a binary image. This helps in highlighting the stronger edges in the image.
    # The first return value (here assigned to _) is the threshold value used, but since it's the same
    # as the value we provided (50), it is not needed. The second return value is the resultant binary image.
    _, binary_image = cv2.threshold(
        sobel_magnitude, 50, 255, cv2.THRESH_BINARY)

    # Extract contours from the binary image.
    # 'contours' is a list of all the contours found in the image, where each contour is represented
    # as a list of points.
    # The second return value (here assigned to _) represents the hierarchy of the contours,
    # but it is not used in this case.
    # 'cv2.RETR_EXTERNAL' is a flag that tells OpenCV to retrieve only the extreme outer contours,
    # which helps in reducing the complexity of the image and speeds up the processing.
    # 'cv2.CHAIN_APPROX_SIMPLE' is another flag that compresses horizontal, vertical, and diagonal segments
    # and leaves only their end points, which is efficient and sufficient for further processing in many cases.
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largestContour = max(
        contours, key=cv2.contourArea) if contours else np.array([])

    # Convert the grayscale image to 3 channels
    img_grey_3_channels = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

    # for contour in contours:
    #     cv2.drawContours(img_grey_3_channels, [contour], -1, (0,0,250))
    # Optionally, draw the largest contour on the original image
    if largestContour.size > 0:
        cv2.drawContours(img_grey_3_channels, [
                         largestContour], -1, (34, 105, 240), 2)
    # Prepare llpython data
    llpython = []
    return largestContour, img_grey_3_channels, llpython

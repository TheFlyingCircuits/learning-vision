import cv2
import numpy as np


def runPipeline(image, llrobot):
    # Convert the image to grayscale
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Sobel kernels for x and y directions
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)

    sobel_y_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=np.float32)

    # Filter the image using the defined kernels
    gradient_x = cv2.filter2D(img_grey, cv2.CV_64F, sobel_x_kernel)
    gradient_y = cv2.filter2D(img_grey, cv2.CV_64F, sobel_y_kernel)

    # Calculate the gradient magnitude
    mag = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the magnitude based on the theoretical maximum (similar to the second example)
    maxMag = np.sqrt((1**2 + 1**2)) * 0.25
    mag = mag / maxMag
    mag[mag > 1] = 1  # Clipping to the maximum value
    scaledMag = mag * 255 * 2  # Scale up to enhance visibility
    mag = scaledMag.astype(np.uint8)

    # Calculate the gradient angle
    angle = np.arctan2(gradient_y, gradient_x)

    # Normalize the angle to the range [0, 179] for the HSV color space
    angle_normalized = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)

    # Create a full saturation channel
    saturation = np.uint8(np.full(img_grey.shape, 255))

    # Merge channels to get the HSV image
    hsv_image = cv2.merge([angle_normalized, saturation, mag])

    # Convert the HSV image to BGR to display using OpenCV
    bgr_outputImage = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Prepare llpython data, if any calculations are needed
    llpython = []

    # Assuming largestContour is needed as per the original code
    # Here, simply set to empty since no contours processing is done
    largestContour = np.array([[]])

    return largestContour, bgr_outputImage, llpython

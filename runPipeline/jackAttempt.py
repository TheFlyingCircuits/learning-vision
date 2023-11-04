import cv2
import numpy as np


def runPipeline(image, llrobot):
    # Convert the input image from BGR (Blue, Green, Red) color space to grayscale.
    # This is because the Sobel operator used for edge detection typically operates on single channel images.
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the Sobel kernel for the x-direction. This kernel will respond strongly to edges running vertically.
    # The values are derived from the Sobel operator which approximates the derivative of the image intensity in the x-direction.
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)

    # Define the Sobel kernel for the y-direction. This kernel will respond to edges running horizontally.
    # As with the x-direction kernel, it is designed to approximate the derivative of image intensity, but in the y-direction.
    sobel_y_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=np.float32)

    # Apply the Sobel x-kernel to the grayscale image to get the gradient in the x-direction.
    # The output image depth is set to cv2.CV_64F to avoid overflow.
    gradient_x = cv2.filter2D(img_grey, cv2.CV_64F, sobel_x_kernel)

    # Similarly, apply the Sobel y-kernel to get the gradient in the y-direction.
    gradient_y = cv2.filter2D(img_grey, cv2.CV_64F, sobel_y_kernel)

    # Calculate the magnitude of the gradient.
    # The magnitude represents the rate of change of brightness in the image and is calculated using the Pythagorean theorem.
    # cv2.cartToPolar computes the magnitude and angle of the 2D vectors formed by the gradients.
    mag, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

    # Normalize the gradient magnitude to the range [0, 255] as typical for 8-bit images.
    # This is done by clipping the magnitude to the range [0, 255], then scaling it based on the min and max values present.
    mag = np.clip(mag, 0, 255)
    minMag, maxMag = mag.min(), mag.max()
    if maxMag != minMag:  # To avoid division by zero when normalizing.
        mag = (mag - minMag) * (255.0 / (maxMag - minMag))
    mag = mag.astype(np.uint8)

    # Normalize the angle of the gradient to the range [0, 179] as used in the HSV color space.
    # The angles are in degrees from the cv2.cartToPolar function, but the H channel in HSV can only accommodate values from 0 to 179.
    angle_normalized = ((angle / 360.0) * 179).astype(np.uint8)

    # Create a saturation channel for the HSV image with full saturation for all pixels.
    # This allows the visualization of the gradient magnitude in terms of brightness with color indicating direction.
    saturation = np.uint8(np.full(img_grey.shape, 255))

    # Merge the normalized angle, full saturation, and normalized magnitude to form an HSV image.
    # This image can now represent the direction of edges (Hue) and the strength of edges (Value) with full saturation.
    hsv_image = cv2.merge([angle_normalized, saturation, mag])

    # Convert the HSV image back to the BGR color space for displaying with OpenCV functions.
    # OpenCV typically uses BGR, so this conversion allows us to view the result in a familiar color space.
    bgr_outputImage = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Prepare data for llrobot if needed; currently, this is just a placeholder empty list.
    llpython = []

    # Placeholder for the largestContour data. In this code snippet, it's an empty array
    # since we haven't performed any contour detection or processing.
    largestContour = np.array([[]])

    # Return a tuple with the placeholder largestContour, the edge visualization image in BGR format, and the llrobot data.
    return largestContour, bgr_outputImage, llpython

import cv2
import numpy as np

def rref(matrix):
    """
    Convert a matrix into Reduced Row Echelon Form.

    Args:
    matrix (np.array): A numpy matrix.

    Returns:
    np.array: The matrix in Reduced Row Echelon Form.
    """

    # Convert the matrix to float type to avoid integer division issues
    # This ensures that our calculations are done with decimal numbers, not integers.
    A = matrix.astype(np.float64)
    rows, cols = A.shape  # Get the number of rows and columns of the matrix

    # Initialize pivot row index
    r = 0

    for c in range(cols):
        # Identify the pivot for the current column
        # The pivot is the first non-zero element in the column below the current row
        pivot = np.where(A[r:rows, c] != 0)[0]
        if len(pivot) == 0:
            continue  # If no pivot is found in this column, skip to the next column

        print(pivot)

        pivot_row = r + pivot[0]  # Get the actual row index of the pivot

        # Swap the current row with the pivot row to bring the pivot to the diagonal
        A[[r, pivot_row]] = A[[pivot_row, r]]

        # Normalize the pivot row (make the pivot element 1)
        # This is done by dividing the entire row by the pivot element
        A[r] = A[r] / A[r, c]

        # Eliminate all other elements in the current column
        # This is achieved by subtracting a suitable multiple of the pivot row from the other rows
        for i in range(rows):
            if i != r:
                # Subtract the multiple of the pivot row from the current row
                # The multiple is calculated to make the element in the pivot column zero
                A[i] = A[i] - A[i, c] * A[r]

        r += 1  # Move to the next row
        if r == rows:
            break  # Exit loop if we have processed all rows

    return A  # Return the matrix in Reduced Row Echelon Form

kFocalLength = 1

#pixelPoints - array of 4 [x, y] in pixels from camera
#relativePoints - array of 4 [x, y] relative to AprilTag coordinate center
def solvePointPNP(pixelPoints, relativePoints):


    solveMatrix = np.array([])


    for i in range(4):

        pointPixelCoords = pixelPoints[i]
        pointRelativeCoords = relativePoints[i]

        #the xyz coordinates of our point can be described as a linear combination of the following:
        #[r11, r21, r31, r21, r22, r23, r31, r32, r33, t1, t2, t3]
        #with corresponding weights
        xWorldWeights = np.array([pointRelativeCoords[0], 0, 0, pointRelativeCoords[1], 0, 0, 0, 0, 0, 1, 0, 0])
        yWorldWeights = np.array([0, pointRelativeCoords[0], 0, 0, pointRelativeCoords[1], 0, 0, 0, 0, 0, 1, 0])
        zWorldWeights = np.array([0, 0, pointRelativeCoords[0], 0, 0, pointRelativeCoords[1], 0, 0, 0, 0, 0, 1])

        solveMatrix = np.append(solveMatrix, kFocalLength*xWorldWeights - pointPixelCoords[0]*zWorldWeights)
        solveMatrix = np.append(solveMatrix, kFocalLength*yWorldWeights - pointPixelCoords[1]*zWorldWeights)

    np.set_printoptions(precision=2)

    print("unsolved\n", solveMatrix)

    solveMatrix = rref(np.reshape(solveMatrix, (8, -1)))
    
    print("solved\n", solveMatrix)

    #coefficients for the basis of the null space of solveMatrix
    #the free variable here is always t3 according to simon
    coefficients = -1*solveMatrix[:, -1]

    #print(coefficients)
    #since the rotation matrix is orthonormal, you can directly solve for t3
    t3 = 1/np.sqrt(coefficients[0]**2+coefficients[1]**2+coefficients[2]**2)

    solvedVector = np.append(t3*coefficients,t3)
    print(solvedVector)

    #print(solvedVector)
    return solvedVector
    
    #DO CROSS PRODUCT TO FIND THIRD COLUMN OF R, RIGHT NOW IT IS 0s


tagLength = 1

#points MUST be 2, 1, 3, 4 quadrant order for opencv implementation
objectPoints = np.array([[-tagLength/2,tagLength/2,0],
                         [tagLength/2,tagLength/2,0],
                         [tagLength/2,-tagLength/2,0],
                         [-tagLength/2,-tagLength/2,0]])
imagePoints = np.array([[-0.264,-0.421],
                        [-0.148,-0.214],
                        [-0.235,-0.075],
                        [-0.391,-0.300]])
cameraMatrix = np.array([[kFocalLength, 0, 0],
                         [0, kFocalLength, 0],
                         [0, 0, 1]])

retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs = None, flags=cv2.SOLVEPNP_IPPE_SQUARE)



print(cv2.Rodrigues(rvec)[0], '\n', tvec)

#rotLength = np.linalg.norm(rvec)



#pixel points rel to center, april tag corners rel to apriltag center
#solvePointPNP([[-0.094,-0.199],[-0.028,-0.074],[-0.152,-0.951],[-0.072,0.034]], [[-0.5,0.5],[0.5,0.5],[-0.5,-0.5],[0.5,-0.5]])


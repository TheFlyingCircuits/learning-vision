import numpy as np

import runPipeline.largestContour as largestContour
import runPipeline.jackAttempt as jackAttempt
import runPipeline.exampleGoal as exampleGoal
import runPipeline.pretty as pretty


def runPipeline(image, llrobot, pipeline_num):
    """
    Run a specified image processing pipeline on an input image.

    This function selects and executes one of several predefined image processing
    pipelines based on the given pipeline number. Each pipeline is implemented
    in a separate module and is designed to perform certain image manipulations
    and analysis. If the pipeline number is 0 or any other number that doesn't
    correspond to a specific pipeline, the function returns the original image
    with no processing applied.

    ### Parameters:
    image : ndarray
        The input image on which the pipeline will operate.
    llrobot : object
        A robot-specific object that may contain state, configuration, or other
        information that can be used or modified by the pipeline functions.
    pipeline_num : int
        An integer that determines which image processing pipeline to run.
        - 0: No processing, return the original image.
        - 1-9: Run different pipelines. No processing if pipeline doesn't exist at value.

    ### Returns:
    tuple
        A tuple containing three elements:
        - The first element is a numpy array, which could be an empty array if no processing is done or some data produced by the pipeline.
        - The second element is the processed image (ndarray) if a processing pipeline is run, otherwise the original image.
        - The third element is the `llrobot` object which may or may not have been altered by the pipeline.

    ### Example:
    >>> processed_data, processed_image, updated_llrobot = runPipeline(input_image, robot_state, 1)
    >>> cv2.imshow('Processed Image', processed_image)
    """

    if pipeline_num == 0:
        # If it is 0, return an empty numpy array, the original image, and the llrobot variable unchanged
        return np.array([]), image, llrobot
    elif pipeline_num == 1:
        return largestContour.runPipeline(image, llrobot)
    elif pipeline_num == 2:
        return jackAttempt.runPipeline(image, llrobot)
    elif pipeline_num == 3:
        return exampleGoal.runPipeline(image, llrobot)
    elif pipeline_num == 4:
        return pretty.runPipeline(image, llrobot)

    # If none of the above conditions are met (e.g., an unsupported 'pipeline_num' is provided),
    # return an empty numpy array, the original image, and the llrobot variable unchanged
    return np.array([]), image, llrobot

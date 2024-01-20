import cv2
import time
import pipelineManager as pm
import os

if __name__ == '__main__':
        

    # Initialize the capture object to access the webcam; '0' signifies the default camera
    cap = cv2.VideoCapture(0)
    # Read an image from the filesystem, this image is loaded once for 'pic' mode
    img = cv2.imread("images/GeneralField1.jpg")

    # Define the mode of operation, 'cam' for webcam, 'pic' for static image processing
    mode = "pic"

    # Check if the webcam capture object was successfully created
    if not cap.isOpened():
        # Print an error message if the webcam cannot be accessed
        print("Error opening webcam")
        # Exit the program with an error code 1, indicating an issue occurred
        exit(1)

    # Variable to keep track of which pipeline to run
    pipeline_num = 5
    # Set the width of the frames captured from the webcam to 1280 pixels
    cap.set(3, 1280)
    # Set the height of the frames captured from the webcam to 960 pixels
    cap.set(4, 960)
    # Set the frame rate (frames per second) of the webcam capture to 22
    cap.set(cv2.CAP_PROP_FPS, 22)

    # Record the previous frame's time to calculate FPS; initialized to the current time initially
    prev_frame_time = 0

    # Choose the font type for the FPS display on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    #generates list of all files in images folder
    imgFiles = os.listdir("images")

    # Start an infinite loop to continuously capture frames from the webcam
    while(True):
        # If the mode is 'cam', capture from the webcam
        if mode == 'cam':
            # Read a single frame from the webcam
            ret, frame = cap.read()
            # If frame capture failed, ret will be False
            if not ret:
                # Print an error message indicating frame capture failure
                print("Failed to grab frame")
                # Break out of the loop if no frame is captured
                break
        # If the mode is not 'cam', use the loaded static image
        else:
            frame = img.copy()  # Uses a copy of the image to prevent it from smearing the FPS count

        # Process the frame using a custom pipeline function, which returns a tuple where the second item is the processed frame
        pipeline_out = pm.manage(frame, [], pipeline_num)
        frame = pipeline_out[1]

        # Capture the current time to calculate FPS
        new_frame_time = time.time()

        # Calculate the time difference
        time_diff = new_frame_time - prev_frame_time

        # Check if the time difference is zero to prevent division by zero
        if time_diff > 0:
            # Calculate the FPS using the time difference between the current and previous frame
            fps = 1 / time_diff
        else:
            # If the time difference is zero, handle it as an exceptional case, e.g., set FPS to a very high number or keep the last known value
            # This sets the FPS to infinity, which indicates the frame was processed instantly
            fps = float('inf')

        # Update the previous frame's time to the current time for the next iteration
        prev_frame_time = new_frame_time

        # Convert the FPS to an integer and then to a string to display it, unless it's 'inf' which means instant
        if fps != float('inf'):
            fps_text = str(int(fps))
        else:
            # You can choose to display 'inf' or 'MAX' or any other indicator that makes sense in your context
            fps_text = "MAX"

        # Draw the FPS on the frame using the specified font, color (BGR), and line type
        cv2.putText(frame, fps_text, (10, 30), font,
                    1, (34, 105, 240), 2, cv2.LINE_AA)

        # Display the frame in a window named 'Vision'
        cv2.imshow("Vision", frame)

        # Wait for 1 ms for a key press event and mask it with 0xFF to get the last 8 bits
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, break out of the loop to end the program
        if key == ord('q'):
            break

        # Check if one of the keys for changing resolution or FPS is pressed
        #TODO: this doesn't work
        if key in [ord('z'), ord('x'), ord('c'), ord('v')]:
            # Stop capturing from the webcam before changing settings
            cap.release()

            # Reinitialize the capture object to apply the new settings
            cap = cv2.VideoCapture(0)

            # Change the resolution and FPS based on the key pressed
            if key == ord('z'):
                cap.set(3, 320)  # Set width to 320 pixels
                cap.set(4, 240)  # Set height to 240 pixels
                cap.set(cv2.CAP_PROP_FPS, 90)  # Set FPS to 90
            elif key == ord('x'):
                cap.set(3, 640)  # Set width to 640 pixels
                cap.set(4, 480)  # Set height to 480 pixels
                cap.set(cv2.CAP_PROP_FPS, 90)  # Set FPS to 90
            elif key == ord('c'):
                cap.set(3, 960)  # Set width to 960 pixels
                cap.set(4, 720)  # Set height to 720 pixels
                cap.set(cv2.CAP_PROP_FPS, 22)  # Set FPS to 22
            elif key == ord('v'):
                cap.set(3, 1280)  # Set width to 1280 pixels
                cap.set(4, 960)  # Set height to 960 pixels
                cap.set(cv2.CAP_PROP_FPS, 22)  # Set FPS to 22

        # Check if the key pressed is a digit key to select
        if 48 <= key <= 57:  # ASCII values for '0' to '9'
            fileName = "images/" + imgFiles[key-48]
            print(fileName)
            img = cv2.imread(fileName)

        # Toggle between 'cam' and 'pic' modes when the spacebar is pressed
        if key == 32:  # ASCII value for the spacebar
            # If the current mode is 'pic', switch to 'cam', else switch to 'pic'
            if mode == "pic":
                mode = "cam"
            else:
                mode = "pic"

        if key == 13:  # ASCII value for enter
            
            pm.reloadPipelines()  # Reloads the pipelines without needing to restart entire script
            
        if key == ord('a'):
            #switch to default open pipeline
            if pipeline_num==5:
                pipeline_num = 0
            else:
                pipeline_num =5

    # After breaking out of the loop, release the webcam resource
    cap.release()

    # Destroy all OpenCV windows to clean up any remaining GUI elements
    cv2.destroyAllWindows()
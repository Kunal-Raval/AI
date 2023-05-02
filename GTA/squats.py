import cv2

# Load the cascade classifier for detecting the body
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Set up the video capture device
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Set the parameters for the high knees exercise
min_height = 100
max_height = 400
min_knee_height = 200

# Loop over the frames of the video
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame")
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the bodies in the frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50), maxSize=(500, 500))

    # Check if any bodies were detected
    if len(bodies) == 0:
        print("Error: No body detected")
        continue

    # Loop over the detected bodies
    for (x,y,w,h) in bodies:
        # Check if the body height is within the expected range
        if h < min_height or h > max_height:
            print("Incorrect: Body height not within expected range")
            continue

        # Calculate the height of the knees relative to the body height
        knee_height = y + h/2

        # Check if the knee height is above the minimum expected height
        if knee_height < min_knee_height:
            print("Incorrect: Knees not lifted high enough")
            continue

        # If all checks pass, the exercise is performed correctly
        print("Correct: High knees exercise performed correctly")

    # Display the current frameq
    cv2.imshow('frame', frame)

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()

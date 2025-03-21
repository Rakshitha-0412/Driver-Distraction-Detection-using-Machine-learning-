# Import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco/yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco/yolov3.cfg"])

# Load our YOLO object detector trained on the COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream, pointer to the output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)
(W, H) = (None, None)

# Define the positions of vertical and horizontal lines to divide the frame
vertical_line_position = 0
horizontal_line_position = 0

# Loop over frames from the video file stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # If the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        vertical_line_position = W // 2
        horizontal_line_position = H // 2

    # Construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # Scale the bounding box coordinates back relative to
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    # Draw vertical line
    cv2.line(frame, (vertical_line_position, 0), (vertical_line_position, H), (0, 255, 0), 2)

    # Draw horizontal line
    cv2.line(frame, (0, horizontal_line_position), (W, horizontal_line_position), (0, 255, 0), 2)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Determine the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Determine in which part of the frame the center is located
            if center_x < vertical_line_position and center_y < horizontal_line_position:
                part_indicator = "FOOD"
            elif center_x >= vertical_line_position and center_y < horizontal_line_position:
                part_indicator = "WATER"
            elif center_x < vertical_line_position and center_y >= horizontal_line_position:
                part_indicator = "MEDICINE"
            else:
                part_indicator = "EMERGENCY"

            # Draw a bounding box rectangle and label on the frame
            text = "{} in {}".format(LABELS[classIDs[i]], part_indicator)
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
vs.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()

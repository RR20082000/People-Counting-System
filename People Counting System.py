from centroidtracker import CentroidTracker #measure object dispersion and distance and update in each frame respectively
from trackableobject import TrackableObject #simply assign ID to the centroid. basically take output of the CentroidTracker and assign it an ID
from imutils.video import FPS #this FPS library is used to manage the frame rate in the code
import imutils #We use it to resize the frame and display it
import numpy as np
import dlib, cv2 #dlib is mainly used to cnage the colour scheme from BGR to RGB for better computation
                #CV2 is used for captuuring video, drawing over the frames, read each frame, etc

maximum_capacity = int(input("Enter maximum capacity of the area: "))

#MobileNet is a lightweight deep neural network architecture designed for embedded vision applications
#which use classification mechanism similar to logistic regression
#Single shot object detection or SSD takes one single shot to detect multiple objects within the image.
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#capture the video as input to a 'vs' variable. we use this variable further in the code
vs = cv2.VideoCapture('New video.mp4')

W = None #defining the width of the frame keeping default blank
H = None #defining the height of the frame keeping default blank
skip_frames = 30 #A threshold of skipping frames used further in code
conf = 0.4 #Second threshold to find the confidence of the model

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
x = []
empty = []
empty1 = []

# start the frames per second throughput estimator in the FPS library in imutils library
fps = FPS().start()

# loop over all the frames from the video stream
while True:
    frame = vs.read() #read the incoming frame
    frame = frame[1] #store it in the array

    if frame is None: #break the loop if there is no more frame being captured.
        break

    # resize the frame to have a maximum width of 500 pixels (the less data we have, the faster we can process it)
    # then convert the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    status = "Waiting"
    rects = []

    #if in any given case, the total frame modded 30 gives 0, thats when we set status and initialize our new set of object trackers
    if totalFrames % (skip_frames) == 0: #this is the case almost everytime since 0 mod 30 is always 0 except for when its the end of file
        status = "Detecting"
        print("detecting")
        trackers = [] #initialise the tracker array

        # convert the frame to a blob and pass the blob through the network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.001943, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > conf:
                # extract the index of the class label from the detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than object *detectors* to obtain a higher frame processing throughput
    else:
        for tracker in trackers: #loop over the trackers
            status = "Tracking" # set the status of our system to be 'tracking' rather than 'waiting' or 'detecting
            tracker.update(rgb) # update the tracker
            pos = tracker.get_position() # grab the updated position

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were moving 'up' or 'down'
    cv2.line(frame, (0, H // 2), (W, H // 2), (255,255,255), 3)
    #cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 3)  # Vertical Line

    # use the centroid tracker to associate the (1) old object centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
        else: # otherwise, there is a trackable object so we can utilize it to determine direction

            # the difference between the y-coordinate of the *current* centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object is moving up) AND
                # the centroid is above the center line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    empty.append(totalUp)
                    to.counted = True

                # if the direction is positive (indicating the object is moving down) AND
                # the centroid is below the center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    empty1.append(totalDown)
                    to.counted = True

                x = []
                x.append(len(empty) - len(empty1))# compute the sum of total people inside
        print("Total people inside:", x)

        trackableObjects[objectID] = to # store the trackable object in our dictionary
        text = "ID {}".format(objectID)# draw both the ID of the object and the centroid of the object on the output frame
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

    # construct a tuple of information which will be on display
    info = [
        ("Enter", totalUp),
        ("Exit", totalDown),
        ("Status", status),
    ]

    info2 = [
        ("Total people inside", x),
    ]

    if len(empty) - len(empty1) < maximum_capacity:
        status = 'not full.'
    else:
        status = 'full.'

    info3 = [
        ("The area is,", status)
    ]

    info4 = [
        ("Vacant area is ,", maximum_capacity - len(empty) - len(empty1))
    ]


    # Display the output
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) #0,0,0 for black

    for (i, (k, v)) in enumerate(info2):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for (i, (k, v)) in enumerate(info3):
        text = "{} {}".format(k, v)
        cv2.putText(frame, text, (265, H - ((i * 20) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for (i, (k, v)) in enumerate(info4):
        text = "{} {}".format(k, v)
        cv2.putText(frame, text, (265, H - ((i * 20) + 120)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if cv2.waitKey(1) == 27: #on press of ASCII 27 key that translates to the escape key, break loop.
        break
    totalFrames += 1# increment the total number of frames processed thus far and then update the FPS counter
    fps.update()
    cv2.imshow("Application", frame) #display the output frame by frame as a video on the screen

# stop all Frames related activities and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows() #close any open windows of this program
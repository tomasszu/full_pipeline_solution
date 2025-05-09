import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import time
import torch

from SendDetections import SendDetections

#============================================================================= <<<<<<<<<<<<<<<<<<<<<<<<<<
#Choose camera recording 1. to 3. or 5.
CAM = 1
#============================================================================= <<<<<<<<<<<<<<<<<<<<<<<<<<

def detections_process(model, frame, tracker, send_detections):
    confidence_threshold = 0.6

    results = model(frame)[0]
    #print(results.boxes)

    #DIMITRIRIOS CASE TEST
    # reults = model.track()

    # print(results.boxes)

    

    detections = sv.Detections.from_ultralytics(results)
    #print(detections)

    #mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections = detections[np.isin(detections.class_id, CLASS_ID)]
    detections = detections[np.greater(detections.confidence, confidence_threshold)]
    detections = tracker.update_with_detections(detections)

    send_detections(frame, detections)

    send_detections.clear()
    
    # send_detections(frame, detections)

    return detections

def frame_annotations(detections, frame):

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    #print(detections)

    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )


    annotated_labeled_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )


    annotated_labeled_frame = trace_annotator.annotate(
        scene=annotated_labeled_frame,
        detections=detections
    )

    return annotated_labeled_frame

if CAM == 1:
    start, end = sv.Point(x=-500, y=292), sv.Point(x=1878, y=292)
    attention_vector1 = [[0,175],[1279,175]], ">"
    attention_vector2 = [[0,505],[1140,0]], ">"
    cap = cv2.VideoCapture(f'cam{CAM}_cuts2.avi')
elif CAM == 2:
    start, end = sv.Point(x=-500, y=711), sv.Point(x=1878, y=198)
    cap = cv2.VideoCapture(f'cam{CAM}_cuts2.avi')
    attention_vector1 = [[0,120],[1279,570]], ">"
    attention_vector2 = [[63,0],[412,960]], "<"
elif CAM == 3:
    start, end = sv.Point(x=-500, y=600), sv.Point(x=1278, y=300)
    attention_vector1 = [[0,100],[2086,400]], ">"
    attention_vector2 = [[1500,0],[1900,2000]], ">"
    cap = cv2.VideoCapture(f'cam{CAM}_cuts.avi')
elif CAM == 5:
    start, end = sv.Point(x=1600, y=200), sv.Point(x=2600, y=2500)
    attention_vector1 = [[0,3000],[2000,0]], ">"
    attention_vector2 = None
    cap = cv2.VideoCapture(f'cam{CAM}_cuts2.avi')

model = YOLO("yolov8x.pt")
model.to(device="cuda")
tracker = sv.ByteTrack()

f = open("output.txt", "w")

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

# Initialize sending class once
send_detections = SendDetections(CLASS_ID)


ret, frame = cap.read()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG' or 'MP4V'
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 960))



while ret:
    detections = detections_process(model, frame, tracker, send_detections)

    annotated_frame = frame_annotations(detections, frame)

    display = annotated_frame
    out.write(display)
    display = cv2.resize(display, (1280, 960))
    cv2.imshow("Vehicle Detection", display)
    # cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    ret, frame = cap.read()

 
cv2.destroyAllWindows()
cap.release()
f.close()

out.release()

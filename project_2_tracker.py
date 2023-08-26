import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initilaize object detection
od = ObjectDetection()
cap = cv2.VideoCapture("los_angeles.mp4")
# load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
count = 0
center_points_prf = []
tracking_obj = {}
track_id = 0
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    # center points current frame
    center_points_crf = []
    # detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for class_id, score, box in zip(class_ids, scores, boxes):
        (x, y, w, h) = box
        # print("Frame No", count, "", x, y, w, h)
        Cx = int(x + (w / 2))
        Cy = int(y + (h / 2))
        center_points_crf.append((Cx, Cy))
        class_name = classes[class_id]
        # cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 2)
        # cv2.putText(frame, str(round(score,2)), (x+150, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # only at the beginning we compare previous and current frame
    if count <= 2:

        for pt in center_points_crf:
            for pt2 in center_points_prf:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_obj[track_id] = pt
                    track_id += 1
    else:
        tracking_obj_copy=tracking_obj.copy()
        center_points_crf_copy = center_points_crf.copy()

        for object_id, pt2 in tracking_obj_copy.items():

            object_exits = False
            for pt in center_points_crf_copy:

                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                #update  ID object position

                if distance < 20:
                    tracking_obj[object_id] = pt
                    object_exits =True
                    if pt in center_points_crf:
                        center_points_crf.remove(pt)
                    continue



            #remove ids which were lost
            if not object_exits:

                tracking_obj.pop(object_id)
        # add new ids found
        for pt in center_points_crf:
            tracking_obj[track_id] =pt
            track_id +=1


    for object_id, pt in tracking_obj.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Trakcing objects")
    print(tracking_obj)

    print("cur frame")
    print(center_points_crf)

    print("prev frame")
    print(center_points_prf)

    cv2.imshow("Frame", frame)

    # make a copy of the points
    center_points_prf = center_points_crf.copy()
    key = cv2.waitKey(1)

    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

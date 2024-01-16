import cv2
from ultralytics import YOLO
import onepose
import numpy as np
from functools import cmp_to_key

detection_model = YOLO("yolov8x.pt")

def gen_keypts(imgpath):
    pose_estimiation_model = onepose.create_model()

    img = cv2.imread(imgpath)
    draw_img = img.copy()

    results = detection_model(img)[0]
    boxes = results.boxes.xyxy
    clses = results.boxes.cls
    probs = results.boxes.conf

    allkeypts = []

    def area(box):
        return (box[3].item()-box[1].item()) * (box[2].item()-box[0].item())

    def compare(x, y):
        return area(x[1]) - area(y[1])

    zipped = list(zip(clses, boxes, probs))
    zipped = sorted(zipped, key=cmp_to_key(compare), reverse = True)

    for [cls, box, prob] in zipped[:2]:
        if cls != 0:
            continue

        x1, y1, x2, y2 = box
        # crop image
        person_img = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        
        keypoints = pose_estimiation_model(person_img)
        num_keypoints = len(keypoints['points'])
        
        for i in range(num_keypoints):
            keypoints['points'][i][0] += x1
            keypoints['points'][i][1] += y1
        
        onepose.visualize_keypoints(draw_img, keypoints, pose_estimiation_model.keypoint_info, pose_estimiation_model.skeleton_info)

        def flatten(xss):
            return [x for xs in xss for x in xs]
        
        allkeypts.extend(flatten(keypoints['points'].tolist()))
    #cv2.imshow("draw_img", draw_img)
    #cv2.waitKey(0)
    
    cv2.imwrite("bteam_out1/"+imgpath[8:], draw_img)

    return allkeypts
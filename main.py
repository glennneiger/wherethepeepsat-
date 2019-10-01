import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings('ignore')

from imageai.Detection import ObjectDetection
import cv2

fps = 30
switch_thresh = 4
prob = 40
filename = '../test1.avi'
frame_skip = 3

rate = 1/fps

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('../yolo.h5')
detector.loadModel()

cap = cv2.VideoCapture(filename)

i = 0
time = 0
switch_count = []
while(True):
    ret, frame = cap.read()
    if ret:
        if i% frame_skip ==0:
            # print(f'checking frame {i}')
            custom = detector.CustomObjects(person=True)
            _, detections = detector.detectCustomObjectsFromImage(custom_objects=custom,
                    input_image=frame,
                    input_type='array',
                    output_type='array',
                    minimum_percentage_probability=prob)
            switch_count.append(min(len(detections),1))

            if i == 0:
                if len(detections) > 0:
                    status = 'person'
                else:
                    status = 'empty'
                print(f'initial status is {status}')

            else:
                if len(detections) > 0:
                    if status == 'empty' and sum(switch_count[-switch_thresh:]) == switch_thresh:
                        status = 'person'
                        print(f'{status} detected at {time // 60} min : {round(time % 60, 1)} sec')
                elif len(detections) == 0:
                    if status == 'person' and sum(switch_count[-switch_thresh:]) == 0:
                        status = 'empty'
                        print(f'{status} detected at {time // 60} min : {round(time % 60, 1)} sec')
    else:
        break
    i += 1
    time += rate




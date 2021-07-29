import numpy as np
import argparse, pytz, time
import cv2
from sendMSG import writeMsg
from math import pow, sqrt
from datetime import datetime
from workWithCSV import append_list_as_row

# Parse the arguments from command line
arg = argparse.ArgumentParser(description='Social distance and Face mask detection')

arg.add_argument('-v', '--video', default='', type = str,help = 'Video file path. If no path is given, video is captured using device.')
arg.add_argument('-c', '--confidence', type = float, default = 0.5, help = 'Minimum probability to filter weak detections')
arg.add_argument('-t', '--threshold', type = float, default = 0.4, help = 'Threshold when applying non-maxima suppression')
arg.add_argument('-s', '--use_sms', type=bool, default=0, help='SMS need to be send or not')
arg.add_argument('-e', '--use_email', type=bool, default=0, help='Email need to be send or not')
args = vars(arg.parse_args())

if args['use_sms']:
    from sendMSG import sendSMS
if args['use_email']:
    from sendMSG import sendEmail
IST = pytz.timezone('Asia/Ho_Chi_Minh')

device_name         = 'C001'
result_csv          = './Capture/result.csv'
CONFIDENCE_CUTOFF   = args['confidence']
NMS_THRESHOLD       = args['threshold']

# Load model to detect person
weight_person       = './model/person_detect/yolov4_tiny_person.cfg'
model_peron         = './model/person_detect/yolov4_tiny_person_best.weights'
net_person          = cv2.dnn.readNetFromDarknet(weight_person, model_peron)
net_person.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_person.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load model to detect Mask/No mask
weight_face         = './model/mask_detect/yolov4_mask_2class.cfg'
model_face          = './model/weights/yolov4_mask_2class_final.weights'
net_face            = cv2.dnn.readNetFromDarknet(weight_face, model_face)
net_face.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_face.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Labels (Mask/No mask)
classesFile = './model/mask_detect/object.names'
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Set color for Mask/No mask
colors = [(0,0,255), (0,255,0)]

# Image size
IMG_WIDTH, IMG_HEIGHT = 416, 416

# Focal length F = (P x D) / H (my Height (H) = 172, distance I stand between camera (D) = 360cm, Height of my Bounding Box (P) = 300 px)
F = 625
count_frame = 0
cur = 0
# Capture video from file or through device
if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def predict_box(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    output = net.getUnconnectedOutLayersNames()
    outs = net.forward(output)

    confidences = []
    boxes = []
    classIDs = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE_CUTOFF:
                x_mid, y_mid, w, h = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                x, y = int(x_mid - w//2), int(y_mid - h//2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_CUTOFF, NMS_THRESHOLD)

    final_box = []
    final_classIDs = []
    final_confidences = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_box.append(boxes[i])
            final_classIDs.append(classIDs[i])
            final_confidences.append(confidences[i])
        return final_box, final_classIDs, final_confidences


def draw_box(frame, boxes, classIDs, confidences, class_list, color_list):
    for i in range(len(boxes)):
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in color_list[classIDs[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = '{}: {:.4f}'.format(class_list[classIDs[i]], confidences[i])
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def calculate_distance(frame, boxes):
    position = {}
    close_objects = set()
    for i in range(len(boxes)):
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        x_mid = round(x + w/2, 4)
        y_mid = round(y + h/2, 4)

        distance_to_camera = (165 * F)/h

        x_mid_cm = (x_mid * distance_to_camera) / F
        y_mid_cm = (y_mid * distance_to_camera) / F

        position[i] = (x_mid_cm, y_mid_cm, distance_to_camera, x_mid, y_mid)

    for i in position.keys():
            for j in position.keys():
                if i < j:
                    distance = sqrt(pow(position[i][0]-position[j][0],2) + pow(position[i][1]-position[j][1],2) + pow(position[i][2]-position[j][2],2))
                # Check if distance less than 2 metres or 200 centimetres
                    if distance < 200:
                        close_objects.add(i)
                        close_objects.add(j)
                        # Draw line between middle point of boxes if < 200cm
                        cv2.line(frame, (int(position[i][3]), int(position[i][4])), (int(position[j][3]), int(position[j][4])), (0,0,255), 2)
                        # Put text to display distance between boxes if < 200cm
                        if position[i][3] <= position[j][3]:
                            x_center_line = int(position[i][3] + (position[j][3] - position[i][3])/2)
                        else:
                            x_center_line = int(position[j][3] + (position[i][3] - position[j][3])/2)
                        if position[i][4] <= position[j][4]:
                            y_center_line = int(position[i][4] + (position[j][4] - position[i][4])/2)
                        else:
                            y_center_line = int(position[j][4] + (position[i][4] - position[j][4])/2)
                        cv2.putText(frame, f'{int(distance)} cm', (x_center_line - 35, y_center_line - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    boxes_person_normal = []
    boxes_close = []
    for i in range(len(boxes)):
        if i in close_objects:
            boxes_close.append(boxes[i])
        else:
            boxes_person_normal.append(boxes[i])
    return boxes_close, boxes_person_normal


while cap.isOpened():
    ret, frame = cap.read()
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    result = frame.copy()

    boxes_face          = []
    boxes_person        = []
    boxes_close         = []
    boxes_person_normal = []

    try:
        boxes_face, classIDs_face, confidences_face = predict_box(net_face, result)
        draw_box(result, boxes_face, classIDs_face, confidences_face, classes, colors)

        # print(classIDs_face)
        mask_count = sum(classIDs_face)
        nomask_count = len(classIDs_face) - mask_count

        boxes_person, classIDs_person, confidences_person = predict_box(net_person, result)
        boxes_close, boxes_person_normal = calculate_distance(result, boxes_person)

        draw_box(result, boxes_person_normal, classIDs_person, confidences_person, ['Person'], [(0,255,0)])
        draw_box(result, boxes_close, classIDs_person, confidences_person, ['Person'], [(0,0,255)])

    except:
        pass

    border_size=100
    border_text_color=(255,255,255)
    style = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.copyMakeBorder(result, border_size, 0,0,0, cv2.BORDER_CONSTANT)

    text = 'NoMaskCount: {}  MaskCount: {}'.format(nomask_count, mask_count)
    cv2.putText(result,text, (5, int(border_size-70)), style, 0.65, border_text_color, 2)

    text = f'Social Distancing Violations: {len(boxes_close)}'
    cv2.putText(result, text, (5, int(border_size-30)), style, 0.65, border_text_color, 2)

    text = f'Camera ID: {device_name}'
    cv2.putText(result, text, (frame_width - 250, int(border_size-70)), style, 0.65, border_text_color, 2)

    text = f'Status:'
    cv2.putText(result, text, (frame_width - 250, int(border_size-30)), style, 0.65, border_text_color, 2)
    
    if (nomask_count == 0) and len(boxes_close) == 0:
        text = 'Safe'
        cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0, 255, 0), 2)
        count_frame = 0
    elif nomask_count >=2:
        text = 'Danger !!!'
        cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0, 0, 255), 2)

        count_frame+=1
        if count_frame >=7:
            # Capture image next image after few seconds:
            if time.time() - cur >= 10:
                # Save image
                datetime_ist = datetime.now(IST)
                image_name = str(datetime_ist.strftime('%Y-%m-%d_%H-%M-%S')) + f'_{device_name}'
                cv2.imwrite(f'./Capture/{image_name}.jpg', frame)
                # Write a message
                msg = writeMsg(device_name, nomask_count, mask_count, boxes_close, datetime_ist)
                print(msg)
                if args['use_sms']:
                    sendSMS(msg)
                # Send email
                if args['use_email']:
                    sendEmail(msg, f'./Capture/{image_name}.jpg')
                cur = time.time()
                count_frame=0
                save_list = [device_name, datetime_ist.strftime('%Y-%m-%d'), datetime_ist.strftime('%H:%M:%S'), mask_count, nomask_count, len(boxes_close)]
                append_list_as_row(result_csv, save_list)
    
    elif len(boxes_close) >= 3:
        text = 'Danger !!!'
        cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0, 0, 255), 2)

        count_frame+=1
        if count_frame >=30:
            # Capture image next image after few seconds:
            if time.time() - cur >= 20:
                # Save image
                datetime_ist = datetime.now(IST)
                image_name = str(datetime_ist.strftime('%Y-%m-%d_%H-%M-%S')) + f'_{device_name}'
                cv2.imwrite(f'./Capture/{image_name}.jpg', frame)
                # Write a message
                msg = writeMsg(device_name, nomask_count, mask_count, boxes_close, datetime_ist)
                print(msg)
                if args['use_sms']:
                    sendSMS(msg)
                # Send email
                if args['use_email']:
                    sendEmail(msg, f'./Capture/{image_name}.jpg')
                cur = time.time()
                count_frame=0
                save_list = [device_name, datetime_ist.strftime('%Y-%m-%d'), datetime_ist.strftime('%H:%M:%S'), mask_count, nomask_count, len(boxes_close)]
                append_list_as_row(result_csv, save_list)
    else:
        text = 'Warning !'
        cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0,255,255), 2)
        count_frame = 0
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    # Show frame
    cv2.imshow('Frame', result)
    # cv2.resizeWindow('Frame',800,600)
    key = cv2.waitKey(1) & 0xFF

    # Press `esc` to exit
    if key == 27:
        break
# Clean
cap.release()
cv2.destroyAllWindows()
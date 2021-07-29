import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import cv2
import os, glob, time
from math import pow, sqrt
from datetime import datetime
from workWithCSV import append_list_as_row, search_history, groupby_hour
from sendMSG import writeMsg

IST = pytz.timezone('Asia/Ho_Chi_Minh')
device_name = 'C001'
result_csv = './Capture/result.csv'

# Load model to detect person
weight_person = './model/person_detect/yolov4_tiny_person.cfg'
model_peron = './model/person_detect/yolov4_tiny_person_best.weights'
net_person = cv2.dnn.readNetFromDarknet(weight_person, model_peron)
net_person.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_person.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load model to detect Mask/No mask
weight_face = "./model/mask_detect/yolov4_mask_2class.cfg"
model_face = "./model/weights/yolov4_mask_2class_final.weights"
net_face = cv2.dnn.readNetFromDarknet(weight_face, model_face)
net_face.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_face.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Labels (Mask/No mask)
classesFile = "./model/mask_detect/object.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Set color for Mask/No mask
colors = [(0,0,255), (0,255,0)]

# Image size
IMG_WIDTH, IMG_HEIGHT = 416, 416

# Focal length F = (P x D) / H (my Height (H) = 172, distance I stand between camera (D) = 360cm, Height of my Bounding Box (P) = 300 px)
F = 625
cur = 0
count_frame = 0

st.sidebar.title('Select page')
page = st.sidebar.selectbox('Select page',['Run Detection','Captured Images', 'History'])
if page == 'Run Detection':

    # st.sidebar.title('Confidence')
    # pers = st.sidebar.slider('Person', min_value=0.1, max_value=1.0, step=0.1)
    # msk = st.sidebar.slider('Mask/No Mask', min_value=0.1, max_value=1.0, step=0.1)

    st.sidebar.title('Danger thresholds')
    nomask = st.sidebar.slider('No mask', min_value=1, max_value=10)
    close = st.sidebar.slider('Number of close people', min_value=2, max_value=20)

    st.sidebar.title('Alert')
    use_email = st.sidebar.checkbox('Alert via email')
    use_sms = st.sidebar.checkbox('Alert via SMS')

    st.sidebar.title('Alert Frequency')
    frequency = st.sidebar.slider('In Second', min_value=10, max_value=100, step=5)

    if use_email:
        from sendMSG import sendEmail
    if use_sms:
        from sendMSG import sendSMS

    st.title("Webcam Live Feed")
    run = st.checkbox('Open camera')
    FRAME_WINDOW = st.image([])
    if run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while cap.isOpened():
            ret, frame = cap.read()
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            result = frame.copy()
            # Detect person
            blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
            net_person.setInput(blob)
            output_layers_person = net_person.getUnconnectedOutLayersNames()
            outs_person = net_person.forward(output_layers_person)

            confidences_person = []
            boxes_person = []
            pos_dict = dict()
            coordinates = dict()

            for out_person in outs_person:
                for detection_person in out_person:
                    scores_person = detection_person[5:]
                    classID_person = np.argmax(scores_person)
                    confidence_person = scores_person[classID_person]
                    if confidence_person >= 0.5:
                        if classID_person == 0:
                            x2, y2, w2, h2 = detection_person[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                            p1 = int(x2 - w2//2), int(y2 - h2//2)

                            boxes_person.append([*p1, int(w2), int(h2)])
                            (x2, y2) = (boxes_person[0][0], boxes_person[0][1])
                            (w2, h2) = (boxes_person[0][2], boxes_person[0][3])
                            confidences_person.append(float(confidence_person))

            indices2 = cv2.dnn.NMSBoxes(boxes_person, confidences_person, 0.4, 0.3)
            if len(indices2) > 0:
                for i in indices2.flatten():
                    (x2, y2) = (boxes_person[i][0], boxes_person[i][1])
                    (w2, h2) = (boxes_person[i][2], boxes_person[i][3])
                    x_end = x2 + w2
                    y_end = y2 + h2
                    coordinates[i] = (x2, y2, x_end, y_end)

                    # Mid point of bounding box
                    x_mid = round(x2 + w2/2, 4)
                    y_mid = round(y2 + h2/2, 4)
                    # height = round(endY-startY, 4)
                    distance = (165 * F)/h2

                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm, y_mid_cm, distance, x_mid, y_mid)

            close_objects = set()
            for i in pos_dict.keys():
                for j in pos_dict.keys():
                    if i < j:
                        dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))
                        # Check if distance less than 200 centimetres
                        if dist < 200:
                            close_objects.add(i)
                            close_objects.add(j)
                            # Draw line between middle point of boxes if < 200cm
                            cv2.line(result, (int(pos_dict[i][3]), int(pos_dict[i][4])), (int(pos_dict[j][3]), int(pos_dict[j][4])), (0,0,255), 2)
                            # Put text to display distance between boxes if < 200cm
                            if pos_dict[i][3] <= pos_dict[j][3]:
                                x_center_line = int(pos_dict[i][3] + (pos_dict[j][3] - pos_dict[i][3])/2)
                            else:
                                x_center_line = int(pos_dict[j][3] + (pos_dict[i][3] - pos_dict[j][3])/2)
                            if pos_dict[i][4] <= pos_dict[j][4]:
                                y_center_line = int(pos_dict[i][4] + (pos_dict[j][4] - pos_dict[i][4])/2)
                            else:
                                y_center_line = int(pos_dict[j][4] + (pos_dict[i][4] - pos_dict[j][4])/2)
                            cv2.putText(result, f'{int(dist)} cm', (x_center_line - 35, y_center_line - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            # Draw bounding box of person

            for i in pos_dict.keys():
                if i in close_objects:
                    color_person = (0,0,255)
                else:
                    color_person = (0,255,0)

                x2, y2, x_end, y_end = coordinates[i]
                cv2.rectangle(result, (x2, y2), (x_end, y_end), color_person, 2)
                y = y2 - 15 if y2 - 15 > 15 else y2 + 15

            # Set model input
            net_face.setInput(blob)
            # Define the layers that we want to get the outputs from
            output_layers = net_face.getUnconnectedOutLayersNames()
            # Run 'prediction'
            outs_face = net_face.forward(output_layers)

            confidences = []
            boxes = []
            classIDs = []
            # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores. Assign the box's class label as the class with the highest score.
            for out_face in outs_face:
                for detection in out_face:
                    scores = detection[5:]
                    # Get label index
                    classID = np.argmax(scores)
                    confidence_face = scores[classID]
                    # Extract position data of face area (only area with high confidence)
                    if confidence_face >= 0.5:
                        x, y, w, h = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                        p0 = int(x - w//2), int(y - h//2)
                        boxes.append([*p0, int(w), int(h)])
                        confidences.append(float(confidence_face))
                        classIDs.append(classID)

            # Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            count = np.take(classIDs, indices)

            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color_face = [int(c) for c in colors[classIDs[i]]]
                    cv2.rectangle(result, (x, y), (x + w, y + h), color_face, 2)
                    text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                    cv2.putText(result, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_face, 1)

            # Create black border on top and put display text in it
            mask_count=(count==1).sum()
            nomask_count=(count==0).sum()

            border_size=100
            border_text_color=(255,255,255)
            style = cv2.FONT_HERSHEY_SIMPLEX
            result = cv2.copyMakeBorder(result, border_size, 0,0,0, cv2.BORDER_CONSTANT)

            text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
            cv2.putText(result,text, (5, int(border_size-70)), style, 0.65, border_text_color, 2)

            text = f"Social Distancing Violations: {len(close_objects)}"
            cv2.putText(result, text, (5, int(border_size-30)), style, 0.65, border_text_color, 2)

            text = f"Camera ID: {device_name}"
            cv2.putText(result, text, (frame_width - 250, int(border_size-70)), style, 0.65, border_text_color, 2)

            text = f"Status:"
            cv2.putText(result, text, (frame_width - 250, int(border_size-30)), style, 0.65, border_text_color, 2)
            
            if (nomask_count == 0) and len(close_objects) == 0:
                text = "Safe"
                cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0, 255, 0), 2)
                count_frame = 0
            elif nomask_count >=nomask:
                text = "Danger !!!"
                cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0, 0, 255), 2)

                count_frame+=1
                if count_frame >=7:
                    # Capture image next image after few seconds:
                    if time.time() - cur >= frequency:
                        # Save image
                        datetime_ist = datetime.now(IST)
                        image_name = str(datetime_ist.strftime("%Y-%m-%d_%H-%M-%S")) + f'_{device_name}'
                        cv2.imwrite(f'./Capture/{image_name}.jpg', frame)

                        # Write a message
                        msg = writeMsg(device_name, nomask_count, mask_count, close_objects, datetime_ist)
                        print(msg)
                        # Send SMS
                        if use_sms:
                            sendSMS(msg)
                        # Send email   
                        if use_email:
                            sendEmail(msg, f'./Capture/{image_name}.jpg')
                        count_frame=0
                        cur = time.time()
                        save_list = [device_name, datetime_ist.strftime("%Y-%m-%d"), datetime_ist.strftime("%H:%M:%S"), mask_count, nomask_count, len(close_objects)]
                        append_list_as_row(result_csv, save_list)
            elif len(close_objects) >=close:
                text = "Danger !!!"
                cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0, 0, 255), 2)
                # Eliminate the chance that people just walk through each other really fast
                count_frame+=1
                if count_frame >=30:
                    # Capture image next image after few seconds:
                    if time.time() - cur >= frequency:
                        # Save image
                        datetime_ist = datetime.now(IST)
                        image_name = str(datetime_ist.strftime("%Y-%m-%d_%H-%M-%S")) + f'_{device_name}'
                        cv2.imwrite(f'./Capture/{image_name}.jpg', frame)

                        # Write a message
                        msg = writeMsg(device_name, nomask_count, mask_count, close_objects, datetime_ist)
                        print(msg)
                        # Send SMS
                        if use_sms:
                            sendSMS(msg)
                        # Send email   
                        if use_email:
                            sendEmail(msg, f'./Capture/{image_name}.jpg')
                        count_frame=0
                        cur = time.time()
                        save_list = [device_name, datetime_ist.strftime("%Y-%m-%d"), datetime_ist.strftime("%H:%M:%S"), mask_count, nomask_count, len(close_objects)]
                        append_list_as_row(result_csv, save_list)
            else:
                text = "Warning !"
                cv2.putText(result, text, (frame_width - 170, int(border_size-30)), style, 0.65, (0,255,255), 2)
                count_frame = 0
            cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
            FRAME_WINDOW.image(result, use_column_width='always')

            # cv2.resizeWindow('Frame',800,600)
            key = cv2.waitKey(1) & 0xFF

            # Press `esc` to exit
            if key == 27:
                break

        # Clean
        cap.release()
        cv2.destroyAllWindows()

if page == 'Captured Images':
    # Search Image base on input date and time.
    st.sidebar.title('Search Image')
    cameraID_img = st.sidebar.selectbox('Camera ID', ['All', 'C001', 'C002'])
    from_date= int(''.join(str(st.sidebar.date_input('From:')).split('-')) + ''.join(str(st.sidebar.time_input('')).split(':'))[:6])
    to_date = int(''.join(str(st.sidebar.date_input('To:')).split('-')) + ''.join(str(st.sidebar.time_input(' ')).split(':'))[:6])

    st.title('Captured Images')
    captured_image = [i for i in glob.glob('./Capture/*.jpg')]
    show_image = []

    # Show all image between giving date and time.
    for img in captured_image:
        img_date = int(''.join(os.path.basename(img)[:10].split('-')) + ''.join(os.path.basename(img)[11:19].split('-')))
        captured_by = os.path.basename(img)[20:24]
        if cameraID_img != 'All':
            if img_date >= from_date and img_date <= to_date and captured_by == cameraID_img:
                show_image.append(img)
        else:
            if img_date >= from_date and img_date <= to_date:
                show_image.append(img)

    n_cols = 4
    n_rows = 1 + len(show_image) // n_cols
    rows = [st.beta_container() for _ in range(n_rows)]
    cols_per_row = [r.beta_columns(n_cols) for r in rows]

    for image_index, captured in enumerate(show_image):
        with rows[image_index // n_cols]:
            cols_per_row[image_index // n_cols][image_index % n_cols].image(captured)

if page == 'History':
    st.header('History')
    st.sidebar.header('Search')
    choose_ID = st.sidebar.selectbox('Camera ID', ['All', 'C001', 'C002'])
    from_history = dt.datetime.combine(st.sidebar.date_input('From:'), st.sidebar.time_input(''))
    to_history = dt.datetime.combine(st.sidebar.date_input('To:'), st.sidebar.time_input(' '))
    st.dataframe(search_history(from_history, to_history, choose_ID))
    chart = st.selectbox('Select chart',['Bar chart','Line chart'])
    if chart == 'Bar chart':
        st.bar_chart(search_history(from_history, to_history, choose_ID).groupby('Date').sum()[['No_mask_count', 'Social_distancing_violations']])
        try:
            st.bar_chart(groupby_hour(search_history(from_history, to_history, choose_ID)))
        except:
            pass
    elif chart == 'Line chart':
        st.line_chart(search_history(from_history, to_history, choose_ID).groupby('Date').sum()[['No_mask_count', 'Social_distancing_violations']])
        try:
            st.line_chart(groupby_hour(search_history(from_history, to_history, choose_ID)))
        except:
            pass
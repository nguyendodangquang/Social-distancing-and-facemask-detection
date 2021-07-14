# Social distancing and face mask detection
![g](https://user-images.githubusercontent.com/6376530/123517524-a92dfa80-d6cb-11eb-903c-c389b8d7dd4a.PNG)

## Why I started with this project?
Few months ago, Covid-19 happened one again in Vietnam and it’s more dangerous than ever. It has affected a lot of people, families, and companies. 
With Social distancing and wearing a face mask, we can slow the spreading speed of coronavirus.

![g](https://github.com/nguyendodangquang/Social-distancing-and-facemask-detection/blob/master/README/1_gpUdUKqu85R6usaayEMn1w.gif)

_By Katapult Magazin, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=88464207_


<img src="https://user-images.githubusercontent.com/6376530/123532747-1a0bfb80-d73a-11eb-8ff8-9e2232b6ac0a.PNG" width=60% height=60%>

## My project pipeline
![pipeline](https://user-images.githubusercontent.com/6376530/123532789-7e2ebf80-d73a-11eb-993a-bf5787dd79f3.PNG)

I used 2 separate data sets to train 2 separate custom yolov4 model. One model (Yolov4-tiny) is to detect if there is any person in the image captured from camera, and calculate the distance between 2 or more people in the image, then check if the distance is lower than 2 meter or not, the other model (yolov4) is to detect if a person is wearing a mask or not.

## Why yolov4?
Yolov4 has high accuracy and fast, to perform real-time detection compare to others models.

<img src="https://user-images.githubusercontent.com/6376530/123533038-76701a80-d73c-11eb-93ff-eada100edfd0.jpg" width=50% height=50%>

## Face mask detection (yolov4)

This dataset has total 1034 images, I used 80% of the data for training, 20% for validation.

Final result (mAP): **88,38%**

You can download **weights** for face mask detection here: https://drive.google.com/file/d/1rVnBIqvlinL2BUONjOaFDj8PM9Yw4k1q/view?usp=sharing

## Person detection (yolov4-tiny)

This dataset has total 902 images, I used 80% of the data for training, 20% for validation.

Final result (mAP): **98,33%**

## How I calculated distance between people?
### 1. Calculate focal length of camera

![unnamed](https://user-images.githubusercontent.com/6376530/123533143-3493a400-d73d-11eb-9fac-d245ef8f4640.gif)

Each camera have a unique focal length. I calculated mine by stand front the camera, measure distance between me and the camera (d), take my real height (R = 172cm), and calculated height of bounding box of me (from result when detected by yolov4-tiny) in pixel.

Calculate focal length **f = (r x d) / R**

### 2. Calculate distance between people to camera when detecting
With f, we can calculate distance between objects and the camera:
- R: real height of a person (I assumed average height of a person is 165cm)
- r: height of bounding box the person (in pixel)
- **d**:  **d = (r x f) / R**

### 3. Calculate Euclidean distance distance between 2 people

<img src="https://user-images.githubusercontent.com/6376530/123533401-6efe4080-d73f-11eb-8e63-fe89c0d75fc7.png" width=50% height=50%>

We already calculate distance d between each person to camera and with position giving by yolov4 when detect a person (middle point of the bounding box), we can calculate Euclidean distance by using this formula:

<img src="https://user-images.githubusercontent.com/6376530/123912882-c7d01200-d9a7-11eb-848b-47470e50f52f.png" width=50% height=50%>

## How to run
### Run with streamlit

```
streamlit run app.py
```

### Run with python

Real-time detect using camera:

```
python main.py --confidence 0.5
```

Run with video:

```
python main.py --confidence 0.5 --video <path_to_video>
```

Use **email and sms to alert** (need to change phone number (using Twilio), account and password of your email in sendMSG.py)

```
python main.py --confidence 0.5 --use_sms 1 --use_email 1
```
## Reference
https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

https://github.com/Subikshaa/Social-Distance-Detection-using-OpenCV

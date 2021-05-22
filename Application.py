# Importing required libraries, obviously
import streamlit as st
import cv2, time, pandas
from datetime import datetime
from PIL import Image
import numpy as np
import os
#import MotionDetection

# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def detect(image):

    image = np.array(image.convert('RGB'))

    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=4)

        roi = image[y:y + h, x:x + w]


    return image, faces


def motion():
    static_back = None

    motion_list = [None, None]

    time = []

    df = pandas.DataFrame(columns=["Start", "End"])

    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()

        motion = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if static_back is None:
            static_back = gray
            continue

        diff_frame = cv2.absdiff(static_back, gray)

        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(thresh_frame.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            motion = 1

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        motion_list.append(motion)

        motion_list = motion_list[-2:]

        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(datetime.now())

        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(datetime.now())

        cv2.imshow("Gray Frame", gray)

        cv2.imshow("Difference Frame", diff_frame)

        cv2.imshow("Threshold Frame", thresh_frame)

        cv2.imshow("Color Frame", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            if motion == 1:
                time.append(datetime.now())
            break

    for i in range(0, len(time), 2):
        df = df.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)

    df.to_csv("Time_of_movements.csv")

    video.release()

    cv2.destroyAllWindows()

def about():
    st.write(
        '''
        1. **Face Detection**:
        
        **Haar Cascade** is an object detection algorithm. It can be used to detect objects in images or videos. 
        
        The algorithm has four stages:
        
            1. Haar Feature Selection
            2. Creating  Integral Images
            3. Adaboost Training
            4. Cascading Classifiers
            
        Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
        https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid

        2. **Motion Detection**:
        
        Motion Detector will allow you to detect motion and also store the time interval of the motion.
        Videos can be treated as stack of pictures called frames. 
        Different frames(pictures) are compared to the first frame which should be static(No movements initially). 
        We compare two images by comparing the intensity value of each pixels.
        Also, we store the time interval of motion in a CSV file. 
        
        Motion Detector has four Windows:
        
            1. Gray Frame
            2. Difference Frame
            3. Threshold Frame
            4. Color Frame

        ''')


def contact():
    st.write(
        '''
        1. **Vishal Kaushik** [LinkedIn](https://www.linkedin.com/in/vishal-kaushik-7b0255195/)
        
            Email: vishalkaushik0804@gmail.com            
            Mobile: 8847075281                 

        2. **Vaibhav Jain** [LinkedIn](https://www.linkedin.com/in/vaibhav-jain-07111999/)
        
            Email: vaibhavjain602@gmail.com   
            Mobile: 7206097051
            
            
        ''')

#@st.cache
def get_data(filename):
    date_data = pandas.read_csv(filename)

    return date_data

def main():
    st.title("Welcome To Face Detection and Motion Detection Application :blush: ")
    #st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About","Contact"]
    choice = st.sidebar.selectbox("Pick Something Fun", activities)

    if choice == "Home":

        st.write("Go to the About section from the sidebar to learn more about it.")


        st.write("**1. Face Detection:**")

        # You can specify more file types below if you want
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:

            image = Image.open(image_file)

            if st.button("Process"):
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img, result_faces = detect(image=image)
                st.image(result_img, use_column_width=True)
                st.success("Found {} faces\n".format(len(result_faces)))

        st.write('\n')
        st.write("**2. Motion Detection:**")
        st.write("This will open your webcam.")
        if st.button("Start"):
            motion()
            date_data = get_data("Time_of_movements.csv")
            st.write("Times of Movement are: ")
            st.write(date_data.head())

            #df = pandas.DataFrame(date_data[:200], columns=['Start', 'End')
            st.write("\nVisual Representation of Movements as a Line Chart is given below: \n")
            st.bar_chart(date_data['Start'])
            #import MotionDetection

    elif choice == "About":
        about()

    elif choice == "Contact":
        contact()


if __name__ == "__main__":
    main()










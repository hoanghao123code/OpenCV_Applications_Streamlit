from time import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

# Trích xuất đặc trưng ảnh
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(
        ii,
        0,
        0,
        ii.shape[0],
        ii.shape[1],
        feature_type=feature_type,
        feature_coord=feature_coord,
    )
    
# Tập dataset 
# Load image
# face = []
# non_face = []

# dir_face = os.listdir('./images/faces_24x24/')
# dir_non_face = os.listdir('./images/non_faces_24x24/')

# for i in range(len(dir_face)):
#     path_face =  './images/faces_24x24/' + dir_face[i]
#     path_non = './images/non_faces_24x24/' + dir_non_face[i]
    
#     # path = 'D:\\OpenCV\\Grabcut\\Grabcut_Streamlit\\images\\faces_24x24\\' + dir_face[i]
#     # print(path, 1, 0.5, 0.5, 23.5, 23.5)
#     # print(path)
#     image_face = cv.imread(path_face)
#     image_face = cv.cvtColor(image_face, cv.COLOR_BGR2GRAY)
#     face.append(image_face)
    
#     image_non = cv.imread(path_non)
#     image_non = cv.cvtColor(image_non, cv.COLOR_BGR2GRAY)
#     non_face.append(image_non)

# face_dataset = [ ]
# for i in range(len(face)):
#     face_dataset.append(face[i])
# for i in range(len(non_face)):
#     face_dataset.append(non_face[i])
# labels = np.array([1] * 440 + [0] * 400)
# face_dataset = np.array(face_dataset)



# print(labels)
# To speed up the example, extract the two types of features only
# Sử dụng 5 feature
# feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

# # Compute the result
# t_start = time()
# X = [extract_feature_image(image, feature_types) for image in face_dataset]

# # Với mỗi x trong X chứa khoảng 190k feature type (bộ lọc)
# X = np.stack(X)
# time_full_feature_comp = time() - t_start

# print(time_full_feature_comp)
# # Label images (100 faces and 100 non-faces)
# # y = np.array([1] * 100 + [0] * 100)
# y = labels
# # Chia train, test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=150, random_state=0, stratify=y
# )

# print(len(X_train), len(y_train))


# Importing OpenCV package 
# import cv2 
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# image_upload = cv.imread('./images/faces_24x24/s1_1.png') 

 
if image_upload is not None:
# Converting image to grayscale
    if not os.path.exists('images'):
        os.makedirs('images')
    image = Image.open(image_upload)
    image.save('images/' + image_upload.name)
    img = cv.imread('images/' + image_upload.name)
    
    if img is not None and len(img.shape) == 3:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

        # Loading the required haar-cascade xml classifier file 
        haar_cascade = cv.CascadeClassifier('./images/haarcascade_frontalface_default.xml') 

        # if os.path.exists('.\\images\\haarcascade_frontalface_default.xml'):
        #     print(1)
        # else:
        #     print(0)

        # Applying the face detection method on the grayscale image 
        # face_rect = 0
        # if gray_img is not None:
        faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9) 

        # print(len(faces_rect))

        # Iterating through rectangles of detected faces 
        # print(face_rect)
        # for (x, y, w, h) in faces_rect: 
        #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 

        st.image(img)
        if len(faces_rect) > 0:
            st.markdown(' <span style = "color:red; font-size:22px;"> Đây là hình ảnh có chứa khuôn mặt</span>', unsafe_allow_html=True)
        else:
            st.markdown(' <span style = "color:red; font-size:22px;"> Đây là hình ảnh không chứa khuôn mặt</span>', unsafe_allow_html=True)
            



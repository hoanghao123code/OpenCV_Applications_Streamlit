from time import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
import xml.etree.ElementTree as ET
import pickle
from PIL import Image



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

cascade_file = 'D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Face_detect\cascade.xml'

tree = ET.parse(cascade_file)

root = tree.getroot()

rect_feature = []
haar_features = []

for feature in root.findall(".//features/_"):
    for rect in feature.findall(".//rects/_"):
        value = rect.text.strip().split()
        x, y, w, h, weight = map(float, value)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        haar_features.append((x, y, w, h, weight))

# Tập dataset 
# Load image
face = []
non_face = []
face_dataset = []
labels = []

dir_face = os.listdir('./images/faces_24x24/')
dir_non_face = os.listdir('./images/non_faces_24x24/')


for i in range(len(dir_face)):
    path_face =  './images/faces_24x24/' + dir_face[i]
    path_non = './images/non_faces_24x24/' + dir_non_face[i]
    
    image_face = cv.imread(path_face)
    # image_face = cv.cvtColor(image_face, cv.COLOR_BGR2GRAY)
    face.append(image_face)
    
    image_non = cv.imread(path_non)
    # image_non = cv.cvtColor(image_non, cv.COLOR_BGR2GRAY)
    non_face.append(image_non)

for i in range(len(face)):
    face_dataset.append(face[i])
for i in range(len(non_face)):
    face_dataset.append(non_face[i])
labels = np.array([1] * 400 + [0] * 400)
face_dataset = np.array(face_dataset)



# Trích xuất đặc trưng ảnh
def extract_feature_image(img):
    image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = cv.resize(image_gray, (24, 24))
    ii = cv.integral(image)
    val = 0
    for rect in haar_features:
        x, y, w, h, weight = rect
        val += weight * (ii[y + h][x + w] + ii[y][x] - ii[y + h][x] - ii[y][x + w])
    return val


# def extract_image_dataset():
#     for i in range(len(face_dataset)):
#         value = extract_feature_image(face_dataset[i])
#         X_train.append(value)
# extract_image_dataset()
# # print(X_train)
# y_train = labels

# pickle_file = 'D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Train_test\X_train.pkl'
# with open(pickle_file, 'wb') as file:
#     pickle.dump(X_train, file)

# pickle_file_y = "D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Train_test\y_train.pkl"
# with open(pickle_file_y, 'wb') as file:
#     pickle.dump(y_train, file)

# model = KNeighborsClassifier(n_neighbors = 5)
# model.fit(X_train, y_train)
X_train = []
y_train = []
# with open('./images/Train_test/X_train.pkl/', 'rb') as file:
#     X_train = pickle.load(file)

# with open('./images/Train_test/y_train.pkl/', 'rb') as file:
#     y_train = pickle.load(file)
if os.path.exists('./images/faces_24x24'):
    print(1)
else:
    print(0)
# def detect_face_Sub_window(image):
#     sz = 24
#     step = 2
#     lst_rect = []
#     for x in range(0, image.shape[0] - sz, step):
#         for y in range(0, image.shape[1] - sz, step):
#             sub_window = image[y + sz, x + sz]
#             feature_sub = extract_feature_image(sub_window)
#             predictions = model.predict(features)
#             if predictions[0] == 1:
#                 lst_rect.append(x, y, sz, sz)
#     return lst_rect
    


# Importing OpenCV package 
# import cv2 

# st.markdown("#### Chọn ảnh bạn cần phát hiện khuôn mặt")
# image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

 
# if image_upload is not None:
#     if not os.path.exists('images'):
#         os.makedirs('images')
#     image = Image.open(image_upload)
#     image.save('images/' + image_upload.name)
#     img = cv.imread('images/' + image_upload.name)
    
#     if img is not None and len(img.shape) == 3:
#         gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

#         # Loading the required haar-cascade xml classifier file 
#         haar_cascade = cv.CascadeClassifier('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Face_detect\cascade.xml') 

#         # Applying the face detection method on the grayscale image 
#         faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
#         # Iterating through rectangles of detected faces 
        
#         features = []
#         for (x, y, w, h) in faces_rect:
#             features.append([x, y, w, h])
#             img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # predictions = Knn.predict(features)
#         # print(type(predictions))
#         st.image(img, channels="BGR")
            



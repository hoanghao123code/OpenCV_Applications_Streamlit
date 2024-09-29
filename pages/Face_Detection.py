from time import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
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

# Trích xuất đặc trưng ảnh
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    max_features = 7000
    ii = integral_image(img)
    features = haar_like_feature(
        ii,
        0,
        0,
        ii.shape[0],
        ii.shape[1],
        feature_type=feature_type,
        feature_coord=feature_coord,
    )
    if max_features is not None and len(features) > max_features:
        features = features[:max_features]
    
    return features
    
# Tập dataset 
# Load image
face = []
non_face = []
face_dataset = []
labels = []

dir_face = os.listdir('./images/faces_24x24/')
dir_non_face = os.listdir('./images/non_faces_24x24/')

for i in range(100):
    path_face =  './images/faces_24x24/' + dir_face[i]
    path_non = './images/non_faces_24x24/' + dir_non_face[i]
    
    image_face = cv.imread(path_face)
    image_face = cv.cvtColor(image_face, cv.COLOR_BGR2GRAY)
    face.append(image_face)
    
    image_non = cv.imread(path_non)
    image_non = cv.cvtColor(image_non, cv.COLOR_BGR2GRAY)
    if len(non_face) < 50:
        non_face.append(image_non)

for i in range(len(face)):
    face_dataset.append(face[i])
for i in range(len(non_face)):
    face_dataset.append(non_face[i])
labels = np.array([1] * 100 + [0] * 50)
face_dataset = np.array(face_dataset)


# To speed up the example, extract the two types of features only
# Sử dụng 5 feature
feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

# Compute the result
t_start = time()
X = [extract_feature_image(image, feature_types) for image in face_dataset]

# Với mỗi x trong X chứa khoảng 190k feature type (bộ lọc)
X = np.stack(X)
time_full_feature_comp = time() - t_start

print(time_full_feature_comp)
# Label images (400 faces and 400 non-faces)
y = labels
# Chia train, test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=130, random_state=0, stratify=y
)

Knn = KNeighborsClassifier(n_neighbors=5)

Knn.fit(X_train, y_train)


# Importing OpenCV package 
# import cv2 

image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

 
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

        # Applying the face detection method on the grayscale image 
        faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
        # Iterating through rectangles of detected faces 
        
        features = []
        for (x, y, w, h) in faces_rect:
            features.append([x, y, w, h])
        predictions = Knn.predict(features)
        print(type(predictions))
        st.image(img, channels="RGB")
        if predictions > 0.5:
            st.markdown(' <span style = "color:red; font-size:22px;"> Đây là hình ảnh có chứa khuôn mặt</span>', unsafe_allow_html=True)
        else:
            st.markdown(' <span style = "color:red; font-size:22px;"> Đây là hình ảnh không chứa khuôn mặt</span>', unsafe_allow_html=True)
            



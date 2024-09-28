# from time import time

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

# from skimage.data import lfw_subset
# from skimage.transform import integral_image
# from skimage.feature import haar_like_feature
# from skimage.feature import haar_like_feature_coord
# from skimage.feature import draw_haar_like_feature

# # Trích xuất đặc trưng ảnh
# def extract_feature_image(img, feature_type, feature_coord=None):
#     """Extract the haar feature for the current image"""
#     ii = integral_image(img)
#     return haar_like_feature(
#         ii,
#         0,
#         0,
#         ii.shape[0],
#         ii.shape[1],
#         feature_type=feature_type,
#         feature_coord=feature_coord,
#     )
    
# # Tập dataset 
# images = lfw_subset()
# # To speed up the example, extract the two types of features only
# # Sử dụng 5 feature
# feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

# # Compute the result
# t_start = time()
# X = [extract_feature_image(img, feature_types) for img in images]

# # Với mỗi x trong X chứa khoảng 190k feature type (bộ lọc)
# X = np.stack(X)
# time_full_feature_comp = time() - t_start

# # Label images (100 faces and 100 non-faces)
# y = np.array([1] * 100 + [0] * 100)

# # Chia train, test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=150, random_state=0, stratify=y
# )


# Importing OpenCV package 
import cv2 
import matplotlib.pyplot as plt
import streamlit as st

# Reading the image 
img = cv2.imread('D:\\OpenCV\\Grabcut\\Grabcut_Streamlit\\images\\images.png') 
# Converting image to grayscale 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# Loading the required haar-cascade xml classifier file 
haar_cascade = cv2.CascadeClassifier('D:\\OpenCV\\WaterShed\\haarcascade_frontalface_default.xml') 
if haar_cascade.empty():
    print("Không thể mở tệp Haar Cascade. Vui lòng kiểm tra đường dẫn.")
else:
    print("Tệp Haar Cascade đã được mở thành công.")
# Applying the face detection method on the grayscale image 
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9) 



# Iterating through rectangles of detected faces 
for (x, y, w, h) in faces_rect: 
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 

plt.imshow(img) 
plt.axis('off')



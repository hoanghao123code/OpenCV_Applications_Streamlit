from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os

from io import BytesIO
from PIL import Image, ImageOps, ImageDraw
from scipy.spatial.distance import cdist
# from rembg import remove
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title('🎈Semantic Keypoint Detection')


sift = cv.SIFT_create()

# surf = cv.xfeatures2d.SURF_create(400)

orb = cv.ORB_create()

def SIFT_result(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    output_image = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, descriptors, output_image

# def SURF_result(image_path):
#     image = cv.imread(image_path)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     keypoints, descriptors = surf.detectAndCompute(gray, None)
#     output_image = cv.drawKeypoints(image, keypoints, None, (255, 0, 0), 4)
#     return output_image

def ORB_result(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    output_image = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    return keypoints, descriptors, output_image


def load_image_dataset(path):
    path_dataset = './images/SIFT_SURF_ORB/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path_image = path_dataset + path + "/" + "images"
    path_label = path_dataset + path + "/" + "points"
    
    lst_all_image = os.listdir(path_image)
    lst_all_label = os.listdir(path_label)
    lst_image = []
    lst_label = []
    for i in range(len(lst_all_image)):
        path_image_cur = path_image + "/" + lst_all_image[i]
        image = cv.imread(path_image_cur)
        lst_image.append(image)
        
        path_label_cur = path_label + "/" + lst_all_label[i]
        label = np.load(path_label_cur)
        lst_label.append(label)
    return lst_image, lst_label

# lst_image, lst_label = load_image_dataset('draw_checkerboard')
# lst_image2, lst_label2 = load_image_dataset('draw_lines')
# lst_image3, lst_label3 = load_image_dataset('draw_multiple_polygons')
# lst_image4, lst_label4 = load_image_dataset('draw_stripes')

# c = st.columns(4)
# i = 0
# image_1 = SIFT_result(lst_image[i + 1])
# image_2 = SIFT_result(lst_image2[i])
# image_3 = SIFT_result(lst_image3[i])
# image_4 = SIFT_result(lst_image4[i])

# c[0].image(image_1, channels="BGR")
# c[1].image(image_2, channels="BGR")
# c[2].image(image_3, channels="BGR")
# c[3].image(image_4, channels="BGR")

def calculate_precision(predicted_keypoints, groundtruth_keypoints, threshold=3):
    true_positive = 0
    predicted_points  = np.array([kp.pt for kp in predicted_keypoints])
    groundtruth_points = np.array(groundtruth_keypoints)

   # Nếu không có predicted hoặc ground truth keypoints
    if len(predicted_points) == 0 or len(groundtruth_points) == 0:
        return 0.0
    
    # Tính khoảng cách Euclidean giữa predicted và groundtruth
    distances = cdist(predicted_points, groundtruth_points, metric='euclidean')
    
    # Kiểm tra các predicted keypoint có keypoint ground truth nào trong phạm vi <= threshold
    matched_predictions = np.min(distances, axis=1) <= threshold
    
    # Tính True Positive (TP) và False Positive (FP)
    TP = np.sum(matched_predictions)  # Số lượng predicted đúng
    FP = len(predicted_keypoints) - TP  # Số lượng predicted sai

    # Tính Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    return precision

def calculate_recall(predicted_keypoints, groundtruth_keypoints, threshold=3):
    true_positive = 0
    predicted_points  = np.array([kp.pt for kp in predicted_keypoints])
    groundtruth_points = np.array(groundtruth_keypoints)

   # Nếu không có predicted hoặc ground truth keypoints
    if len(predicted_points) == 0 or len(groundtruth_points) == 0:
        return 0.0
    
    # Tính khoảng cách Euclidean giữa predicted và groundtruth
    distances = cdist(predicted_points, groundtruth_points, metric='euclidean')
    
    # Kiểm tra các predicted keypoint có keypoint ground truth nào trong phạm vi <= threshold
    matched_predictions = np.min(distances, axis=1) <= threshold
    
    # Tính True Positive (TP) và False Positive (FP)
    TP = np.sum(matched_predictions)  
    FP = len(predicted_keypoints) - TP  

    matched_groundtruth = np.min(distances, axis=0) <= threshold
    
    # False Negative (FN): ground truth không có predicted khớp
    FN = np.sum(~matched_groundtruth)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return recall

def Metric():
    lst_name_dataset = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes', 'gaussian_noise']
    lst_image = []
    lst_label = []
    for i in range(len(lst_name_dataset)):
        image, label = load_image_dataset(lst_name_dataset[i])
        lst_image.append(image)
        lst_label.append(label)
    lst_precision = []
    lst_recall = []
    for i in range(len(lst_image)):
        lst_precision = []
        for j in range(len(lst_image[i])):
            keypoints, _, _ = SIFT_result(lst_image[i][j])
            

def Text_of_App():
    st.header("1. Giới thiệu Synthetic shapes datasets")
    st.write("Dataset Synthetic shapes datasets gồm 9 class bao gồm ảnh và label như **Draw checkerboard, Draw cube, Draw ellipses, "
            + "Draw lines, Draw multiple polygon, Draw polygon, Draw star, Draw stripes và Gaussian noise**")
    st.write("**Một số ảnh trong Dataset**")
    path_dataset = './images/SIFT_SURF_ORB/image_dataset_synthetic_shapes.PNG'
    image_dataset = cv.imread(path_dataset)
    st.image(image_dataset,channels="BGR")
    st.header("2. Thuật toán SIFT")
    st.write("Duới đây là kết quả của một số ảnh khi áp dụng thuật toán **SIFT**")
    st.image('./images/SIFT_SURF_ORB/result_of_SIFT.PNG', channels="BGR")

    st.header("3. Thuật toán ORB")
    st.write("Duới đây là kết quả của một số ảnh khi áp dụng thuật toán **ORB**")
    st.image('./images/SIFT_SURF_ORB/result_of_ORB.PNG', channels="BGR")

    st.header("4. Đánh giá")
    st.write("Tiến hành đánh giá trên 2 độ đo **Precision** và **Recall** khi áp dụng **SIFT và ORB**")
    st.markdown("**Keypoint** đó được cho là đúng nếu khoảng cách của **Keypoint** đó so với khoảng các của **Keypoint** thực tế là dưới 3 pixel")
    st.markdown("**4.1 Độ đo Precision**")

    st.image('./images/SIFT_SURF_ORB/precision_metric.png', width=400, channels="BGR")
    st.markdown("**4.2 Độ đo Recall**")
    st.image('./images/SIFT_SURF_ORB/recall_metric.png', width=400, channels="BGR")

    
def App():
    Text_of_App()
    # Metric()
App()
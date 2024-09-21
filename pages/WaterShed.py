from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os
import pandas as pd

from io import BytesIO
from PIL import Image
# from rembg import remove
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title("🎈Hoang Hao WaterShed App")

def IoU(mask_pred, mask_gt):
    # mask_pred = [mask_pred > 0].astype(np.uint8)
    # mask_gt = [mask_gt > 0].astype(np.uint8)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0.0:
        return 0.0
    iou = intersection / union
    return iou

path = ['D:\\OpenCV\\WaterShed\\1xemay278.jpg', 'D:\\OpenCV\\WaterShed\\1xemay544.jpg',
        'D:\\OpenCV\\WaterShed\\1xemay645.jpg', 'D:\\OpenCV\\WaterShed\\1xemay1458.jpg']
path_gt = ['D:\\OpenCV\\images\\1xemay278.png', 'D:\\OpenCV\\images\\1xemay544.png',
           'D:\\OpenCV\\images\\1xemay645.png', 'D:\\OpenCV\\images\\1xemay1458.png']

list_image = ["Ảnh 1xemay278", "Ảnh 1xemay544", "Ảnh 1xemay645", "Ảnh 1xemay1458"]

def marker(path, kernels, ratio_thresh):
    img_path = path
    img_bgr = cv.imread(img_path)
    # img_blur = cv.medianBlur(src = img_bgr, ksize = 3)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    
    ret, img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones(kernels, np.uint8)
    opening = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel = kernel, iterations = 2)

    # Dist transform
    dist_transform = cv.distanceTransform(src = opening, distanceType = cv.DIST_L2, maskSize = 5)
    
    # Sure foreground
    ret, sure_foreground = cv.threshold(src = dist_transform, thresh = ratio_thresh * np.max(dist_transform), maxval = 255, type = 0)
    
    sure_foreground = np.uint8(sure_foreground)

    # Sure background
    sure_background = cv.dilate(src = opening, kernel = kernel, iterations = 3)

    # Unknown
    unknown = cv.subtract(sure_background, sure_foreground)

    # Markers
    ret, markers = cv.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(image = img_bgr, markers = markers)
    return markers

def img_training(idx1, idx2):
    col1, col2 = st.columns(2)
    
    img_ori1 = cv.imread(path[idx1])
    img_ori2 = cv.imread(path[idx2])
    col1.markdown("### Ảnh gốc")
    
    img_gt1 = cv.imread(path_gt[idx1], cv.IMREAD_GRAYSCALE)
    img_gt2 = cv.imread(path_gt[idx2], cv.IMREAD_GRAYSCALE)
    
    if img_gt1 is not None and img_gt2 is not None:
        mask_gt1 = img_gt1.copy()
        img_gt1[mask_gt1 == 85] = 255
        img_gt1[mask_gt1 != 85] = 0
        
        mask_gt2 = img_gt2.copy()
        img_gt2[mask_gt2 == 85] = 255
        img_gt2[mask_gt2 != 85] = 0
        col1.image(img_ori1)
        col1.markdown("#### " + list_image[idx1])
        col1.image(img_ori2)
        col1.markdown("#### " + list_image[idx2])
        
        col2.markdown("### Ảnh ground truth")
        col2.image(img_gt1)
        col2.markdown("#### " + list_image[idx1])
        col2.image(img_gt2)
        col2.markdown("#### " + list_image[idx2])
    

def calc():
    kernels = [(3, 3), (5, 5), (7, 7)]
    ratio_thresh = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    list_IoU = []
    for kernel in kernels:
        for ratio in ratio_thresh:
            for i in range(2):
                # Ground truth
                img_gt = cv.imread(path_gt[i], cv.IMREAD_GRAYSCALE)
                if img_gt is not None:
                    mask_gt = img_gt.copy()
                    img_gt[mask_gt == 85] = 255
                    img_gt[mask_gt != 85] = 0
                    # st.image(img_gt, channels = 'gray')
                    
                    markers = marker(path[i], kernel, ratio)
                    num_labels = np.unique(markers)
                    img_bg = cv.imread(path[i], cv.IMREAD_GRAYSCALE)
                    img_bg[img_bg != 0] = 0
                    for labels in num_labels:
                        if labels == -1:
                            continue
                        id = np.where(markers == labels)
                        x_min = min(id[0])
                        x_max = max(id[0])
                        
                        y_min = min(id[1])
                        y_max = max(id[1])
                        
                        height = (x_max - x_min) / img_bg.shape[0]
                        width = (y_max - y_min) / img_bg.shape[1]
                        if height >= 0.3 and height <= 0.6 and width >= 0.0 and width <= 0.3:
                            img_bg[markers == labels] = 255
                    # st.image(img_bg)
                    list_IoU.append(IoU(img_bg, img_gt))
    list_IoU = np.array(list_IoU)
    kernel_num = np.array([3, 5, 7, 9, 11, 13, 15])
    ratio_num = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    IoU_kernel3 = list_IoU[:7]
    IoU_kernel5 = list_IoU[7:14]
    IoU_kernel7 = list_IoU[14:21]
   
    
    img2_IoU_kernel3 = list_IoU[21:28]
    img2_IoU_kernel5 = list_IoU[28:35]
    img2_IoU_kernel7 = list_IoU[35:42]
    
    c1, c2 = st.columns(2)
    fig1, ax = plt.subplots()
    ax.plot(ratio_num, IoU_kernel3, label='Kernel=3')
    ax.plot(ratio_num, IoU_kernel5, label='Kernel=5')
    ax.plot(ratio_num, IoU_kernel7, label='Kernel=7')

    ax.set_xlabel('Ratio')
    ax.set_ylabel('IoU')    
    ax.set_title('Biểu đồ IoU theo Ratio thresh và Kernel của ảnh 1xemay278')  
    ax.legend()

    fig2, ax2= plt.subplots()
    ax2.plot(ratio_num, img2_IoU_kernel3, label='Kernel=3')
    ax2.plot(ratio_num, img2_IoU_kernel5, label='Kernel=5')
    ax2.plot(ratio_num, img2_IoU_kernel7, label='Kernel=7')

    ax2.set_xlabel('Ratio')
    ax2.set_ylabel('IoU')    
    ax2.set_title('Biểu đồ IoU theo Ratio thresh và Kernel của ảnh 1xemay544')  
    ax2.legend()

    c1.pyplot(fig1)
    c2.pyplot(fig2)

def run():
    st.markdown("## 1. Tập Train và Test")
    st.markdown("### 1.1 Tập Train")
    img_training(0, 1)
    st.markdown("### 1.2 Tập Test")
    img_training(2, 3)
    st.markdown("## 2. Lựa chọn các tham số phù hợp trong quá trình Train")
    st.markdown("#### kernel = [(3, 3), (5, 5), (7, 7)]")
    st.markdown("#### Hệ số nhân sử dụng trong tính toán ngưỡng: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]")
    if st.button("# Click vào đây để tiến hành huấn luyện"):
        with st.spinner("Đang xử lí..."):
            calc()
run()


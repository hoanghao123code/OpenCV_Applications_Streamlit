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

path = ['./images/1xemay278.jpg', './images/1xemay544.jpg', 
        './images/1xemay645.jpg', './images/1xemay1458.jpg']

path_gt = ['./images/1xemay278.png', './images/1xemay544.png', 
        './images/1xemay645.png', './images/1xemay1458.png']

name = ['1xemay278.jpg', '1xemay544.jpg', '1xemay645.jpg', '1xemay1458.jpg']

list_image = ["Ảnh 1xemay278", "Ảnh 1xemay544", "Ảnh 1xemay645", "Ảnh 1xemay1458"]

path_IoU_img = './images/image_IoU.png'
image_IoU = cv.imread(path_IoU_img)

list_images = []
list_image_gt = []

def load_image():
    for i in range(4):
        list_images.append(cv.imread(path[i]))
        list_image_gt.append(cv.imread(path_gt[i], cv.IMREAD_GRAYSCALE))
load_image()


def marker(idx_image, kernels, ratio_thresh):
   
    img_bgr = list_images[idx_image]
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
    
    img_ori1 = list_images[idx1]
    img_ori2 = list_images[idx2]
    col1.markdown("### Ảnh gốc")

    
    img_gt1 = list_image_gt[idx1]
    img_gt2 = list_image_gt[idx2]
    # Ảnh nhị phân ground truth
   
    mask_gt1 = 0 
    if img_gt1 is not None:
        mask_gt1 = img_gt1.copy()
    img_gt1[mask_gt1 == 85] = 255
    img_gt1[mask_gt1 != 85] = 0
    
    mask_gt2 = 0
    if img_gt2 is not None:
        mask_gt2 = img_gt2.copy()
    img_gt2[mask_gt2 == 85] = 255
    img_gt2[mask_gt2 != 85] = 0
    
    # In ảnh
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
    
    # Các tham số
    kernels = [(3, 3), (5, 5), (7, 7)]
    ratio_thresh = np.arange(0.0, 1.0, 0.04)
    list_IoU_img1 = []
    list_IoU_img2 = []
    
    ans = []
    #Thử với các tham số
    for kernel in kernels:
        for ratio in ratio_thresh:
            for i in range(2):
                # Ground truth
                img_gt = cv.imread(path_gt[i], cv.IMREAD_GRAYSCALE)
                mast_gt = 0
                if img_gt is not None:
                    mask_gt = img_gt.copy()
                img_gt[mask_gt == 85] = 255
                img_gt[mask_gt != 85] = 0
                
                # Marker
                markers = marker(i, kernel, ratio)
                num_labels = np.unique(markers)
                img_bg = cv.imread(path[i], cv.IMREAD_GRAYSCALE)
                img_bg[img_bg != 0] = 0
                
                # Tô màu cho từng kí tự của ảnh
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
                ans.append(img_bg)
                if i == 0:
                    list_IoU_img1.append(IoU(img_bg, img_gt))
                else:
                    list_IoU_img2.append(IoU(img_bg, img_gt))
    # print(list_IoU)
    list_IoU_img1 = np.array(list_IoU_img1)
    list_IoU_img2 = np.array(list_IoU_img2)
    
    IoU_kernel3 = list_IoU_img1[:25]
    IoU_kernel5 = list_IoU_img1[25:50]
    IoU_kernel7 = list_IoU_img1[50:100]
   
    img2_IoU_kernel3 = list_IoU_img2[:25]
    img2_IoU_kernel5 = list_IoU_img2[25:50]
    img2_IoU_kernel7 = list_IoU_img2[50:100]
    
    fig1, ax = plt.subplots()
    ax.plot(ratio_thresh, IoU_kernel3, label='Kernel=3')
    ax.plot(ratio_thresh, IoU_kernel5, label='Kernel=5')
    ax.plot(ratio_thresh, IoU_kernel7, label='Kernel=7')

    ax.set_xlabel('Ratio')
    ax.set_ylabel('IoU')    
    ax.set_title('Biểu đồ IoU theo Ratio thresh và Kernel của ảnh 1xemay278')  
    ax.legend()

    fig2, ax2= plt.subplots()
    ax2.plot(ratio_thresh, img2_IoU_kernel3, label='Kernel=3')
    ax2.plot(ratio_thresh, img2_IoU_kernel5, label='Kernel=5')
    ax2.plot(ratio_thresh, img2_IoU_kernel7, label='Kernel=7')

    ax2.set_xlabel('Ratio')
    ax2.set_ylabel('IoU')    
    ax2.set_title('Biểu đồ IoU theo Ratio thresh và Kernel của ảnh 1xemay544')  
    ax2.legend()
    c1, c2 = st.columns(2)
    c1.pyplot(fig1)
    c2.pyplot(fig2)
    
    # Tìm tham số tốt nhất
    
    best_3 = IoU_kernel3 + img2_IoU_kernel3
    best_5 = IoU_kernel5 + img2_IoU_kernel5
    best_7 = IoU_kernel7 + img2_IoU_kernel7
    max_IoU = max(max(best_3), max(best_5), max(best_7))
    best = 0
    kernel_best = (3, 3)
    if max_IoU == max(best_3):
        best = best_3
        kernel_best = (3, 3)
    if max_IoU == max(best_5):
        best = best_5
        kernel_best = (5, 5)
    
    if max_IoU == max(best_7):
        best = best_7
        kernel_best = (7, 7)
        
    id = np.where(best == max_IoU)
    st.markdown("### * Tham số tốt nhất là:")
    st.markdown(f"####  - Kernel = **{kernel_best}** ")
    st.markdown(f"####  - Hệ số nhân sử dụng trong tính toán ngưỡng: **{ratio_thresh[id[0][0]]}**")
    
    st.markdown(f"### * Kết quả khi áp dụng các chỉ số vừa tìm được vào **{list_image[0]}** và **{list_image[1]}** là:")
    colm1, colm2 = st.columns(2)
    colm1.image(ans[0])
    colm1.markdown(f"#### **{list_image[0]}**")
    
    colm2.image(ans[1])
    colm2.markdown(f"#### **{list_image[1]}**")
    
    st.markdown("## * Áp dụng các chỉ số tốt nhất vào tập Test")
    
    best_kernel = kernel_best[0]
    best_ratio = id[0][0]
    
    ret = []
    watershed_res = []
    for i in range(2, 4, 1):
    # Ground truth
        img_gt = cv.imread(path_gt[i], cv.IMREAD_GRAYSCALE)
        mask_gt = img_gt.copy()
        img_gt[mask_gt == 85] = 255
        img_gt[mask_gt != 85] = 0
        
        # Marker
        markers = marker(i, best_kernel, best_ratio)
        num_labels = np.unique(markers)
        img_bg = cv.imread(path[i], cv.IMREAD_GRAYSCALE)
        img_bg[img_bg != 0] = 0
        
        # Tô màu cho từng kí tự của ảnh
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
        watershed_res.append(img_bg)
        ret.append(IoU(img_bg, img_gt))
    st.markdown(f"### * Kết quả khi áp dụng các chỉ số vừa tìm được vào **{list_image[2]}** và **{list_image[3]}** là:")
    cc1, cc2 = st.columns(2)
    cc1.image(watershed_res[0])
    cc1.markdown(f"#### - IoU của **{list_image[2]}** là: **{ret[0]:.2f}**")
    cc2.image(watershed_res[1])
    cc2.markdown(f"#### - IoU của **{list_image[3]}** là: **{ret[1]:.2f}**")

    
def run():
    st.markdown("## 1. Tập Train và Test")
    st.markdown("### 1.1 Tập Train")
    img_training(0, 1)
    st.markdown("### 1.2 Tập Test")
    img_training(2, 3)
    st.markdown("## 2. Lựa chọn các tham số phù hợp trong quá trình Train với thuật toán WaterShed Segmentation")
    st.markdown("### * Các tham số được sử dụng:")
    st.markdown("####  - Kernel = [(3, 3), (5, 5), (7, 7)]")
    st.markdown("####  - Hệ số nhân sử dụng trong tính toán ngưỡng: 0.00, 0.04, 0.08,... 1.0")
    st.markdown("### * Độ đo: IoU")
    st.image(image_IoU, width=350)
    if st.button("# Click vào đây để tiến hành huấn luyện"):
        with st.spinner("Đang xử lí..."):
            calc()
            

if len(list_images) > 0:
    run()


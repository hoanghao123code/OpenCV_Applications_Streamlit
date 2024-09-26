from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os
import pandas as pd
import time
import asyncio
import threading

from io import BytesIO
from PIL import Image
# from rembg import remove
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.title("🎈Hoang Hao WaterShed Segmentation App")

def IoU(mask_pred, mask_gt):
    # mask_pred = [mask_pred > 0].astype(np.uint8)
    # mask_gt = [mask_gt > 0].astype(np.uint8)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0.0:
        return 0.0
    iou = intersection / union
    return iou

def Dice_coefficient(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    # intersection = np.sum(mask_pred * mask_gt)
    sum_pred = mask_pred.sum()
    sum_gt = mask_gt.sum()
    if (sum_pred + sum_gt == 0):
        return 1.0
    dice = (2.0 * intersection) / (sum_pred + sum_gt)
    return dice

path = ['./images/1xemay278.jpg', './images/1xemay544.jpg', 
        './images/1xemay645.jpg', './images/1xemay1458.jpg']

path_gt = ['./images/1xemay278.png', './images/1xemay544.png', 
        './images/1xemay645.png', './images/1xemay1458.png']

name = ['1xemay278.jpg', '1xemay544.jpg', '1xemay645.jpg', '1xemay1458.jpg']

list_image = ["Ảnh 1xemay278", "Ảnh 1xemay544", "Ảnh 1xemay645", "Ảnh 1xemay1458"]

path_IoU_img = './images/image_IoU.png'
image_IoU = Image.open(path_IoU_img)

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
    col1.markdown('   <p style="text-indent: 130px;"> <span style = "color:red; font-size:22px;"> Ảnh gốc</span>', unsafe_allow_html=True)
    # col1.markdown("### Ảnh gốc")

    
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
    col1.markdown(f'   <p style="text-indent: 110px;"> <span style = "color:red; font-size:22px;"> {list_image[idx1]}</span>', unsafe_allow_html=True)
    
    col1.image(img_ori2)
    col1.markdown(f'   <p style="text-indent: 110px;"> <span style = "color:red; font-size:22px;"> {list_image[idx2]} </span>', unsafe_allow_html=True)
    
    col2.markdown('   <p style="text-indent: 100px;"> <span style = "color:red; font-size:22px;"> Ảnh Ground truth</span>', unsafe_allow_html=True)
    col2.image(img_gt1)
    col2.markdown(f'   <p style="text-indent: 100px;"> <span style = "color:red; font-size:22px;">{list_image[idx1]}</span>', unsafe_allow_html=True)
    
    col2.image(img_gt2)
    col2.markdown(f'   <p style="text-indent: 100px;"> <span style = "color:red; font-size:22px;"> {list_image[idx2]}</span>', unsafe_allow_html=True)
    

def image_with_other_thesh(i, kernels, thresh, num_labels):
    lst_pred = []
    img_gt = cv.imread(path_gt[i], cv.IMREAD_GRAYSCALE)
    mask_gt = img_gt.copy()
    img_gt[mask_gt == 85] = 255
    img_gt[mask_gt != 85] = 0
    for kernel in kernels:
    # Ground truth
        # Marker
        markers = marker(i, kernel, thresh)
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
        lst_pred.append(img_bg)
    cot1, cot2, cot3, cot4, cot5 = st.columns(5)
    cot1.image(img_bg, caption="Ảnh 1xemay278 trong tập train")
    cot2.image(lst_pred[0], caption="Groundtruth của ảnh")
    cot3.image(lst_pred[1], caption="Groundtruth của ảnh")
    cot4.image(lst_pred[2], caption="Groundtruth của ảnh")
    cot5.image(lst_pred[3], caption="Groundtruth của ảnh")

def get_with_Kernel(lst, x):
    x1 = lst[ :x]
    x2 = lst[x : 2 * x]
    x3 = lst[2 * x : 3 * x]
    return x1, x2, x3

def Plot_IoU(IoU_1, IoU_2, thresh):
    # Plot theo IoU
    x1, x2, x3 = get_with_Kernel(IoU_1, 20)
    y1, y2, y3 = get_with_Kernel(IoU_2, 20)
   
    img2_IoU_kernel3 = IoU_2[:20]
    img2_IoU_kernel5 = IoU_2[20:40]
    img2_IoU_kernel7 = IoU_2[40:60]
    
    fig1, ax = plt.subplots()
    ax.plot(thresh, x1, label='Kernel = (3, 3)')
    ax.plot(thresh, x2, label='Kernel = (5, 5)')
    ax.plot(thresh, x3, label='Kernel = (7, 7)')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('IoU')    
    ax.set_title('Biểu đồ IoU theo Threshold và Kernel của ảnh 1xemay278')  
    ax.legend()

    fig2, ax2= plt.subplots()
    ax2.plot(thresh, y1, label='Kernel = (3, 3)')
    ax2.plot(thresh, y2, label='Kernel = (5, 5)')
    ax2.plot(thresh, y3, label='Kernel = (7, 7)')

    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('IoU')    
    ax2.set_title('Biểu đồ IoU theo Threshold và Kernel của ảnh 1xemay544')  
    ax2.legend()
    c1, c2 = st.columns(2)
    c1.pyplot(fig1)
    c2.pyplot(fig2)

def Plot_Dice(lst_dice_1, lst_dice_2, thresh):
    st.markdown("#### * Độ đo: Dice Coefficient")
    image_dice = Image.open('./images/dice_coefficient.png')
    st.image(image_dice)
    
    #Plot theo Dice coefficient
    lst_dice_1 = np.array(lst_dice_1)
    lst_dice_2 = np.array(lst_dice_2)
    
    x1, x2, x3 = get_with_Kernel(lst_dice_1, 20)
    y1, y2, y3 = get_with_Kernel(lst_dice_2, 20)
    
    fi1, axis1 = plt.subplots()
    axis1.plot(thresh, x1, label='Kernel = (3, 3)')
    axis1.plot(thresh, x2, label='Kernel = (5, 5)')
    axis1.plot(thresh, x3, label='Kernel = (7, 7)')
    axis1.set_xlabel('Threshold')
    axis1.set_ylabel('Dice coefficient')
    axis1.set_title('Biểu đồ Dice theo Threshold và Kernel của ảnh 1xemay278')
    axis1.legend()
    
    
    fi2, axis2 = plt.subplots()
    axis2.plot(thresh, y1, label='Kernel = (3, 3)')
    axis2.plot(thresh, y2, label='Kernel = (5, 5)')
    axis2.plot(thresh, y3, label='Kernel = (7, 7)')
    axis2.set_xlabel('Threshold')
    axis2.set_ylabel('Dice coefficient')
    axis2.set_title('Biểu đồ Dice theo Threshold và Kernel của ảnh 1xemay544')
    axis2.legend()
    
    coll1, coll2 = st.columns(2)
    coll1.pyplot(fi1)
    coll2.pyplot(fi2)


def best_para(lst_IoU_1, lst_IoU_2, lst_dice_1, lst_dice_2, lst_thresh):
    
    # Lấy độ đo IoU của Kernel 3, 5, 7
    x1, x2, x3 = get_with_Kernel(lst_IoU_1, 20)
    y1, y2, y3 = get_with_Kernel(lst_IoU_2, 20)
    
    # Lấy độ đo Dice của Kernel 3, 5, 7
    d_x1, d_x2, d_x3 = get_with_Kernel(lst_dice_1, 20)
    d_y1, d_y2, d_y3 = get_with_Kernel(lst_dice_2, 20)
    
    # Tổng độ đo của từng kernel
    sum_K3 = x1 + y1 + d_x1 + d_y1
    sum_K5 = x2 + y2 + d_x2 + d_y2
    sum_K7 = x3 + y3 + d_x3 + d_y3
    
    # Lấy tổng độ đo lớn nhất của 3 kernel
    max_metrics = max(max(sum_K3), max(sum_K5), max(sum_K7))
    
    # Tìm kernel và thresh tốt nhất 
    best = 0
    kernel_best = (3, 3)
    if max_metrics == max(sum_K3):
        best = sum_K3
        kernel_best = (3, 3)
    if max_metrics == max(sum_K5):
        best = sum_K5
        kernel_best = (5, 5)
    
    if max_metrics == max(sum_K7):
        best = sum_K7
        kernel_best = (7, 7)
        
    id = np.where(best == max_metrics)
    st.markdown("##### * Tham số tốt nhất là:")
    st.markdown(f"######  - Kernel = {kernel_best}")
    st.markdown(f"######  - Threshold = {lst_thresh[id[0][0]]}")
    return kernel_best, lst_thresh[id[0][0]]

def Apply_best_Para(best_kernel, best_thresh):
    ret = []
    watershed_res = []
    for i in range(2, 4, 1):
    # Ground truth
        img_gt = cv.imread(path_gt[i], cv.IMREAD_GRAYSCALE)
        mask_gt = img_gt.copy()
        img_gt[mask_gt == 85] = 255
        img_gt[mask_gt != 85] = 0
        
        # Marker
        markers = marker(i, best_kernel, best_thresh)
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
                
        img_pred = 0
        img_ground = 0
        if img_bg is not None:
            img_pred = img_bg.copy()
        if img_gt is not None:
            img_ground = img_gt.copy()
        img_ground[img_ground == 255] = 1
        img_pred[img_pred == 255] = 1
        watershed_res.append(img_bg)
        ret.append(IoU(img_pred, img_ground))
    st.markdown(f"#### 2.2 Kết quả khi áp dụng các chỉ số vừa tìm được vào tập Test")
    cc1, cc2 = st.columns(2)
    cc1.image(watershed_res[0])
    # col1.markdown(f'   <p style="text-indent: 110px;"> <span style = "color:red; font-size:22px;"> {list_image[idx1]}</span>', unsafe_allow_html=True)
    
    cc1.markdown(f'   <p style="text-indent: 110px;"> <span style = "color:red; font-size:22px;"> IoU = {ret[0]:.2f} </span>', unsafe_allow_html=True)
    cc2.image(watershed_res[1])
    cc2.markdown(f'   <p style="text-indent: 130px;"> <span style = "color:red; font-size:22px;"> IoU = {ret[1]:.2f} </span>', unsafe_allow_html=True)

def calc():   
    # Các tham số
    kernels = [(3, 3), (5, 5), (7, 7)]
    lst_thresh = np.arange(0.0, 0.4, 0.02)
    lst_IoU_1 = []
    lst_IoU_2 = []
    
    lst_dice_1 = []
    lst_dice_2 = []
    ans = []
    #Thử với các tham số
    for kernel in kernels:
        for ratio in lst_thresh:
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
                img_pred = 0
                img_ground = 0
                if img_bg is not None:
                    img_pred = img_bg.copy()
                if img_gt is not None:
                    img_ground = img_gt.copy()
                img_ground[img_ground == 255] = 1
                img_pred[img_pred == 255] = 1
                if i == 0:
                    lst_IoU_1.append(IoU(img_pred, img_ground))
                    lst_dice_1.append(Dice_coefficient(img_pred, img_ground))
                else:
                    lst_IoU_2.append(IoU(img_pred, img_ground))
                    lst_dice_2.append(Dice_coefficient(img_pred, img_ground))
                    
    lst_IoU_1 = np.array(lst_IoU_1)
    lst_IoU_2 = np.array(lst_IoU_2)
    
    Plot_IoU(lst_IoU_1, lst_IoU_2, lst_thresh)
    Plot_Dice(lst_dice_1, lst_dice_2, lst_thresh)
    # Tìm tham số tốt nhất
    
    best_kernel, best_thresh = best_para(lst_IoU_1, lst_IoU_2, lst_dice_1, lst_dice_2, lst_thresh)
    Apply_best_Para(best_kernel, best_thresh)
    return "Xong"

# async def run_progress(duration):
#     progress_bar = st.progress(0)
#     for i in range(100):
#         await asyncio.sleep(duration / 100)
#         progress_bar.progress(i + 1)

# async def process():
#     # calc_thread = threading.Thread(target=calc)
#     # calc_thread.start()
#     # run_progress(24)
#     # calc_thread.join()
#     await asyncio.gather(
#         calc(),
#         run_progress(24)
#     )
def run():
    st.markdown("### 1. Tập Train và Test")
    st.markdown("#### 1.1 Tập Train")
    img_training(0, 1)
    st.markdown("#### 1.2 Tập Test")
    img_training(2, 3)
    st.markdown("### 2. Xác định các tham số tối ưu")
    st.markdown("##### Các tham số được sử dụng")
    st.write("- Kernel = [(3, 3), (5, 5), (7, 7)]" )
    st.write("- Threshold = [0.00, 0.02, ..., 0.4]")
    st.markdown("#### * Độ đo: IoU")
    st.image(image_IoU, width=350)
    # asyncio.run(process())
    calc()

if len(list_images) > 0:
    run()


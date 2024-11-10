import streamlit as st
import firebase_admin
import json, toml
from firebase_admin import credentials
from google.cloud import firestore, storage
from PIL import Image, ImageOps
from google.cloud.firestore import FieldFilter as fil
from google.cloud.firestore_v1.vector import Vector
from io import BytesIO

import cv2 as cv
import pandas as pd
import numpy as np
import unicodedata
import re
import tempfile
import sys
import os
import argparse
import time
import random
import requests
import pickle

sys.path.append("./services") 
from semantic_keypoint_detection.Superpoint import SuperPointNet, SuperPointFrontend

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🎈Instance Search App")

# Khởi tạo Firestore Client bằng credentials từ file JSON
db = firestore.Client.from_service_account_info(st.secrets)


bucket = storage.Client.from_service_account_info(st.secrets).get_bucket('face-detection-2024.appspot.com')

def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    if desc1 is None or desc2 is None:
        return None, None, None
    matches = bf.match(desc1, desc2)
    # print(type(desc1))
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv.findHomography(matched_pts1,
                                    matched_pts2,
                                    cv.RANSAC)
    inliers = inliers.flatten()
    return H, inliers

def convert_pts_to_keypoints(pts):
    keypoints = []
    for i in range(pts.shape[1]):
        # Tạo cv2.KeyPoint từ tọa độ (x, y) trong pts
        kp = cv.KeyPoint(x=pts[0, i], y=pts[1, i], size=1)
        keypoints.append(kp)
    return keypoints
def extract_superpoint_keypoint_and_descriptor(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = img_gray.astype('float32')/255.0
    pts, desc, heatmap = fe.run(img_gray)
    return pts, desc, heatmap

def compare_and_draw_superpoint_match(pts1, desc_1, pts2, desc_2):
    # image_kp1 = image_1.copy()
    # image_gray = cv.cvtColor(image_kp1, cv.COLOR_BGR2GRAY)
    # image_gray = image_gray.astype('float32') / 255.0
    # pts1, desc_1, _ = extract_superpoint_keypoint_and_descriptor(image_1)
    # pts2, desc_2, _ = extract_superpoint_keypoint_and_descriptor(image_2)
    if desc_2 is None:
        return 0.0
    kp1 = convert_pts_to_keypoints(pts1)
    kp2 = convert_pts_to_keypoints(pts2)
    desc1 = desc_1.T
    desc2 = desc_2.T
    # print(len(kp1), len(kp2))
    # print(desc_1.shape)
    # print(desc_2.shape)
    m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
    if m_kp1 is None and m_kp2 is None and matches is None:
        return 0.0
    if len(m_kp1) < 4 or len(m_kp2) < 4:
        return 0.0
    H, inliers = compute_homography(m_kp1, m_kp2)
    # Draw SuperPoint matches
    matches = np.array(matches)[inliers.astype(bool)].tolist()
    return len(matches)

fe = SuperPointFrontend(weights_path = './services/semantic_keypoint_detection/superpoint_v1.pth', 
                            nms_dist = 4, 
                            conf_thresh = 0.015,
                            nn_thresh = 0.7,
                            cuda = False)


def get_kp_and_desc():
    path = "D:\\OpenCV\\val2017\\val2017"
    lst_dts = os.listdir(path)
    kp_and_desc = []
    m_len = 30
    for i in range(m_len * 17, m_len * 18, 1):
        image = cv.imread(path + "/" + lst_dts[i])
        kp, desc, _ = extract_superpoint_keypoint_and_descriptor(image)
    #     kp = convert_pts_to_keypoints(kp_)
    #     matches_img, _ = compare_and_draw_superpoint_match(image, image)
        kp_and_desc.append((kp, desc, image))

    file = './data_processed/Semantic_Keypoint_Detection/kp_and_desc18.pkl'

    with open(file, 'wb') as file:
        pickle.dump(kp_and_desc, file)

def scale_image(image):
    max_size = 250
    w = min(image.shape[1], max_size)
    h = w * image.shape[0] // image.shape[1]
    image = cv.resize(image, (w, h))
    return image

def process():
    kp_and_desc = []
    for i in range(1, 11, 1):
        file_path = "./data_processed/Semantic_Keypoint_Detection/kp_and_desc" + str(i)
        file_path = file_path + ".pkl"
        kp_desc = []
        with open(file_path, "rb") as file:
            kp_desc = pickle.load(file)
        for (kp, desc, image) in kp_desc:
            kp_and_desc.append((kp, desc, image))
    image_upload = st.file_uploader("Tải ảnh truy vấn", type=["png", "jpg", "jpeg"])
    k = st.slider("Chọn số lượng ảnh tương tự cần hiển thị", 1, 100, 5)
    if st.button(":material/search: Tìm kiếm"):
        results = []
        if image_upload is not None:
            status_text = st.empty()
            progress_bar = st.progress(0)
            image = Image.open(image_upload)
            image_np = np.array(image)
            kp2, desc_2, _ = extract_superpoint_keypoint_and_descriptor(image_np)
            for (kp1, desc_1, image_cur) in kp_and_desc:
                num_of_matches = compare_and_draw_superpoint_match(kp1, desc_1, kp2, desc_2)
                results.append((num_of_matches, image_cur))
                progress_bar.progress(len(results) / len(kp_and_desc))
                status_text.text(f"Đang tìm kiếm: {int((100 * len(results)) / len(kp_and_desc))}%")
            results = sorted(results, key=lambda x:x[0], reverse=True)
            status_text.text("Hoàn thành!")
            st.markdown("**Ảnh truy vấn**")
            st.image(scale_image(image_np))
            st.markdown("**Ảnh kết quả**")
            c = st.columns(5)
            for i in range(k):
                c[i % 5].image(scale_image(results[i][1]), channels="BGR")
        else:
            st.warning("Vui lòng chọn ảnh cần tìm kiếm!")
def Dataset_and_Process():
    st.header("1. Giới thiệu Dataset")
    st.markdown(
                """
                - Dataset bao gồm $500$ ảnh đầu tiên của tập **val2017** của dataset **COCO**
                bao gồm nhiều đối tượng khác nhau, từ người, động vật đến các đồ vật như xe cộ, thiết bị gia dụng
                """)
    path = "./images/Semantic_Keypoint_Detection/example_dataset_COCO/"
    lst_name = os.listdir(path)
    c = st.columns(5)
    for i in range(len(lst_name)):
        path_image = path + lst_name[i]
        image = cv.imread(path_image)
        c[i % 5].image(image, channels="BGR")
    st.header("2. Tìm kiếm ảnh")
    process()
    
def App():
    Dataset_and_Process()
App()
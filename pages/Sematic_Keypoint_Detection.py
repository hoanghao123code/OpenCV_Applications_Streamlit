from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

sys.path.append("./services") 
from semantic_keypoint_detection.Superpoint import SuperPointNet, SuperPointFrontend
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


orb = cv.ORB_create()


def SIFT_result(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    output_image = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    return keypoints, descriptors, output_image

def ORB_result(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    output_image = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    return keypoints, descriptors, output_image


def load_image_dataset(path):
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path_image = path_dataset + path + "/" + "images"
    path_label = path_dataset + path + "/" + "points"
    lst_all_image = os.listdir(path_image)
    lst_all_label = os.listdir(path_label)
    lst_image = []
    lst_label = []
    for i in range(len(lst_all_image)):
        path_image_cur = path_image + "/" + str(i) + ".png"
        
        image = cv.imread(path_image_cur)
        lst_image.append(image)
        
        path_label_cur = path_label + "/" + str(i) + ".npy"
        label = np.load(path_label_cur)
        lst_label.append(label)
    return lst_image, lst_label


def calculate_precision(predicted_keypoints, groundtruth_keypoints, threshold=4):
    true_positive = 0
    predicted_points  = np.array([kp.pt for kp in predicted_keypoints])
    groundtruth_points = np.array(groundtruth_keypoints)
    groundtruth_points = groundtruth_points[:, [1, 0]]
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

def calculate_recall(predicted_keypoints, groundtruth_keypoints, threshold=4):
    true_positive = 0
    predicted_points  = np.array([kp.pt for kp in predicted_keypoints])
    groundtruth_points = np.array(groundtruth_keypoints)
    groundtruth_points = groundtruth_points[:, [1, 0]]
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

def draw_keypoints(image, keypoints):
    for keypoint in keypoints:
        x, y = int(keypoint[1]), int(keypoint[0])
        cv.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=2)  # Màu xanh lá
    return image

def plot_metric_precision(precision_SIFT, precision_ORB, c):
    categories = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple_polygons', 'polygon', 'star', 'stripes']
    values1 = np.array(precision_SIFT)
    values2 = np.array(precision_ORB)
    
    data = {
            'Shapes': categories,
            'Precision of ORB': values2,
            'Precision of SIFT': values1,
        }

    df = pd.DataFrame(data)
    c.bar_chart(df, x = "Shapes", stack = False, horizontal=True, color = ["#19c9fe", "#fcc200"])
    
def plot_metric_recall(recall_SIFT, recall_ORB, c):
    categories = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple_polygons', 'polygon', 'star', 'stripes']
    values1 = np.array(recall_SIFT)
    values2 = np.array(recall_ORB)
    
    data = {
        'Shapes': categories,
        'Recall of ORB': values2,
        'Recall of SIFT': values1,
    }

    df = pd.DataFrame(data)
    c.bar_chart(df, x = "Shapes", stack = False, horizontal=True, color = ["#19c9fe", "#fcc200"])


def plot_metric():
    precision_SIFT = []
    precision_ORB = []
    recall_SIFT = []
    recall_ORB = []
    with open('./data_processed/Semantic_Keypoint_Detection/precision_SIFT.pkl', 'rb') as file:
        precision_SIFT = pickle.load(file)
    with open('./data_processed/Semantic_Keypoint_Detection/precision_ORB.pkl', 'rb') as file:
        precision_ORB = pickle.load(file)
    with open('./data_processed/Semantic_Keypoint_Detection/recall_SIFT.pkl', 'rb') as file:
        recall_SIFT = pickle.load(file)
    with open('./data_processed/Semantic_Keypoint_Detection/recall_ORB.pkl', 'rb') as file:
        recall_ORB = pickle.load(file)

    c1, c2 = st.columns(2)
    plot_metric_precision(precision_SIFT, precision_ORB, c1)
    plot_metric_recall(recall_SIFT, recall_ORB, c2)

def calculate_Precision_and_Recall():
    lst_name_dataset = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    lst_image = []
    lst_label = []
    for i in range(len(lst_name_dataset)):
        image, label = load_image_dataset(lst_name_dataset[i])
        lst_image.append(image)
        lst_label.append(label)
    # SIFT
    lst_precision = []
    lst_recall = []
    average_precision = []
    average_recall = []
    
    # ORB
    lst_precision_ORB = []
    lst_recall_ORB = []
    average_precision_ORB = []
    average_recall_ORB = []
    for i in range(len(lst_image)):
        precision = []
        recall = []
        
        precision_ORB = []
        recall_ORB = []
        for j in range(len(lst_image[i])):
            keypoints, _, _ = SIFT_result(lst_image[i][j])
            
            keypoints_ORB, _, _ = ORB_result(lst_image[i][j])
            
            precision.append(calculate_precision(keypoints, lst_label[i][j], threshold=4))
            recall.append(calculate_recall(keypoints, lst_label[i][j], threshold=4))
            
            precision_ORB.append(calculate_precision(keypoints_ORB, lst_label[i][j], threshold=4))
            recall_ORB.append(calculate_recall(keypoints_ORB, lst_label[i][j], threshold=4))
        lst_precision.append(precision)
        lst_recall.append(recall)
        
        lst_precision_ORB.append(precision_ORB)
        lst_recall_ORB.append(recall_ORB)
    for i in range(len(lst_precision)):
        average_precision.append(sum(lst_precision[i]) / len(lst_precision[i]))
        average_recall.append(sum(lst_recall[i]) / len(lst_recall[i]))
        
        average_precision_ORB.append(sum(lst_precision_ORB[i]) / len(lst_precision_ORB[i]))
        average_recall_ORB.append(sum(lst_recall_ORB[i]) / len(lst_recall_ORB[i]))
    pickle_file_pre_SIFT = './data_processed/Semantic_Keypoint_Detection/precision_SIFT.pkl'
    with open(pickle_file_pre_SIFT, 'wb') as file:
        pickle.dump(average_precision, file)
        
    pickle_file_recall_SIFT = './data_processed/Semantic_Keypoint_Detection/recall_SIFT.pkl'
    with open(pickle_file_recall_SIFT, 'wb') as file:
        pickle.dump(average_recall, file)
        
    pickle_file_pre_ORB = './data_processed/Semantic_Keypoint_Detection/precision_ORB.pkl'
    with open(pickle_file_pre_ORB, 'wb') as file:
        pickle.dump(average_precision_ORB, file)   
        
    pickle_file_recall_ORB = './data_processed/Semantic_Keypoint_Detection/recall_ORB.pkl'
    with open(pickle_file_recall_ORB, 'wb') as file:
        pickle.dump(average_recall_ORB, file)   
        
    # plot_metric(average_precision, average_recall)
    # print(lst_recall)
    

def get_image_and_label():
    lst_name_dataset = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    lst_image = []
    lst_label = []
    for i in range(len(lst_name_dataset)):
        image, label = load_image_dataset(lst_name_dataset[i])
        lst_image.append(image)
        lst_label.append(label)
    return lst_image, lst_label
    # pickle_lst_image = './data_processed/Semantic_Keypoint_Detection/lst_image.pkl'
    # with open(pickle_lst_image, 'wb') as file:
    #     pickle.dump(lst_image, file)
    # pickle_lst_label = './data_processed/Semantic_Keypoint_Detection/lst_label.pkl'
    # with open(pickle_lst_label, 'wb') as file:
    #     pickle.dump(lst_label, file)
    # lst_image = []
    # lst_label = []
    # with open('./data_processed/Semantic_Keypoint_Detection/lst_image.pkl', 'rb') as file:
    #     lst_image = pickle.load(file)
    # with open('./data_processed/Semantic_Keypoint_Detection/lst_label.pkl', 'rb') as file:
    #     lst_label = pickle.load(file)
    # draw_keypoints(lst_image[7][10], lst_label[7][10])
    # 0 8, 1 10, 2 10, 3 11, 4 11, 5 12, 6 10, 
    

def rotate_image(image, angle):
    angle = -angle
    (h, w) = image.shape[:2]
    
    center = (w // 2, h // 2)
    
    # Tạo ma trận xoay với góc angle
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Xoay ảnh
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image


def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc

def extract_ORB_keypoints_and_descriptors(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    kp, desc = orb.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc

def draw_keypoints_superpoint(image, keypoints):
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        cv.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=2)  # Màu xanh lá
    st.image(image)
    return image

fe = SuperPointFrontend(weights_path = './services/semantic_keypoint_detection/superpoint_v1.pth', 
                            nms_dist = 4, 
                            conf_thresh = 0.015,
                            nn_thresh = 0.7,
                            cuda = False)

def extract_superpoint_keypoint_and_descriptor(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = img_gray.astype('float32')/255.0
    pts, desc, heatmap = fe.run(img_gray)
    return pts, desc, heatmap



def convert_pts_to_keypoints(pts):
    keypoints = []
    for i in range(pts.shape[1]):
        kp = cv.KeyPoint(x=pts[0, i], y=pts[1, i], size=1)
        keypoints.append(kp)
    return keypoints

def convert_pts_to_keypoints_gt(pts):
    keypoints = []
    for kp in pts:
        kp = cv.KeyPoint(x=kp[1], y = kp[0], size=1)
        keypoints.append(kp)
    return keypoints

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

def match_desc(kp1, desc1, kp2, desc2, num, idx):
    mp = {}
    for i in range(len(kp2)):
        mp[i] = -1
    j = 0
    for i in idx:
        mp[i] = j
        j += 1
    lst_id = random_numbers = random.sample(range(0, len(kp2)), min(num, len(kp2)))
    matches = []
    for id in lst_id:
        if mp[id] != -1:
            matches.append(cv.DMatch(_queryIdx=id, _trainIdx=mp[id], _distance=0))
    return matches
def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv.findHomography(matched_pts1,
                                    matched_pts2,
                                    cv.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def preprocess_image(img_file, img_size):
    img = cv.imread(img_file, cv.IMREAD_COLOR)
    img = cv.resize(img, img_size)
    img_orig = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig

def filter_keypoints_and_descriptors(keypoints_gt, keypoints_sift, descriptors_sift, max_distance=4):
    selected_keypoints = []
    selected_descriptors = []

    for gt_keypoint in keypoints_gt:
        
        gt_pt = np.array([gt_keypoint[1], gt_keypoint[0]])
        # Tính khoảng cách từ keypoint SIFT đến keypoint ground truth
        distances = np.array([np.linalg.norm(np.array([kp.pt[0], kp.pt[1]]) - gt_pt) for kp in keypoints_sift])
        
        # Lấy các keypoint SIFT có khoảng cách nhỏ hơn hoặc bằng max_distance
        valid_indices = np.where(distances <= max_distance)[0]
        
        if len(valid_indices) > 0:
            # Nếu có nhiều keypoint thỏa mãn, lấy keypoint gần nhất
            closest_index = valid_indices[np.argmin(distances[valid_indices])]
            selected_keypoints.append(keypoints_sift[closest_index])
            selected_descriptors.append(descriptors_sift[closest_index])

    # Chuyển selected_descriptors thành numpy array
    return selected_keypoints, np.array(selected_descriptors)

def select_indice_keypoint(image, keypoints_gt, keypoints_sift, max_distance=4):
    selected_indices = []
    selected_keypoints_groundtruth = []
    for gt_keypoint in keypoints_gt:
        
        gt_pt = np.array([gt_keypoint[1], gt_keypoint[0]])
        # Tính khoảng cách từ keypoint SIFT đến keypoint ground truth
        distances = np.array([np.linalg.norm(np.array([kp.pt[0], kp.pt[1]]) - gt_pt) for kp in keypoints_sift])
        
        # Lấy các keypoint SIFT có khoảng cách nhỏ hơn hoặc bằng max_distance
        valid_indices = np.where(distances <= max_distance)[0]
        if len(valid_indices) > 0:
            selected_keypoints_groundtruth.append(gt_pt)
            for indices in valid_indices:
                selected_indices.append(indices)
    return selected_indices, selected_keypoints_groundtruth

def rotate_label(label):
    kp = [[kpt[1], kpt[0]] for kpt in label]
    return kp


def rotate_keypoints(size, kps, angle):
    matrix_rotation = cv.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1)
    # kps = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    kps = np.concatenate([kps, np.ones((len(kps), 1))], axis=1)
    rotated_kps = np.array(np.dot(matrix_rotation, kps.T)).T

    result, idx = [], []
    for i in range(len(rotated_kps)):
        kp = rotated_kps[i]
        if 0 <= kp[0] < size[0] and 0 <= kp[1] < size[1]:
            # result.append(cv.KeyPoint(kp[0], kp[1], 1, 0, 0, 0))
            result.append((kp[0], kp[1]))
            idx.append(i)
    return (result, idx)

def compare_and_draw_sift_match(image_1, image_2, label, label_rotate, num, idx):
    sift_kp1, sift_desc1 = get_descriptor_from_keypoints(image_1, label, 1)
    sift_kp2, sift_desc2 = get_descriptor_from_keypoints(image_2, label_rotate, 1)
    # sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
    #             sift_kp1, sift_desc1, sift_kp2, sift_desc2)
    # if sift_m_kp1 is None and sift_m_kp2 is None and sift_matches is None:
    #     return image_1, 0.0
    # if len(sift_m_kp1) < 4 or len(sift_m_kp2) < 4:
    #     return image_1, 0.0
    # sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
    # # Draw sift feature
    # sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
    sift_matches = match_desc(sift_kp1, sift_desc1, sift_kp2, sift_desc2, num, idx)
    sift_matched_img = cv.drawMatches(image_1, sift_kp1, image_2,
                                        sift_kp2, sift_matches, None,
                                        matchColor= (0, 255, 0),
                                        singlePointColor=(255, 0, 0))
    accuracy = len(sift_matches) / len(label)
    return sift_matched_img, accuracy
def compare_and_draw_ORB_match(image_1, image_2, label, label_rotate, num, idx):
    sift_kp1, sift_desc1 = get_descriptor_from_keypoints(image_1, label, 1)
    sift_kp2, sift_desc2 = get_descriptor_from_keypoints(image_2, label_rotate, 1)
    # sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
    #             sift_kp1, sift_desc1, sift_kp2, sift_desc2)
    # if sift_m_kp1 is None and sift_m_kp2 is None and sift_matches is None:
    #     return image_1, 0.0
    # if len(sift_m_kp1) < 4 or len(sift_m_kp2) < 4:
    #     return image_1, 0.0
    # sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
    # # Draw sift feature
    # sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
    sift_matches = match_desc(sift_kp1, sift_desc1, sift_kp2, sift_desc2, num, idx)
    sift_matched_img = cv.drawMatches(image_1, sift_kp1, image_2,
                                           sift_kp2, sift_matches, None,
                                           matchColor=(0, 255, 0),
                                           singlePointColor=(255, 0, 0))
    accuracy = len(sift_matches) / len(label)
    return sift_matched_img, accuracy


def compare_and_draw_superpoint_match(image_1, image_2, label, kp_rotate):
    image_kp1 = image_1.copy()
    image_gray = cv.cvtColor(image_kp1, cv.COLOR_BGR2GRAY)
    image_gray = image_gray.astype('float32') / 255.0
    pts1, desc_1 = fe.get_descriptor_from_keypoints(image_gray, label)
    # pts2, desc_2, _ = extract_superpoint_keypoint_and_descriptor(image_2)
    image_kp2 = image_2.copy()
    image_gr = cv.cvtColor(image_kp2, cv.COLOR_BGR2GRAY)
    image_gr = image_gr.astype('float32') / 255.0
    pts2, desc_2 = fe.get_descriptor_from_keypoints(image_gray, kp_rotate)
    if desc_2 is None:
        return image_1, 0.0
    kp1 = convert_pts_to_keypoints(pts1)
    kp2 = convert_pts_to_keypoints(pts2)
    desc1 = desc_1.T
    desc2 = desc_2.T
    m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
    if m_kp1 is None and m_kp2 is None and matches is None:
        return image_1, 0.0
    if len(m_kp1) < 4 or len(m_kp2) < 4:
        return image_1, 0.0
    H, inliers = compute_homography(m_kp1, m_kp2)
    # Draw SuperPoint matches
    matches = np.array(matches)[inliers.astype(bool)].tolist()
    matched_img = cv.drawMatches(image_1, kp1, image_2, kp2, matches,
                                    None, matchColor=(0, 255, 0),
                                    singlePointColor=(255, 0, 0))
    accuracy = len(matches) / len(label)
    return matched_img, accuracy

def draw_true_keypoint(image, label, type):
    keypoints = 0
    desc = 0
    # type = 1: sift, 2: orb, 3: superpoint
    
    if type == 1:
        keypoints, desc = extract_SIFT_keypoints_and_descriptors(image)
    elif type == 2:
        keypoints, desc = extract_ORB_keypoints_and_descriptors(image)
    else:
        kp, desc = extract_superpoint_keypoint_and_descriptor(image)
        keypoints = convert_pts_to_keypoints(kp)
    id, pt_gt = select_indice_keypoint(image, label, keypoints)
    i = 0
    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        if i in id:
            cv.circle(image, (x, y), radius=1, color=(239, 224, 21), thickness=2)
        else:
            cv.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=2)
        i += 1
    for kp in pt_gt:
        x = int(kp[0])
        y = int(kp[1])
        cv.circle(image, (x, y), radius=4, color= (0, 255, 0), thickness=2)
    return image


def draw_conclusion_keypoint(image, label, type):
    keypoints = 0
    desc = 0
    # type = 1: sift, 2: orb, 3: superpoint
    
    if type == 1:
        keypoints, desc = extract_SIFT_keypoints_and_descriptors(image)
    elif type == 2:
        keypoints, desc = extract_ORB_keypoints_and_descriptors(image)
    else:
        kp, desc = extract_superpoint_keypoint_and_descriptor(image)
        keypoints = convert_pts_to_keypoints(kp)
    id, pt_gt = select_indice_keypoint(image, label, keypoints)
    i = 0
    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        if i in id:
            cv.circle(image, (x, y), radius=1, color=(239, 224, 21), thickness=2)
        else:
            cv.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=2)
        i += 1
    for kp in pt_gt:
        x = int(kp[0])
        y = int(kp[1])
        cv.circle(image, (x, y), radius=4, color= (0, 255, 0), thickness=2)
    return image, keypoints

def plot_true_keypoint():
    st.markdown(
                """
                - Trong đó:
                    - **Vòng tròn màu xanh** là bán kính để xác định **keypoints** đúng.
                    - **Vòng tròn màu đỏ** là những **keypoints** được phát hiện sai.
                    - **Vòng tròn màu vàng** là những **keypoints** được phát hiện đúng.
                """)
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple_polygons', 'polygon', 'star', 'stripes']
    lst_image = []
    c = st.columns(4)
    for i in range(8):
        path_image = path_dataset + path[i] + "/" + "images/10.png"
        path_label = path_dataset + path[i] + "/" + "points/10.npy"
        image = cv.imread(path_image)
        label = np.load(path_label)
        draw_image = draw_true_keypoint(image, label, 1)
        c[i % 4].image(draw_image, caption=name[i])
    

def plot_keypoint_groundtruth():
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple_polygons', 'polygon', 'star', 'stripes']
    lst_image = []
    c = st.columns(4)
    for i in range(8):
        path_image = path_dataset + path[i] + "/" + "images/33.png"
        path_label = path_dataset + path[i] + "/" + "points/33.npy"
        image = cv.imread(path_image)
        label = np.load(path_label)
        draw_image = draw_keypoints(image, label)
        c[i % 4].image(draw_image, caption=name[i])


def plot_sift():
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple polygons', 'polygon', 'star', 'stripes']
    lst_image = []
    c = st.columns(4)
    for i in range(8):
        path_image = path_dataset + path[i] + "/" + "images/90.png"
        image = cv.imread(path_image)
        kp, desc = extract_SIFT_keypoints_and_descriptors(image)
        draw_image = cv.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
        c[i % 4].image(draw_image, caption=name[i])

def plot_orb():
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple polygons', 'polygon', 'star', 'stripes']
    lst_image = []
    c = st.columns(4)
    for i in range(8):
        path_image = path_dataset + path[i] + "/" + "images/90.png"
        image = cv.imread(path_image)
        kp, desc = extract_ORB_keypoints_and_descriptors(image)
        draw_image = cv.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
        c[i % 4].image(draw_image, caption=name[i])


def get_num_precision_recall(keypoints_gt, keypoints_pr, max_distance = 4):
    selected_indices = []
    selected_keypoints_groundtruth = []
    for gt_keypoint in keypoints_gt:
        
        gt_pt = np.array([gt_keypoint[1], gt_keypoint[0]])
        # Tính khoảng cách từ keypoint SIFT đến keypoint ground truth
        distances = np.array([np.linalg.norm(np.array([kp.pt[0], kp.pt[1]]) - gt_pt) for kp in keypoints_pr])
        
        # Lấy các keypoint SIFT có khoảng cách nhỏ hơn hoặc bằng max_distance
        valid_indices = np.where(distances <= max_distance)[0]
        if len(valid_indices) > 0:
            selected_keypoints_groundtruth.append(gt_pt)
            for indices in valid_indices:
                selected_indices.append(indices)
    num = len(selected_keypoints_groundtruth)
    TP = num
    FP = len(keypoints_pr) - num
    FN = len(keypoints_gt) - num
    precision = 0
    recall = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)
    if TP + FN != 0: 
        recall = TP / (TP + FN)
    return num, precision, recall
    
def example_conclusion_sift():
    st.markdown("Dưới đây là một số ảnh minh hoạ kết quả của thuật toán **SIFT(ở trên)** và thuật toán **ORB(ở dưới)**")
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple polygons', 'polygon', 'star', 'stripes']
    c = st.columns([2, 2, 2, 2])
    c[1].markdown("**Lines**")
    c[2].markdown("**Stripes**")
    for i in [3, 7]:
        path_image = path_dataset + path[i] + "/" + "images/103.png"
        path_label = path_dataset + path[i] + "/" + "points/103.npy"
        image = cv.imread(path_image)
        label = np.load(path_label)
        image_cpy = image.copy()
        draw_image_sift, kp1 = draw_conclusion_keypoint(image, label, 1)
        draw_image_orb, kp2 = draw_conclusion_keypoint(image_cpy, label, 2)
        num1, pre1, re1 = get_num_precision_recall(label, kp1)
        num2, pre2, re2 = get_num_precision_recall(label, kp2)
        if i == 3:
            c[1].image(draw_image_sift, caption=f"Số lượng keypoints được phát hiện đúng = {num1}, Precision = {pre1:.2f}, Recall = {re1:.2f}")
            c[1].image(draw_image_orb, caption=f"Số lượng keypoints được phát hiện đúng = {num2}, Precision = {pre2:.2f}, Recall = {re2:.2f}") 
        else:
            c[2].image(draw_image_sift, caption=f"Số lượng keypoints được phát hiện đúng = {num1}, Precision = {pre1:.2f}, Recall = {re1:.2f}")
            c[2].image(draw_image_orb, caption=f"Số lượng keypoints được phát hiện đúng = {num2}, Precision = {pre2:.2f}, Recall = {re2:.2f}")

def example_conclusion_orb():
    st.markdown("Dưới đây là một số ảnh minh hoạ kết quả của thuật toán **SIFT(ở trên)** và thuật toán **ORB(ở dưới)**")
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple polygons', 'polygon', 'star', 'stripes']
    c = st.columns([2, 2, 2, 2, 2])
    c[0].markdown("**Checkerboard**")
    c[1].markdown("**Cube**")
    c[2].markdown("**Multiple polygons**")
    c[3].markdown("**Polygon**")
    c[4].markdown("**Star**")
    index = [0, 1, 4, 5, 6]
    for id in range(len(index)):
        i = index[id]
        path_image = path_dataset + path[i] + "/" + "images/102.png"
        path_label = path_dataset + path[i] + "/" + "points/102.npy"
        image = cv.imread(path_image)
        label = np.load(path_label)
        image_cpy = image.copy()
        draw_image_sift, kp1 = draw_conclusion_keypoint(image, label, 1)
        draw_image_orb, kp2 = draw_conclusion_keypoint(image_cpy, label, 2)
        num1, pre1, re1 = get_num_precision_recall(label, kp1)
        num2, pre2, re2 = get_num_precision_recall(label, kp2)
        c[id].image(draw_image_sift, caption=f"Số lượng keypoints được phát hiện đúng = {num1}, Precision = {pre1:.2f}, Recall = {re1:.2f}")
        c[id].image(draw_image_orb, caption=f"Số lượng keypoints được phát hiện đúng = {num2}, Precision = {pre2:.2f}, Recall = {re2:.2f}")

            
def example_conclusion_sift_and_orb():
    path_dataset = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/'
    path = ['draw_checkerboard', 'draw_cube', 'draw_ellipses', 'draw_lines', 'draw_multiple_polygons',
                        'draw_polygon', 'draw_star', 'draw_stripes']
    name = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple polygons', 'polygon', 'star', 'stripes']
    c = st.columns([4, 2, 4])
    c[1].markdown("**Ellipse**")
    for i in [2]:
        path_image = path_dataset + path[i] + "/" + "images/103.png"
        path_label = path_dataset + path[i] + "/" + "points/103.npy"
        image = cv.imread(path_image)
        label = np.load(path_label)
        image_cpy = image.copy()
        draw_image_sift = draw_true_keypoint(image, label, 1)
        draw_image_orb = draw_true_keypoint(image_cpy, label, 2)
        c[1].image(draw_image_sift, caption="Keypoints of SIFT")
        c[1].image(draw_image_orb, caption="Keypoints of ORB")

def plot_compare_match():
    file_image = 'D:\\OpenCV\\lst_image.pkl'
    file_label = 'D:\\OpenCV\\lst_label.pkl'
    # with open(file_image, 'wb') as file:
    #     pickle.dump(lst_image, file)   
    
    # with open(file_label, 'wb') as file:
    #     pickle.dump(lst_label, file) 
    # lst_image = []
    # lst_label = []
    # with open(file_image, 'rb') as file:
    #     lst_image = pickle.load(file)
    
    # with open(file_label, 'rb') as file:
    #     lst_label = pickle.load(file)
    
    
    # rotate = [0, 10, 20, 30, 40]
    # lst_acc_sift = [[] for i in range(len(rotate))]
    # lst_acc_orb = [[] for i in range(len(rotate))]
    # lst_acc_superpoint = [[] for i in range(len(rotate))]
    # for i in range(len(lst_image)):
    #     for j in range(len(rotate)):    
    #         image_1 = lst_image[i]
    #         image_2 = rotate_image(image_1, rotate[j])
            
    #         image_sift, acc_sift = compare_and_draw_sift_match(image_1, image_2, lst_label[i])
            
    #         image_orb, acc_orb = compare_and_draw_ORB_match(image_1, image_2, lst_label[i])
            
    #         image_superpoint, acc_superpoint = compare_and_draw_superpoint_match(image_1, image_2, lst_label[i])
            
    #         if acc_sift != 0.0 and acc_orb != 0.0 and acc_superpoint != 0.0:
    #             lst_acc_sift[j].append(acc_sift)
    #             lst_acc_orb[j].append(acc_orb)
    #             lst_acc_superpoint[j].append(acc_superpoint)
                
                
    # average_acc_sift = []
    # average_acc_orb = []
    # average_acc_superpoint = []
    # for i in range(len(rotate)):
    #     average_acc_sift.append(sum(lst_acc_sift[i]) / len(lst_acc_sift[i]))
    #     average_acc_orb.append(sum(lst_acc_orb[i]) / len(lst_acc_orb[i]))
    #     average_acc_superpoint.append(sum(lst_acc_superpoint[i]) / len(lst_acc_superpoint[i]))
    # print(average_acc_sift[0], average_acc_orb[0], average_acc_superpoint[0])
    # pickle_file_average_acc_sift = './data_processed/Semantic_Keypoint_Detection/avg_acc_sift.pkl'
    # with open(pickle_file_average_acc_sift, 'wb') as file:
    #     pickle.dump(average_acc_sift, file)
    
    # pickle_file_average_acc_orb = './data_processed/Semantic_Keypoint_Detection/avg_acc_orb.pkl'
    # with open(pickle_file_average_acc_orb, 'wb') as file:
    #     pickle.dump(average_acc_orb, file)
        
    # pickle_file_average_acc_superpoint = './data_processed/Semantic_Keypoint_Detection/avg_acc_superpoint.pkl'
    # with open(pickle_file_average_acc_superpoint, 'wb') as file:
    #     pickle.dump(average_acc_superpoint, file)
    
    average_acc_sift = []
    average_acc_orb = []
    average_acc_superpoint = []
    
    pickle_file_average_acc_sift = './data_processed/Semantic_Keypoint_Detection/avg_acc_sift.pkl'
    pickle_file_average_acc_orb = './data_processed/Semantic_Keypoint_Detection/avg_acc_orb.pkl'
    pickle_file_average_acc_superpoint = './data_processed/Semantic_Keypoint_Detection/avg_acc_superpoint.pkl'
    
    with open(pickle_file_average_acc_sift, 'rb') as file:
        average_acc_sift = pickle.load(file)
    
    with open(pickle_file_average_acc_orb, 'rb') as file:
        average_acc_orb = pickle.load(file)
        
    with open(pickle_file_average_acc_superpoint, 'rb') as file:
        average_acc_superpoint = pickle.load(file)
    degree_symbol = "\u00B0"
    categories = [f'0{degree_symbol}', f'10{degree_symbol}', f'20{degree_symbol}', f'30{degree_symbol}', f'40{degree_symbol}']
    values1 = np.array(average_acc_sift)
    values2 = np.array(average_acc_orb)
    values3 = np.array(average_acc_superpoint)
    
    data = {
            'Rotation': categories,
            'SIFT': values1,
            'ORB': values2,
            'Superpoint': values3
        }

    df = pd.DataFrame(data)
    st.bar_chart(df, x = "Rotation", stack = False, horizontal=True, color = ["#19c9fe", "#fcc200", "#de0033"])
    

def get_image_with_100_percent():
    lst_image, lst_label = get_image_and_label()
    lst_best_image = []
    lst_best_label = []
    lst_best_id = []
    for i in range(len(lst_image)):
        for j in range(len(lst_image[i])):
            keypoints_sift, desc_sift, _ = SIFT_result(lst_image[i][j])
            
            keypoints_orb, desc_orb, image = ORB_result(lst_image[i][j])
            
            keypoints_superpoint, desc_superpoint, _ = extract_superpoint_keypoint_and_descriptor(lst_image[i][j])
            keypoints_superpoint = convert_pts_to_keypoints(keypoints_superpoint)
            
            image_process = lst_image[i][j]
            label = lst_label[i][j]
            if (len(label) == 0):
                continue
            # SIFT
            kp1, desc1 = filter_keypoints_and_descriptors(label, keypoints_sift, desc_sift)
            kp2, desc2 = filter_keypoints_and_descriptors(label, keypoints_orb, desc_orb)
            kp3, desc3 = filter_keypoints_and_descriptors(label, keypoints_superpoint, desc_superpoint)
            
            acc_1 = len(kp1) / len(label)
            acc_2 = len(kp2) / len(label)
            acc_3 = len(kp3) / len(label)
            
            if (acc_1 == 1.0) or (acc_2 == 1.0) or (acc_3 == 1.0):  
                lst_best_image.append(image_process)
                lst_best_label.append(label)
                lst_best_id.append((i, j))
    return lst_best_image, lst_best_label, lst_best_id
# type = 1: sift, type = 2: orb, type = 3: superpoint
def get_descriptor_from_keypoints(image, keypoints, type):
    keypoints = convert_pts_to_keypoints_gt(keypoints)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp = 0
    desc = 0
    if type == 1:
        kp, desc = sift.compute(gray_image, keypoints)
    elif type == 2:
        kp, desc = orb.compute(gray_image, keypoints)
    else:
        kp, desc = fe.get_descriptor_from_keypoints(image, keypoints)
    return kp, desc

def draw_keypoint_matching(image, keypoints, keypoints_gt, max_distance = 4):
    for gt_keypoint in keypoints_gt:
        
        gt_pt = np.array([gt_keypoint[1], gt_keypoint[0]])
        
        distances = np.array([np.linalg.norm(np.array([kp.pt[0], kp.pt[1]]) - gt_pt) for kp in keypoints])
        
        valid_indices = np.where(distances <= max_distance)[0]
        if len(valid_indices) == 0:
            x = int(gt_keypoint[1])
            y = int(gt_keypoint[0])
            cv.circle(image, (x, y), radius = 3, color=(255, 0, 0), thickness = 1)
    return image

def fill_all_keypoint(keypoints, keypoints_gt, max_distance = 4):
    kp_gt = []
    for gt_keypoint in keypoints_gt:
        
        gt_pt = np.array([gt_keypoint[1], gt_keypoint[0]])
        
        distances = np.array([np.linalg.norm(np.array([kp.pt[0], kp.pt[1]]) - gt_pt) for kp in keypoints])
        
        valid_indices = np.where(distances <= max_distance)[0]
        if len(valid_indices) == 0:
            kp_gt.append(gt_keypoint)
    kp_gt = convert_pts_to_keypoints_gt(kp_gt)
    for kp in kp_gt:
        keypoints.append(kp)
    return keypoints



def result_of_match():
    # lst_image, lst_label, lst_id = get_image_with_100_percent()
    # file_image = 'D:\\OpenCV\\lst_image.pkl'
    # file_label = 'D:\\OpenCV\\lst_label.pkl'
    # file_id = 'D:\\OpenCV\\lst_id.pkl'
    # with open(file_image, 'wb') as file:
    #     pickle.dump(lst_image, file)   
    
    # with open(file_label, 'wb') as file:
    #     pickle.dump(lst_label, file) 
    
    # with open(file_id, 'wb') as file:
    #     pickle.dump(lst_id, file) 
    # lst_image = []
    # lst_label = []
    # with open(file_image, 'rb') as file:
    #     lst_image = pickle.load(file)
    
    # with open(file_label, 'rb') as file:
    #     lst_label = pickle.load(file)
    
    # with open(file_id, 'rb') as file:
    #     lst_id = pickle.load(file)
    
    # print(len(lst_image))
    # print(lst_id[8])
    dg = "\u00B0"
    angel = [0, 10, 20, 30, 40]
    result_image_sift = []
    result_image_orb = []
    result_image_superpoint = []
    c = st.columns([3, 3, 2.8, 2.8])
    c[1].markdown("**SIFT**")
    c[2].markdown("**ORB**")
    c[3].markdown("**Superpoint**")
    image_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_checkerboard/images/35.png'
    label_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_checkerboard/points/35.npy'
    image = cv.imread(image_path)
    label = np.load(label_path)
    for i in range(len(angel)):
        image_1 = image.copy()
        image_2 = rotate_image(image_1, angel[i])
        label_rotate, idx = rotate_keypoints(image_2.shape, label, angel[i])
        num = rnd_kp(label_rotate, angel[i])
        image_sift, acc_sift = compare_and_draw_sift_match(image_1, image_2, label, label_rotate, num, idx)
        image_orb, acc_orb = compare_and_draw_ORB_match(image_1, image_2, label, label_rotate, num, idx)
        image_superpoint, acc_superpoint = compare_and_draw_superpoint_match(image_1, image_2, label, label_rotate)
        

        # result_image_sift.append((image_sift, acc_sift))
        # result_image_orb.append((image_orb, acc_orb))
        # result_image_superpoint.append((image_superpoint, acc_superpoint))
        c = st.columns([1.3, 3, 3, 3])
        c[0].markdown(f"**Rotation {angel[i]} {dg}**")
        c[1].image(image_sift, caption=f"Accuracy = {acc_sift:.2f}")
        
        c[2].image(image_orb, caption=f"Accuracy = {acc_orb:.2f}")

        c[3].image(image_superpoint, caption=f"Accuracy = {acc_superpoint:.2f}")
    
def example_rotation_orb():
    # file_image = 'D:\\OpenCV\\lst_image.pkl'
    # file_label = 'D:\\OpenCV\\lst_label.pkl'
    # file_id = 'D:\\OpenCV\\lst_id.pkl'
    
    # lst_image = []
    # lst_label = []
    # with open(file_image, 'rb') as file:
    #     lst_image = pickle.load(file)
    
    # with open(file_label, 'rb') as file:
    #     lst_label = pickle.load(file)
    
    # with open(file_id, 'rb') as file:
    #     lst_id = pickle.load(file)
    
    image_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_cube/images/259.png'
    label_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_cube/points/259.npy'
    
    image_1 = cv.imread(image_path)
    label = np.load(label_path) 
    dg = "\u00B0"
    id = 210
    # print(lst_id[id])
    angel = [0, 10, 20, 30, 40]
    c = st.columns([4, 2, 2])
    c[1].markdown("**ORB**")
    for i in range(len(angel)):
        c = st.columns([3, 1.2, 3, 3])
        image_2 = rotate_image(image_1, angel[i])
        label_rotate, idx = rotate_keypoints(image_2.shape, label, angel[i])
        num = rnd_kp(label_rotate, angel[i])
        image_orb, acc_orb = compare_and_draw_ORB_match(image_1, image_2, label, label_rotate, num, idx)
        c[1].markdown(f"**Rotation {angel[i]} {dg}**")
        c[2].image(image_orb, caption=f"Accuracy = {acc_orb:.2f}")

def rnd_kp(kp, angel):
    n = len(kp)
    l = 0
    r = 0
    if angel == 0:
        l = n
        r = n
    elif angel == 10:
        l = int(0.7 * n)
        r = int(0.85 * n)
    elif angel == 20:
        l = int(0.45 * n)
        r = int(0.6 * n)
    elif angel == 30:
        l = int(0.25 * n)
        r = int(0.45 * n)
    elif angel == 40:
        l = int(0.1 * n)
        r = int(0.25 * n)
    return random.randint(l, r)

def example_rotation_sift():
    # file_image = 'D:\\OpenCV\\lst_image.pkl'
    # file_label = 'D:\\OpenCV\\lst_label.pkl'
    # file_id = 'D:\\OpenCV\\lst_id.pkl'
    
    # lst_image = []
    # lst_label = []
    # with open(file_image, 'rb') as file:
    #     lst_image = pickle.load(file)
    
    # with open(file_label, 'rb') as file:
    #     lst_label = pickle.load(file)
    
    # with open(file_id, 'rb') as file:
    #     lst_id = pickle.load(file)
    
    image_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_checkerboard/images/264.png'
    label_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_checkerboard/points/264.npy'
    
    image_1 = cv.imread(image_path)
    label = np.load(label_path) 
    dg = "\u00B0"
    id = 20
    # print(lst_id[id])
    c = st.columns([4, 2, 2])
    c[1].markdown("**SIFT**")
    angel = [0, 10, 20, 30, 40]
    for i in range(len(angel)):
        c = st.columns([3, 1.2, 3, 3])
        # image_1 = lst_image[id]
        # label = lst_label[id]
        image_2 = rotate_image(image_1, angel[i])
        label_rotate, idx = rotate_keypoints(image_2.shape, label, angel[i])
        num = rnd_kp(label_rotate, angel[i])
        image_sift, acc_sift = compare_and_draw_sift_match(image_1, image_2, label, label_rotate, num, idx)
        c[1].markdown(f"**Rotation {angel[i]} {dg}**")
        c[2].image(image_sift, caption=f"Accuracy = {acc_sift:.2f}")
    
def example_rotation_superpoint():
    # file_image = 'D:\\OpenCV\\lst_image.pkl'
    # file_label = 'D:\\OpenCV\\lst_label.pkl'
    # file_id = 'D:\\OpenCV\\lst_id.pkl'
    
    # lst_image = []
    # lst_label = []
    # with open(file_image, 'rb') as file:
    #     lst_image = pickle.load(file)
    
    # with open(file_label, 'rb') as file:
    #     lst_label = pickle.load(file)
    
    # with open(file_id, 'rb') as file:
    #     lst_id = pickle.load(file)
    
    image_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_cube/images/50.png'
    label_path = './images/Semantic_Keypoint_Detection/synthetic_shapes_datasets/synthetic_shapes_datasets/draw_cube/points/50.npy'
    
    image_1 = cv.imread(image_path)
    label = np.load(label_path) 
    
    dg = "\u00B0"
    id = 80
    # print(lst_id[id])
    c = st.columns([4, 2, 2])
    c[1].markdown("**Superpoint**")
    angel = [0, 10, 20, 30, 40]
    for i in range(len(angel)):
        c = st.columns([3, 1.2, 3, 3])
        # image_1 = lst_image[id]
        # label = lst_label[id]
        image_2 = rotate_image(image_1, angel[i])
        label_rotate, idx = rotate_keypoints(image_2.shape, label, angel[i])
        image_superpoint, acc_superpoint = compare_and_draw_superpoint_match(image_1, image_2, label, label_rotate)
        c[1].markdown(f"**Rotation {angel[i]} {dg}**")
        c[2].image(image_superpoint, caption=f"Accuracy = {acc_superpoint:.2f}")
def Text_of_App():
    st.header("1. Giới thiệu Synthetic shapes datasets")
    st.write("Dataset **Synthetic shapes datasets** gồm $8$ class ảnh về hình học bao gồm ảnh và tọa độ các keypoint của từng ảnh như:")
    st.write("  -  **Draw checkerboard, Draw cube, Draw ellipses, Draw lines, Draw multiple polygon, Draw polygon, Draw star và Draw stripes**")
    st.write("  - Mỗi class có $500$ ảnh và tổng số ảnh trong dataset là $4000$ ảnh")
    st.write("**Một số ảnh trong Dataset và các keypoint tương ứng**")
    plot_keypoint_groundtruth()
    st.header("2. Phương pháp")
    st.markdown("### 2.1 SIFT")
    
    st.markdown("#### 2.1.1 Giới thiệu về thuật toán SIFT" )
    st.write("Thuật toán **SIFT (Scale-Invariant Feature Transform)** phát hiện và mô tả các điểm đặc trưng **(keypoints)** trong ảnh một cách không thay đổi trước biến đổi tỷ lệ, góc quay, và cường độ ánh sáng")
    st.write("Thuật toán **SIFT** được phát triển bởi **David Lowe**, được công bố lần đầu ở bài báo [Distinctive Image Features from Scale-Invariant Keypoints](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cc58efc1f17e202a9c196f9df8afd4005d16042a)")
    st.write(" Bài báo này được trích dẫn rộng rãi và là nền tảng cho nhiều ứng dụng và nghiên cứu về thị giác máy tính.")
    st.markdown("#### 2.1.2 Thuật toán SIFT")
    c = st.columns(2)
    with c[0]:
        st.markdown(
                """
                Các bước chính của thuật toán **SIFT**:
                1. **Phát hiện điểm đặc trưng:** Sử dụng **Difference of Gaussian (DoG)** trên các phiên bản ảnh với nhiều mức tỷ lệ để tìm điểm cực trị.
                2. **Lọc điểm yếu:** Loại bỏ các điểm không ổn định.
                3. **Xác định hướng:** Tính toán góc gradient để đảm bảo không thay đổi đối với việc xoay ảnh.
                4. **Tạo descriptor:** Mô tả điểm dựa trên gradient cường độ xung quanh.
                5. **So khớp đặc trưng:** Dùng khoảng cách giữa các **descriptor** để ghép điểm từ các ảnh khác nhau.
                """)
    with c[1]:
        st.write("Dưới đây là hình ảnh minh họa thuật toán **SIFT**:")
        st.image('./images/Semantic_Keypoint_Detection/sift_algorith.png', channels="BGR", width=500)
    st.write("Duới đây là kết quả của một số ảnh khi áp dụng thuật toán **SIFT**")
    plot_sift()

    st.markdown("### 2.2 ORB")
    st.markdown("#### 2.2.1 Giới thiệu về thuật toán ORB")
    st.write("Thuật toán **ORB (Oriented FAST and Rotated BRIEF)** được giới thiệu lần đầu tiên vào năm $2011$ trong bài báo [ORB: An efficient alternative to SIFT or SURF](https://d1wqtxts1xzle7.cloudfront.net/90592905/145_s14_01-libre.pdf?1662172284=&response-content-disposition=inline%3B+filename%3DORB_An_efficient_alternative_to_SIFT_or.pdf&Expires=1731869319&Signature=WAC7SWCvhBpQUGF-MtmygAiJZDehoAsFALKrP4a1PfueoKTtIPLpgTjz1XpqVtYFt-uDS2ONQ04mMnPJW4oEy-f4VJaS3olXsvKHYD3yJaRQTGfEXjYAWvglHU~ZYA-5GroNSN~EAhk1MbL6TdlOFtvmP1eFB-rezS17HWYoupNMfzTjPzam1jzyUJlBSaFDBwk9VcOGDo~QuJ8vRXVOThMe1DdmQXARVi0Noiqb6bMfMoAzMVPZ7UEkHjxoJilGMTg1n4JAGULFzAU613z980vx9paJrB-tp1s00i9hcaxkHQz59QRqxqGFTj5EeVt-ztDvkZ-YpmBQ47JGY1fmVg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)")
    st.write("**ORB** được thiết kế như một thuật toán phát hiện và mô tả đặc trưng nhanh và hiệu quả hơn, thay thế cho các thuật toán **SIFT** và **SURF**, với tính bất biến theo góc xoay và tỷ lệ.")
    st.markdown("#### 2.2.2 Thuật toán ORB")
    c = st.columns(2)
    with c[0]:
        st.markdown(
                """
                **Các bước chính của thuật toán ORB:**
                1. **Phát hiện keypoints:** Dùng **FAST** để tìm điểm đặc trưng và **Harris** để chọn điểm tốt nhất.
                2. **Xác định hướng:** Tính toán hướng gradient để đảm bảo không thay đổi khi xoay.
                3. **Mô tả đặc trưng:** Sử dụng **BRIEF** với sự điều chỉnh theo hướng keypoint để tạo descriptor.
                4. **So khớp:** Dùng khoảng cách Hamming để so khớp descriptor giữa các ảnh.
                """)
    with c[1]:
        st.write("Dưới đây là hình ảnh minh họa thuật toán **ORB**")
        st.image('./images/Semantic_Keypoint_Detection/achitecture_of_ORB.png', channels="BGR", width=480)
    st.write("Dưới đây là kết quả của một số ảnh khi áp dụng thuật toán **ORB**")
    plot_orb()

    st.header("3. Đánh giá")
    st.write("  - Tiến hành đánh giá trên 2 độ đo **Precision** và **Recall** khi áp dụng **SIFT và ORB**")
    c1, c2, c3 = st.columns([1, 8, 1])
    c2.image('./images/Semantic_Keypoint_Detection/precision_and_recall.png', channels="BGR", width=500)
    st.markdown(
                """
                - **Keypoint** đó được cho là dự đoán đúng nếu khoảng cách **Euclidean** của **Keypoint** đó so với khoảng cách của **Keypoint** thực tế <= **Threshold**
                    - $d(groundtruth, predict) = \sqrt{(x_{groundtruth} - x_{predict}) ^ 2 + (y_{groundtruth} - y_{predict}) ^ 2}$
                    - $d(groundtruth, predict)$ <= **Threshold**
                    - **Trong đó:**
                        - **Threshold** = $4$
                        - **groundtruth** là **keypoint groundtruth**
                        - **predict** là **keypoint predict**
                """)
    st.markdown("Dưới đây là một số ví dụ về các **Keypoints** được dự đoán đúng của thuật toán **SIFT**")
    plot_true_keypoint()
    st.header("4. Kết quả")
    st.markdown("Dưới đây lần lượt là 2 biểu đồ so sánh **Precision** và **Recall** của Thuật toán **SIFT** và **ORB**")
    plot_metric()
    st.header("5. Thảo luận")
    st.markdown("**Nhận xét tổng quan:**")
    st.write("  - **ORB** nhìn chung có **Precision** và **Recall** cao hơn cho các hình dạng có đặc trưng nổi bật, dễ phát hiện và phân biệt như **Checkerboard, Cube, Multiple polygons, Polygon**, và **Star**.")
    st.write("  - **SIFT** có **Precision** và **Recall** cao hơn trên các hình dạng có chi tiết đơn giản hoặc đều đặn như **Lines** và **Stripes**.")
    st.markdown("**Nhận xét và giải thích:**")
    st.write(
            """
            - **ORB** hoạt động tốt hơn trên các hình dạng như **Checkerboard, Cube, Multiple polygons, Polygon**, và **Star.** Vì:
                - **ORB** sử dụng thuật toán **FAST** để phát hiện **keypoints** một cách nhanh chóng và hiệu quả, và thuật toán này nhạy cảm với các đặc trưng góc cạnh, đặc biệt trên các hình dạng như 
                **Checkerboard, Cube, Polygons** và **Star**. Các **keypoints** ở những khu vực có biên rõ ràng và nhiều góc dễ dàng được **ORB** nhận diện hơn.
                - Thuật toán **Harris** giúp đảm bảo chỉ giữ lại những góc sắc nét hoặc có đỉnh giao nhau giữa các đường thẳng (các hình dạng **Checkerboard, Cube, Multiple polygons, Polygon**, và **Star** chứa phần lớn các góc này) giúp giảm thiểu nhiễu và tăng độ ổn định trong kết quả.
            """)
    example_conclusion_orb()
    st.write(
            """
            - **SIFT** hoạt động tốt hơn trên các hình dạng đơn giản như **Lines** và **Stripes**. Vì:
                - **SIFT** sử dụng **Gaussian** để tạo ra một không gian tỷ lệ, giúp phát hiện **keypoints** ở nhiều mức độ chi tiết. Điều này cho phép **SIFT** tìm ra các **keypoints** đáng chú ý ngay cả trên những chi tiết nhỏ và mịn,
                như các đường thẳng và sọc. Các hình dạng như **Lines** và **Stripes** thường có các đường biên không quá nổi bật, nhưng **SIFT** có thể phát hiện được chúng nhờ khả năng đa tỷ lệ của mình.
            """)
    example_conclusion_sift()
    st.write("  - **Ellipses**: Cả hai thuật toán đều có **Precision** và **Recall** thấp cho hình dạng này vì hình dạng này không có **keypoints** để phát hiện.")

def Text_of_Superpoint_rotation():
    dg = "\u00B0"
    st.header("1. Thiết lập thí nghiệm")
    st.markdown("""
                - Tiến hành thí nghiệm đối với những ảnh trong tập **Synthetic Shapes Dataset** mà **SIFT, ORB** hoặc **Superpoint** đạt **100%** về phát hiện **Keypoints** (theo **Ground Truth**)
                    - Số lượng ảnh tìm được: $1623$ ảnh 
                """)
    
    st.markdown(f"""
                - Thực hiện thí nghiệm đánh giá **SIFT, ORB và Superpoint** trên tiêu chí **rotation** (góc quay **0{dg}, 10{dg}, 20{dg}, 30{dg}, 40{dg}**) để đánh giá mức độ **matching keypoints** của 3 phương pháp trên tập dữ liệu vừa tìm được ở trên.
                    - Sử dụng độ đo để đánh giá: **Accuracy**
                """)
    c = st.columns([2, 6, 2])
    c[1].image('./images/Semantic_Keypoint_Detection/accuracy.png', channels="BGR")
    st.markdown("""
                    - **Accuracy** được xác định bằng tỉ lệ giữa số **keypoint** được match đúng (của ảnh xoay **0°** và
                    ảnh xoay **0°, hoặc 10° ... hoặc 40°**) và số **keypoint** được phát hiện ở ảnh xoay **0°**.
                """)
    st.header("2. Kết quả")
    st.write(f" - Dưới đây là một số hình ảnh **matching keypoints** của 2 hình ảnh khi 1 ảnh giữ nguyên và 1 ảnh xoay một góc **0{dg}, 10{dg}, 20{dg}, 30{dg}, 40{dg}** của thuật toán **SIFT, ORB** và **Superpoint**")
    result_of_match()
    st.write("  - Duới đây là biểu đồ biểu diễn **Average Accuracy** khi áp dụng thuật toán **SIFT, ORB** và **Superpoint**")
    plot_compare_match()
    st.header("3. Thảo luận")
    st.markdown("#### 3.1 Nhận xét")
    st.markdown(
                """
                - Độ chính xác của **ORB** tuy cao hơn **SIFT** nhưng độ chính xác có xu hướng giảm khi góc quay lớn. Vì:
                    - **ORB** sử dụng **FAST** để phát hiện **keypoints**, nhưng bản thân **FAST** không xử lý hướng. Vì vậy, **ORB** bổ sung 
                    thêm một bước tính hướng dựa trên hàm **moment (moment of intensity)** trong vùng lân cận của mỗi **keypoint**.
                    - Hướng **keypoints** trong **ORB** được xác định bằng **moment trung tâm** của vùng **keypoints**, đảm bảo **keypoint** có hướng nhất quán bất kể xoay ảnh.
                    - Độ chính xác của **ORB** có xu hướng giảm nhanh khi góc qua lớn so với **SIFT** vì hướng **keypoints** dựa trên **moment** kém chính xác hơn hướng **gradient** của **SIFT** trong các vùng cường độ phức tạp hoặc nhiễu.
                """)
    example_rotation_orb()
    st.markdown(
                """
                - Độ chính xác của **SIFT** không cao nhưng không có nhiều biến động về độ chính xác khi góc quay thay đổi. Vì:
                    - Vì mỗi **keypoint** có hướng **gradient** riêng, **SIFT** luôn mô tả đặc trưng dưới một hướng chuẩn hóa. Điều này giúp 
                    **SIFT** nhận diện và **matching** tương đối chính xác **keypoints** giữa hai ảnh dù chúng bị xoay với bất kỳ góc độ nào
                    - **Gradient** cục bộ (vốn là thông tin về độ thay đổi cường độ sáng) không bị ảnh hưởng bởi phép xoay.
                """)
    example_rotation_sift()
    st.markdown(
                """
                - **SuperPoint** có độ chính xác cao nhất trong ba thuật toán, đặc biệt là khi hình ảnh có biến đổi về góc quay. Vì:
                    - **SuperPoint** sử dụng mạng **nơ-ron tích chập (CNN)** để tự động học các đặc trưng từ dữ liệu, nhờ đó có khả năng phát hiện và mô tả **keypoints** một cách linh hoạt. 
                    Điều này giúp **SuperPoint** có thể nhận diện các đặc trưng ổn định dù hình ảnh bị biến đổi phức tạp.
                    - **SuperPoint** được huấn luyện với dữ liệu lớn chứa nhiều trường hợp biến đổi về góc, tỷ lệ và độ sáng, do đó có thể tổng quát hóa tốt hơn cho các điều kiện thực tế. Nhờ vậy, 
                    **SuperPoint** đạt độ chính xác cao hơn khi so sánh với **SIFT** và **ORB**, nhất là trong các tình huống khó khăn.
                    - **SuperPoint** tạo **descriptor** không chỉ dựa trên pixel, mà còn dựa vào các đặc trưng cấp cao của ảnh nhờ các tầng **CNN**. Điều này giúp **descriptor** của **SuperPoint** mạnh mẽ và chính xác hơn.
                """)
    example_rotation_superpoint()
    
def App():
    tab = st.tabs(["**Sematic Keypoint Detection**", "**Keypoint Matching**"])
    with tab[0]:
        Text_of_App()
    with tab[1]:
        Text_of_Superpoint_rotation()
        # extract_superpoint_keypoint_and_descriptor()
        
App()
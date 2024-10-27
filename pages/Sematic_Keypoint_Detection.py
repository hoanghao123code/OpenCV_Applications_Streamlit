from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os
import pickle
import matplotlib.pyplot as plt

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
    path_dataset = './images/SIFT_SURF_ORB/synthetic_shapes_datasets/synthetic_shapes_datasets/'
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
    st.image(image)
    return image

def plot_metric_precision(precision_SIFT, precision_ORB, c):
    categories = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple_polygons', 'polygon', 'star', 'stripes']
    values1 = np.array(precision_SIFT)
    values2 = np.array(precision_ORB)
    
    x = np.arange(len(categories))  
    width = 0.35 

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values1, width, label='Precision of SIFT')
    rects2 = ax.bar(x + width/2, values2, width, label='Precision of ORB')

    ax.set_ylabel('Average Precision')
    ax.set_title('Biểu đồ so sánh Average Precision khi áp dụng thuật toán SIFT và ORB')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation = 90)
    ax.legend()

    # Hiển thị biểu đồ trong Streamlit
    c.pyplot(fig)
    
def plot_metric_recall(recall_SIFT, recall_ORB, c):
    categories = ['checkerboard', 'cube', 'ellipses', 'lines', 'multiple_polygons', 'polygon', 'star', 'stripes']
    values1 = np.array(recall_SIFT)
    values2 = np.array(recall_ORB)
    
    x = np.arange(len(categories))  
    width = 0.35 

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values1, width, label='Recall of SIFT')
    rects2 = ax.bar(x + width/2, values2, width, label='Recall of ORB')

    ax.set_ylabel('Average Recall')
    ax.set_title('Biểu đồ so sánh Average Recall khi áp dụng thuật toán SIFT và ORB')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation = 90)
    ax.legend()

    # Hiển thị biểu đồ trong Streamlit
    c.pyplot(fig)


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
def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv.KeyPoints
    keypoints = [cv.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


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


def preprocess_image(img_file, img_size):
    img = cv.imread(img_file, cv.IMREAD_COLOR)
    img = cv.resize(img, img_size)
    img_orig = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig



def compare_and_draw_sift_match(image_1, image_2):
    # Compare sift feature
    sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(image_1)
    sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(image_2)
    
    sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
                sift_kp1, sift_desc1, sift_kp2, sift_desc2)
    if sift_m_kp1 is None and sift_m_kp2 is None and sift_matches is None:
        return image_1, 0.0
    if len(sift_m_kp1) < 4 or len(sift_m_kp2) < 4:
        # print(type(sift_m_kp1))
        return image_1, 0.0
    sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
    # Draw sift feature
    sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
    sift_matched_img = cv.drawMatches(image_1, sift_kp1, image_2,
                                        sift_kp2, sift_matches, None,
                                        matchColor=(0, 255, 0),
                                        singlePointColor=(0, 0, 255))
    accuracy = len(sift_matches) / len(sift_kp1)
    return sift_matched_img, accuracy
def compare_and_draw_ORB_match(image_1, image_2):
    # Compare sift feature
    sift_kp1, sift_desc1 = extract_ORB_keypoints_and_descriptors(image_1)
    sift_kp2, sift_desc2 = extract_ORB_keypoints_and_descriptors(image_2)
    sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
                sift_kp1, sift_desc1, sift_kp2, sift_desc2)
    if sift_m_kp1 is None and sift_m_kp2 is None and sift_matches is None:
        return image_1, 0.0
    if len(sift_m_kp1) < 4 or len(sift_m_kp2) < 4:
        # print(type(sift_m_kp1))
        return image_1, 0.0
    sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
    # Draw sift feature
    sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
    sift_matched_img = cv.drawMatches(image_1, sift_kp1, image_2,
                                           sift_kp2, sift_matches, None,
                                           matchColor=(0, 255, 0),
                                           singlePointColor=(0, 0, 255))
    accuracy = len(sift_matches) / len(sift_kp1)
    return sift_matched_img, accuracy

def plot_compare_match():
    # lst_image = get_image_with_100_percent()
    # rotate = [0, 10, 20, 30, 40]
    # lst_acc_sift = [[] for i in range(len(rotate))]
    # lst_acc_orb = [[] for i in range(len(rotate))]
    # for i in range(len(lst_image)):
    #     for j in range(len(rotate)):    
    #         image_1 = lst_image[i]
    #         image_2 = rotate_image(image_1, rotate[j])
    #         image_sift, acc_sift = compare_and_draw_sift_match(image_1, image_2)
    #         image_orb, acc_orb = compare_and_draw_ORB_match(image_1, image_2)
    #         if acc_sift != 0.0 and acc_orb != 0.0:
    #             lst_acc_sift[j].append(acc_sift)
    #             lst_acc_orb[j].append(acc_orb)
    # average_acc_sift = []
    # average_acc_orb = []
    # for i in range(len(rotate)):
    #     average_acc_sift.append(sum(lst_acc_sift[i]) / len(lst_acc_sift[i]))
    #     average_acc_orb.append(sum(lst_acc_orb[i]) / len(lst_acc_orb[i]))
    # # print(average_acc_sift[0], average_acc_orb[0])
    pickle_file_average_acc_sift = './data_processed/Semantic_Keypoint_Detection/avg_acc_sift.pkl'
    # with open(pickle_file_average_acc_sift, 'wb') as file:
    #     pickle.dump(average_acc_sift, file)
    
    pickle_file_average_acc_orb = './data_processed/Semantic_Keypoint_Detection/avg_acc_orb.pkl'
    # with open(pickle_file_average_acc_orb, 'wb') as file:
    #     pickle.dump(average_acc_orb, file)
    average_acc_sift = []
    average_acc_orb = []
    with open(pickle_file_average_acc_sift, 'rb') as file:
        average_acc_sift = pickle.load(file)
    
    with open(pickle_file_average_acc_orb, 'rb') as file:
        average_acc_orb = pickle.load(file)
    degree_symbol = "\u00B0"
    categories = [f'0{degree_symbol}', f'10{degree_symbol}', f'20{degree_symbol}', f'30{degree_symbol}', f'40{degree_symbol}']
    values1 = np.array(average_acc_sift)
    values2 = np.array(average_acc_orb)
    
    x = np.arange(len(categories))  
    width = 0.35 

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values1, width, label='Accuracy of SIFT match')
    rects2 = ax.bar(x + width/2, values2, width, label='Accuracy of ORB match')

    ax.set_ylabel('Average Accuracy')
    ax.set_title('Biểu đồ so sánh Average Accuracy khi áp dụng thuật toán SIFT và ORB')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    c = st.columns([2, 6, 2])
    c[1].pyplot(fig)

def get_image_with_100_percent():
    # lst_image = []
    # lst_label = []
    # with open('./data_processed/Semantic_Keypoint_Detection/lst_image.pkl', 'rb') as file:
    #     lst_image = pickle.load(file)
    # with open('./data_processed/Semantic_Keypoint_Detection/lst_label.pkl', 'rb') as file:
    #     lst_label = pickle.load(file)
    lst_image, lst_label = get_image_and_label()
    lst_best_image = []
    for i in range(len(lst_image)):
        for j in range(len(lst_image[i])):
            keypoints, _, _ = SIFT_result(lst_image[i][j])
            
            keypoints_ORB, _, image = ORB_result(lst_image[i][j])
            
            precision_sift = calculate_precision(keypoints, lst_label[i][j], threshold=4)
            recall_sift = calculate_recall(keypoints, lst_label[i][j], threshold=4)
            
            precision_orb = calculate_precision(keypoints_ORB, lst_label[i][j], threshold=4)
            recall_orb = calculate_recall(keypoints_ORB, lst_label[i][j], threshold=4)
            if (precision_sift == 1.0) or (precision_orb == 1.0) or (recall_sift == 1.0) or (recall_orb == 1.0):
                lst_best_image.append(lst_image[i][j])
    return lst_best_image

def result_of_match():
    # lst_image = get_image_with_100_percent()
    # pickle_image_100_percent = './data_processed/Semantic_Keypoint_Detection/image_100_percent.pkl'
    # with open(pickle_image_100_percent, 'wb') as file:
    #     pickle.dump(lst_image, file)
    # angel = [0, 10, 20, 30, 40]
    # result_image_sift = []
    # result_image_orb = []
    # for i in range(len(angel)):
    #     image_1 = lst_image[2]
    #     image_2 = rotate_image(image_1, angel[i])
    #     image_sift, acc_sift = compare_and_draw_sift_match(image_1, image_2)
    #     image_orb, acc_orb = compare_and_draw_ORB_match(image_1, image_2)
    #     result_image_sift.append((image_sift, acc_sift))
    #     result_image_orb.append((image_orb, acc_orb))
    #     c[0].image(image_sift)
    #     c[0].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_sift:.2f}</b></div>", unsafe_allow_html=True)
        
    #     c[1].image(image_orb)
    #     c[1].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_orb:.2f}</b></div>", unsafe_allow_html=True)

        
    # for i in range(len(angel)):
    #     image_1 = lst_image[861]
    #     image_2 = rotate_image(image_1, angel[i])
    #     image_sift, acc_sift = compare_and_draw_sift_match(image_1, image_2)
    #     image_orb, acc_orb = compare_and_draw_ORB_match(image_1, image_2)
    #     result_image_sift.append((image_sift, acc_sift))
    #     result_image_orb.append((image_sift, acc_orb))
    #     c[3].image(image_sift)
    #     c[3].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_sift:.2f}</b></div>", unsafe_allow_html=True)
        
    #     c[4].image(image_orb)
    #     c[4].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_orb:.2f}</b></div>", unsafe_allow_html=True)
    pickle_sift_match_ex = './data_processed/Semantic_Keypoint_Detection/sift_match_example.pkl'
    # with open(pickle_sift_match_ex, 'wb') as file:
    #     pickle.dump(result_image_sift, file)   
    
    pickle_orb_match_ex = './data_processed/Semantic_Keypoint_Detection/orb_match_example.pkl'
    # with open(pickle_orb_match_ex, 'wb') as file:
    #     pickle.dump(result_image_orb, file)  
    
    degree_symbol = "\u00B0"
    angel = [0, 10, 20, 30, 40]
    result_image_sift = []
    result_image_orb = []
    with open(pickle_sift_match_ex, 'rb') as file:
        result_image_sift = pickle.load(file)
    with open(pickle_orb_match_ex, 'rb') as file:
        result_image_orb = pickle.load(file)
    n = len(angel)
    for i in range(n):
        c = st.columns([3, 3, 1, 3, 3])
        image_1, acc_1 = result_image_sift[i]
        image_3, acc_3 = result_image_sift[i + n]
        
        image_2, acc_2 = result_image_orb[i]
        image_4, acc_4 = result_image_orb[i + n]
        c[0].image(image_1)
        c[0].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_1:.2f}</b></div>", unsafe_allow_html=True)
        
        c[1].image(image_2)
        c[1].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_2:.2f}</b></div>", unsafe_allow_html=True)

        c[2].markdown(
            f"""
            <div style='display: flex; align-items: center; justify-content: center; height: 100%; font-weight: bold;'>
                {str(angel[i])} {degree_symbol}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        c[3].image(image_3)
        c[3].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_3:.2f}</b></div>", unsafe_allow_html=True)
        
        c[4].image(image_4)
        c[4].markdown(f"<div style='text-align: center;'><b>Accuracy = {acc_4:.2f}</b></div>", unsafe_allow_html=True)
        


def Text_of_App():
    st.header("1. Giới thiệu Synthetic shapes datasets")
    st.write("Dataset **Synthetic shapes datasets** gồm $8$ class ảnh về hình học bao gồm ảnh và tọa độ các keypoint của từng ảnh như:")
    st.write("  -  **Draw checkerboard, Draw cube, Draw ellipses, Draw lines, Draw multiple polygon, Draw polygon, Draw star và Draw stripes**")
    st.write("  - Mỗi class có $500$ ảnh và tổng số ảnh trong dataset là $4000$ ảnh")
    st.write("**Một số ảnh trong Dataset và các keypoint tương ứng**")
    path_dataset = './images/SIFT_SURF_ORB/dataset_with_keypoint.PNG'
    image_dataset = cv.imread(path_dataset)
    c = st.columns([2, 6, 2])
    c[1].image(image_dataset,channels="BGR")
    st.header("2. Phương pháp")
    st.markdown("### 2.1 SIFT")
    
    st.markdown("#### 2.1.1 Giới thiệu về thuật toán SIFT" )
    st.write("Thuật toán **SIFT (Scale-Invariant Feature Transform)** phát hiện và mô tả các điểm đặc trưng **(keypoints)** trong ảnh một cách không thay đổi trước biến đổi tỷ lệ, góc quay, và cường độ ánh sáng")
    st.write("Thuật toán **SIFT** được phát triển bởi **David Lowe**, và bài báo gốc mô tả **SIFT** là:")
    st.write("  - **Lowe, David G. ""Distinctive image features from scale-invariant keypoints."" International Journal of Computer Vision, 2004.**")
    st.write(" Bài báo này được trích dẫn rộng rãi và là nền tảng cho nhiều ứng dụng và nghiên cứu về thị giác máy tính.")
    st.markdown("#### 2.1.2 Thuật toán SIFT")
    c = st.columns(2)
    with c[0]:
        st.markdown(
                """
                Các bước chính của thuật toán SIFT:
                1. **Phát hiện điểm đặc trưng:** Sử dụng **Difference of Gaussian (DoG)** trên các phiên bản ảnh với nhiều mức tỷ lệ để tìm điểm cực trị.
                2. **Lọc điểm yếu:** Loại bỏ các điểm không ổn định.
                3. **Xác định hướng:** Tính toán góc gradient để đảm bảo không thay đổi đối với việc xoay ảnh.
                4. **Tạo descriptor:** Mô tả điểm dựa trên gradient cường độ xung quanh.
                5. **So khớp đặc trưng:** Dùng khoảng cách giữa các **descriptor** để ghép điểm từ các ảnh khác nhau.
                """)
    with c[1]:
        st.write("Dưới đây là hình ảnh minh họa thuật toán **SIFT**:")
        st.image('./images/SIFT_SURF_ORB/sift_algorith.png', channels="BGR", width=500)
    st.write("Duới đây là kết quả của một số ảnh khi áp dụng thuật toán **SIFT**")
    c = st.columns([2, 6, 2])
    c[1].image('./images/SIFT_SURF_ORB/result_of_SIFT.PNG', channels="BGR")

    st.markdown("### 2.2 ORB")
    st.markdown("#### 2.2.1 Giới thiệu về thuật toán ORB")
    st.write("**ORB (Oriented FAST and Rotated BRIEF)** là thuật toán phát hiện và mô tả đặc trưng hình ảnh, "
            + "được phát triển bởi các nhà nghiên cứu **Ethan Rublee, Vincent Rabaud, Kurt Konolige, và Gary R. Bradski** và được giới thiệu lần đầu tiên vào năm $2011$ trong bài báo sau:")
    st.write("  - **Rublee, Ethan, et al. ""ORB: An efficient alternative to SIFT or SURF."" 2011 International Conference on Computer Vision (ICCV). IEEE, 2011.**")
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
        st.write("Dưới đây là hình ảnh minh họa thuật toán ORB")
        st.image('./images/SIFT_SURF_ORB/achitecture_of_ORB.png', channels="BGR", width=480)
    st.write("Duới đây là kết quả của một số ảnh khi áp dụng thuật toán **ORB**")
    c = st.columns([2, 6, 2])
    c[1].image('./images/SIFT_SURF_ORB/result_of_ORB.PNG', channels="BGR")

    st.header("3. Đánh giá")
    st.write("  - Tiến hành đánh giá trên 2 độ đo **Precision** và **Recall** khi áp dụng **SIFT và ORB**")
    c1, c2, c3 = st.columns([1, 8, 1])
    c2.image('./images/SIFT_SURF_ORB/precision_and_recall.png', channels="BGR", width=500)
    st.markdown("   - **Keypoint** đó được cho là dự đoán đúng nếu khoảng cách **Euclidean** của **Keypoint** đó so với khoảng cách của **Keypoint** thực tế <= **Threshold**")
    st.markdown("   - Công thức khoảng cách **Euclidean:**")
    c = st.columns([2, 6, 2])
    c[1].image('./images/SIFT_SURF_ORB/euclidean.png',channels="BGR", width=300)
    st.markdown("   - **Threshold**: 4")
    st.header("4. Kết quả")
    st.markdown("Dưới đây lần lượt là 2 biểu đồ so sánh **Precision** và **Recall** của Thuật toán **SIFT** và **ORB**")
    plot_metric()
    st.header("5. Thảo luận")
    st.markdown("**Nhận xét tổng quan:**")
    st.write("  - **ORB** nhìn chung có **Precision** và **Recall** cao hơn cho các hình dạng có đặc trưng nổi bật, dễ phát hiện và phân biệt.")
    st.write("  - **SIFT** lại hoạt động tốt hơn trên các hình dạng có chi tiết đơn giản hoặc đều đặn, như **Lines** và **Stripes**.")
    st.markdown("**Nhận xét và giải thích:**")
    st.write("  - **ORB** có độ chính xác và độ bao phủ tốt hơn cho một số hình dạng có đặc trưng phân biệt rõ ràng như **Checkerboard, Cube, Polygon**, và **Star**, do **ORB** tối ưu cho việc phát hiện đặc trưng nhanh "
             + "và ít chịu ảnh hưởng từ thay đổi góc xoay. Do đó **Precision** và **Recall** của **ORB** cao hơn **SIFT**")
    st.write("  - **SIFT** hoạt động tốt hơn trên các hình dạng đơn giản, tuần hoàn như **Lines** và **Stripes** vì nó có cách tiếp cận chi tiết trong việc phát hiện đặc trưng, phù hợp với những chi tiết nhỏ của hình dạng này nên "
             + "**Precision** và **Recall** của **SIFT** cao hơn **ORB**")
    st.write("  - **Ellipses** và **Multiple Polygons**: Cả hai thuật toán đều có **Precision** và **Recall** thấp cho các hình dạng này, do chúng có ít đặc trưng nổi bật hoặc quá phức tạp để các thuật toán này dễ dàng phát hiện.")


def Text_of_Superpoint_rotation():
    dg = "\u00B0"
    st.header("1. Thiết lập thí nghiệm")
    st.markdown("""
                - Tiến hành thí nghiệm đối với những ảnh trong tập **Synthetic Shapes Dataset** mà **SIFT** hoặc **ORB** đạt **100%** về phát hiện **Keypoints** (theo **Ground Truth**)
                    - Số lượng ảnh tìm được: $1147$ ảnh 
                """)
    
    st.markdown(f"""
                - Thực hiện thí nghiệm đánh giá **SIFT, ORB** trên tiêu chí **rotation** (góc quay **0{dg}, 10{dg}, 20{dg}, 30{dg}, 40{dg}**) để đánh giá mức độ **matching keypoints** của 2 phương pháp trên tập dữ liệu vừa tìm được ở trên.
                    - Sử dụng độ đo để đánh giá: **Accuracy**
                """)
    c = st.columns([2, 6, 2])
    c[1].image('./images/SIFT_SURF_ORB/accuracy.png', channels="BGR", width=400)
    
    st.header("2. Kết quả")
    st.write(f" - Dưới đây là một số hình ảnh **matching keypoints** của 2 hình ảnh khi 1 ảnh giữ nguyên và 1 ảnh xoay một góc **0{dg}, 10{dg}, 20{dg}, 30{dg}, 40{dg}** của thuật toán **SIFT** và **ORB**")
    c = st.columns([3, 3, 1, 3, 3])
    c[0].markdown(f"<div style='text-align: center;'><b>SIFT</b></div>", unsafe_allow_html=True)
    c[1].markdown(f"<div style='text-align: center;'><b>ORB</b></div>", unsafe_allow_html=True)
    c[2].markdown(f"<div style='text-align: center;'><b>Rotation</b></div>", unsafe_allow_html=True)
    c[3].markdown(f"<div style='text-align: center;'><b>SIFT</b></div>", unsafe_allow_html=True)
    c[4].markdown(f"<div style='text-align: center;'><b>ORB</b></div>", unsafe_allow_html=True)
    result_of_match()
    st.write("  - Duới đây là biểu đồ biểu diễn **Average Accuracy** của khi áp dụng thuật toán **SIFT** và **ORB**")
    plot_compare_match()
    st.header("3. Thảo luận")
    st.markdown("#### 3.1 Nhận xét")
    st.write("  - Độ chính xác của cả hai thuật toán giảm đáng kể khi góc quay tăng.")
    st.write("  - **SIFT**: Mặc dù giảm nhưng vẫn giữ được độ chính xác cao hơn **ORB** trong tất cả các góc quay.")
    st.write("  - **ORB**: Độ chính xác giảm nhanh hơn so với **SIFT** khi góc quay tăng, thể hiện rằng **ORB** có thể nhạy cảm hơn với các góc quay lớn.")
    st.markdown("#### 3.2 Giải thích")
    st.markdown("""
                - **SIFT** có khả năng chịu được sự thay đổi góc quay tốt hơn ORB, điều này có thể do **SIFT** không chỉ dựa vào các đặc trưng về cường độ 
                mà còn sử dụng **gradient** hướng để xác định **keypoint**, từ đó giúp duy trì độ ổn định khi có sự thay đổi góc quay.
                """)
    st.markdown("""
                - **ORB** có xu hướng kém ổn định hơn khi có sự thay đổi về góc quay, điều này làm giảm độ chính xác của nó nhanh hơn so với **SIFT**.
                """
                )
def App():
    tab = st.tabs(["**Sematic Keypoint Detection**", "**Superpoint - Rotation**"])
    with tab[0]:
        Text_of_App()
    with tab[1]:
        Text_of_Superpoint_rotation()
App()
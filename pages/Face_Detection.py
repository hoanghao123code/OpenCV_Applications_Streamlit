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

st.title("🎈Face Detection App")

# cascade_file = './images/Face_detect/cascade.xml'
cascade_file = './images/Face_detect/cascade.xml'

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
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (24, 24))
    ii = cv.integral(image)
    val = 0
    for rect in haar_features:
        x, y, w, h, weight = rect
        val += weight * (ii[y + h][x + w] + ii[y][x] - ii[y + h][x] - ii[y][x + w])
    return val


# X_train = []

# def extract_image_dataset():
#     for i in range(len(face_dataset)):
#         value = extract_feature_image(face_dataset[i])
#         X_train.append(value)
# extract_image_dataset()
# print(X_train)
# y_train = labels

# X_train = np.array(X_train)

# pickle_file = 'D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Train_test\X_train.pkl'
# with open(pickle_file, 'wb') as file:
#     pickle.dump(X_train, file)

# pickle_file_y = "D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Train_test\y_train.pkl"
# with open(pickle_file_y, 'wb') as file:
#     pickle.dump(y_train, file)


X_train = []
y_train = []
with open('./images/Train_test/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('./images/Train_test/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)
    
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
# print(X_train.shape, y_train.shape)
model = KNeighborsClassifier(n_neighbors = 20)
model.fit(X_train, y_train)

def detect_face_Sub_window(image, model):
    sz = 50
    step = image.shape[0] // 20
    lst_rect = []
    for x in range(0, image.shape[1] - sz, step):
        for y in range(0, image.shape[0] - sz, step):
            sub_window = image[y : y + sz, x : x + sz]
            feature_sub = extract_feature_image(sub_window)
            predictions = model.predict(np.array([[feature_sub]]))
            if predictions[0] == 1:
                lst_rect.append((x, y, sz, sz))
    return lst_rect
    

def IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_max = max(x1, x2)
    x_min = min(x1 + w1, x2 + w2)
    y_max = max(y1, y2)
    y_min = min(y1 + h1, y2 + h2)
    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area_1 = w1 * h1
    area_2 = w2 * h2
    union_area = area_1 + area_2 - intersect
    return intersect / union_area if union_area != 0 else 0 
    
# Non Maximum Suppression

def NMS(boxes, Iou_threshold):
    choose_boxes = []
    
    boxes = sorted(boxes, key = lambda box : box[2] * box[3], reverse = True)
    while boxes:
        cur_box = boxes.pop(0)
        choose_boxes.append(cur_box)
        
        boxes = [box for box in boxes if IoU(cur_box, box) < Iou_threshold] 
    return choose_boxes


def IoU_metric(mask_pred, mask_gt):
    # mask_pred = [mask_pred > 0].astype(np.uint8)
    # mask_gt = [mask_gt > 0].astype(np.uint8)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0.0:
        return 0.0
    iou = intersection / union
    return iou

def Dataset_and_Training():
    st.markdown("### 1. Giới thiệu Dataset")
    st.markdown("#### 1.1. Tập Train")
    st.write("Tập train gồm 800 ảnh trong đó có 400 ảnh có chứa khuôn mặt và 400 ảnh không chứa khuôn mặt")
    image_face_dataset = cv.imread('./images/Face_Detection/face_datasest.PNG')
    st.image(image_face_dataset, caption="Một số ảnh chứa khuôn mặt trong tập train", channels="BGR")
    
    image_non_face = cv.imread('./images/Face_Detection/non_face_dataset.PNG')
    st.image(image_non_face, caption="Một số ảnh không chứa khuôn mặt trong tập train", channels="BGR")
    st.markdown("#### 1.2 Tập Test")
    st.write("Tập test sử dụng từ tập dữ liệu ngoài (google) được detect bằng Cascade classifier của OpenCV")
    image_test = cv.imread('./images/Face_Detection/Test_image.PNG')
    st.image(image_test, caption="Ảnh của tập test đã được detect", channels="BGR")
    st.markdown("### 2. Quá trình huấn luyện với Cascade Classifier")
    st.markdown("##### **2.1 Các tham số trong quá trình huấn luyện**")
    st.write(" - **numPos : 400** (Số lượng mẫu **Positive** (chứa khuôn mặt) được dùng trong huấn luyện cho mỗi stage)")
    st.write(" - **numNeg : 400** (Số lượng mẫu **Negative** (không chứa khuôn mặt) được dùng trong huấn luyện cho mỗi stage)")
    st.write(" - **numStages : 5** (Số lượng Cascade stages được train)")
    st.write(" - **w : 24, h : 24** (Lần lượt là chiều rộng và chiều cao của object)")
    st.write(" - **minHitRate : 0.995** (Ít nhất 99.5% các mẫu **Positive** phải được phát hiện đúng (không bỏ sót). Giai đoạn huấn luyện sẽ tiếp tục cho đến khi đạt được tỉ lệ này)")
    st.write(" - **maxFalseAlarmRate : 0.5:** (Trong mỗi giai đoạn huấn luyện, tỉ lệ phát hiện nhầm các mẫu **Negative** (nhận nhầm là **Positive**) phải dưới 50%)")
    st.markdown("### 3. Huấn luyện với KNN và đánh giá")
    st.markdown("##### **3.1 Độ đo: IoU**")
    image_IoU =  cv.imread('./images/image_IoU.png')
    st.image(image_IoU, channels="BGR", width=350)
    st.markdown("##### 3.2 Tiến hành đánh giá với các giá trị K (trong KNN) để tìm ra giá trị tốt nhất")
    st.write("**- K = [1, 2, 3, ... 50]**")
    Plot_IoU()
    st.markdown("##### 3.3 Kết quả khi áp dụng vào tập Test")
    Result_of_Test()
    

def Plot_IoU():
    # haar_cascade = cv.CascadeClassifier('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\haarcascade_frontalface_default.xml')
    
    # lst_dir = os.listdir('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Face_Detection\Test')
    # lst_IoU = []
    # K = np.arange(1, 50, 1)
    # for k in K:
    #     model_k = KNeighborsClassifier(n_neighbors = k)
    #     model_k.fit(X_train, y_train)
    #     average_IoU = 0.0
    #     for i in range(len(lst_dir)):
    #         image = cv.imread('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Face_Detection\Test' + "\\" + lst_dir[i])
    #         image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            
    #         # Load rect của haar
    #         faces_rect = haar_cascade.detectMultiScale(image_gray, 1.1, 9)
            
    #         # Load rect khi pred với KNN
    #         faces_rect_KNN = detect_face_Sub_window(image, model_k)
    #         faces_rect_KNN = NMS(faces_rect_KNN, float(0.15))
            
    #         image_gray2 = image_gray.copy()
    #         for (x, y, w, h) in faces_rect:
    #             image_gray = cv.rectangle(image_gray, (x, y), (x + w, y + h), 1, -1)
                
    #         for (x, y, w, h) in faces_rect_KNN:
    #             image_gray2 = cv.rectangle(image_gray2, (x, y), (x + w, y + h), 1, -1)
    #         # Chuyển ground truth thành ảnh nhị phân
    #         image_gt = image_gray.copy()
    #         image_gt[image_gray != 1] = 0
            
    #         # Chuyển pred thành ảnh nhị phân
    #         image_pred = image_gray2.copy()
    #         image_pred[image_gray2 != 1] = 0
    #         # print(IoU_metric(image_pred, image_gt))
    #         # lst_IoU.append(IoU_metric(image_pred, image_gt))
    #         average_IoU += IoU_metric(image_pred, image_gt)
    #     lst_IoU.append(average_IoU / 10.0)
    # # print(lst_IoU)
    # lst_IoU = np.array(lst_IoU)
    # fig1, ax = plt.subplots()
    # ax.plot(K, lst_IoU)

    # ax.set_xlabel('Tham số K trong KNN')
    # ax.set_ylabel('Average IoU')    
    # ax.set_title('Biểu đồ average IoU theo các giá trị K khác nhau')  
    # ax.legend()
    # st.pyplot(fig1)
    # best_IoU = max(lst_IoU)
    # id = np.where(lst_IoU == best_IoU)
    # print(best_IoU, K[id[0][0]])
    image_IoU = cv.imread('./images/Face_Detection/image_IoU.PNG')
    st.image(image_IoU, channels="BGR")
    st.markdown("##### * Kết quả sau khi huấn luyện:")
    st.write(" - Tham số K tốt nhất là **K = 20** với **Average IoU = 0.23**")
    
def Result_of_Test():
    # haar_cascade = cv.CascadeClassifier('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\haarcascade_frontalface_default.xml')
    # lst_dir = os.listdir('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Face_Detection\Test')
    
    # for i in range(len(lst_dir)):
    #     image = cv.imread('D:\OpenCV\Grabcut\Grabcut_Streamlit\images\Face_Detection\Test' + "\\" + lst_dir[i])
    #     image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     image_gray2 = image_gray.copy()
    #     face_rect = haar_cascade.detectMultiScale(image_gray, 1.1, 9)
    #     face_rect_KNN = detect_face_Sub_window(image, model)
    #     face_rect_KNN = NMS(face_rect_KNN, float(0.15))
    #     for (x, y, w, h) in face_rect:
    #         image_gray = cv.rectangle(image_gray, (x, y), (x + w, y + h), 1, -1)
    #     for (x, y, w, h) in face_rect_KNN:
    #         image_gray2 = cv.rectangle(image_gray2, (x, y), (x + w, y + h), 1, -1)
        
    #     image_gt = image_gray.copy()
    #     image_gt[image_gray != 1] = 0
        
    #     image_pred = image_gray2.copy()
    #     image_pred[image_gray2 != 1] = 0
    #     st.write(lst_dir[i])
    #     st.write(IoU_metric(image_pred, image_gt))
        # st.image(image, channels="BGR")
    image_res = cv.imread('./images/Face_Detection/Result/Result_of_All.PNG')
    if image_res is not None:
        st.image(image_res, caption="Kết quả sau khi áp dụng tham số K tốt nhất vào tập Test", channels="BGR")

def Load_Image_and_Process():
    st.markdown("### 4. Phát hiện khuôn mặt")
    
    st.markdown("#### Chọn ảnh bạn cần phát hiện khuôn mặt")
    image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if image_upload is not None:
        if not os.path.exists('images'):
            os.makedirs('images')
        image = Image.open(image_upload)
        image.save('images/' + image_upload.name)
        img = cv.imread('images/' + image_upload.name)
        img_copy = img.copy()
        if img is not None and len(img.shape) == 3:
            faces_rect = detect_face_Sub_window(img, model)
            faces_rect = NMS(faces_rect, float(0.15))
            for (x, y, w, h) in faces_rect:
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(img, channels="BGR")
def App():
    Dataset_and_Training()
    Load_Image_and_Process()
App()


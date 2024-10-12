import streamlit as st
import firebase_admin
import json, toml
from firebase_admin import credentials
from google.cloud import firestore, storage
from PIL import Image


import cv2 as cv
import pandas as pd
import numpy as np
import unicodedata
import re
import tempfile
import sys
import os
import argparse

sys.path.append("D:\OpenCV\Grabcut\Grabcut_Streamlit\services") 
# from services.face_verification.yunet import YuNet
# from services.face_verification.sface import SFace
from face_verification.yunet import YuNet
from face_verification.sface import SFace

st.title("🎈Face Verification App")

# Khởi tạo Firestore Client bằng credentials từ file JSON
db = firestore.Client.from_service_account_info(st.secrets)


bucket = storage.Client.from_service_account_info(st.secrets).get_bucket('face-detection-2024.appspot.com')

lst_folder = ['HoangHao', 'NgoVanHai', 'TruongDoan', 'NguyenPhuocBinh', 'NguyenVuHoangChuong', 'TranThiThanhHue', 'LeBaNhatMinh']


def get_url_Image(path):
    blob = bucket.blob(path)
    blob.make_public()
    image_path = blob.public_url
    url = f"<img src = '{image_path}' width='100'>"
    return url

# def List_folder():
#     for i in range(len(lst_folder)):
#         blobs = bucket.list_blobs(prefix = lst_folder[i])
#         file_list = [blob.name for blob in blobs]
#         public_url1 = read_Image(file_list[1])
#         lst_ChanDung.append(f"<img src='{public_url1}' width='100'>")
#         if len(file_list) <= 2:
#             lst_TheSV.append(f"<img src='{public_url1}' width='100'>")
#         else:
#             public_url2 = read_Image(file_list[2])
#             lst_TheSV.append(f"<img src='{public_url2}' width='100'>")
#         # print(list(blobs.prefixes))
# st.cache_data(ttl="2h")
def get_Info():
    lst_Ten = []
    lst_Masv = []
    lst_ChanDung = []
    lst_TheSV = []
    doc = db.collection('1').get()
    lent = len(doc)
    for i in range(1, lent + 1, 1):
        doc_ref = db.collection("1").document(str(i))
        doc = doc_ref.get()
        doc_data = doc.to_dict()
        Ten = doc_data.get('Ten')
        Masv = doc_data.get('Ma sinh vien')
        Nganh = doc_data.get('Nganh')
        url_ChanDung = doc_data.get('url_ChanDung')
        url_TheSV = doc_data.get('url_TheSV')
        lst_Ten.append(Ten)
        lst_Masv.append(Masv)
        # lst_Nganh.append(Nganh)
        lst_ChanDung.append(url_ChanDung)
        lst_TheSV.append(url_TheSV)
    return lst_Ten, lst_Masv, lst_ChanDung, lst_TheSV

def Table_of_Data():
    doc = db.collection('1').get()
    lent = len(doc)
    lst_STT = np.arange(1, lent + 1, 1)
    lst_Ten, lst_Masv, lst_ChanDung, lst_TheSV = get_Info()
    data = {
    "STT": lst_STT,
    "Tên" : lst_Ten,
    "Mã sinh viên": lst_Masv,
    # "Ngành": lst_Nganh,
    "Ảnh chân dung": lst_ChanDung,
    "Ảnh thẻ sv" : lst_TheSV
    }
    df = pd.DataFrame(data)
    # st.dataframe(df)
    # st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    html = df.to_html(escape=False, index=False)
    st.write(html, unsafe_allow_html=True)

def add_Image_url():
    doc = db.collection('1').get()
    lent = len(doc)
    for i in range(lent):
        blobs = bucket.list_blobs(prefix = lst_folder[i])
        file_list = [blob.name for blob in blobs]
        public_url1 = get_url_Image(file_list[1])
        doc_ref = db.collection('1').document(str(i + 1))
        public_url2 = public_url1
        if len(file_list) > 2:
            public_url2 = get_url_Image(file_list[2])
        # print(public_url1, public_url2)
        doc_ref.update({
            'url_ChanDung': public_url1,
            'url_TheSV': public_url2
        })

def remove_accents(input_str):
    # Chuẩn hóa chuỗi về dạng tổ hợp (NFD)
    nfkd_form = unicodedata.normalize('NFD', input_str)
    
    
    # Loại bỏ các ký tự thuộc dạng dấu (Mn - Mark, Nonspacing) bằng biểu thức chính quy
    no_accent_str = re.sub(r'[\u0300-\u036f]', '', nfkd_form)
    
    # Thay thế các ký tự đặc biệt như Đ và đ
    no_accent_str = no_accent_str.replace('Đ', 'D').replace('đ', 'd')
    
    return no_accent_str.lower()
    
def get_url(url):
    if url == "":
        return ""
    match = re.search(r'(http[^\s]+\.(jpg|JPG|png|PNG|jpeg|JPEG))', url)
    return match.group(1)

def disPlay_Info(id):
    doc_ref = db.collection('1').document(str(id))
    doc = doc_ref.get()
    doc_data = doc.to_dict()
    Ten = doc_data.get('Ten')
    Masv = doc_data.get('Ma sinh vien')
    # Nganh = doc_data.get('Nganh')
    url_ChanDung = doc_data.get('url_ChanDung')
    url_TheSV = doc_data.get('url_TheSV')
    c1, c2 = st.columns(2)
    c1.write("Tên: " + Ten)
    c2.write("Mã sv: " + Masv)
    
    url_CD = get_url(url_ChanDung)
    url_TSV = get_url(url_TheSV)
    if url_CD != "":
        c1.write("Ảnh chân dung")
        c1.image(url_CD, width=300)
    else:
        c1.write("Ảnh chân dung: Chưa có ảnh")
    if url_TSV != "":
        c2.write("Thẻ sv")
        c2.image(url_TSV, width=300)
    else:
        c2.write("Thẻ sv: Chưa có ảnh")

def normalize_Name():
    lst_name = []
    lst_ten, a, b, c = get_Info()
    for i in range(len(lst_ten)):
        lst_name.append(remove_accents(lst_ten[i]))
    return lst_name
        
def Add_Student(Ten = "", Masv = "", url_ChanDung = "", url_TheSV = ""):
    data = {
        'Ten' : Ten,
        'Ma sinh vien' : Masv,
        'url_ChanDung' : url_ChanDung,
        'url_TheSV' : url_TheSV
    }
    doc = db.collection('1').get()
    document_id = len(doc) + 1
    document_id = str(document_id)
    doc_ref = db.collection('1').document(document_id).create(data)

def normalize_TheSV(lst_TheSV):
    for i in range(len(lst_TheSV)):
        lst_TheSV[i] = lst_TheSV[i].lower()
    return lst_TheSV


# type = 1: Chân dung, type = 2: Thẻ sv
def Add_url_with_Id(id, public_url, type):
    doc_ref = db.collection('1').document(str(id))
    if type == 1:
        doc_ref.update({
            'url_ChanDung': public_url
            # 'url_TheSV': public_url2
        })
    else:
        doc_ref.update({
            # 'url_ChanDung': public_url,
            'url_TheSV': public_url
        })

def Add_Image(uploaded_file, name_file, id, type):
    if uploaded_file is not None:
    # Lưu ảnh vào tạm thời
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_name = temp_file.name

        # Đặt tên file khi upload
        blob = bucket.blob(f"Add_images/{name_file}")  # Thay 'images/' bằng đường dẫn trong bucket bạn muốn lưu

        # Upload file lên Firebase Storage
        blob.upload_from_filename(temp_file_name)

        # Tạo URL cho file vừa upload
        blob.make_public()

        public_url = blob.public_url
        url = f"<img src = '{public_url}' width='100'>"
        Add_url_with_Id(id, url, type)

def CRUD():
    c1, c2, c3, c4 = st.columns(4)
    
    col_name, col_Masv = st.columns(2)
    
    lst_Ten, lst_Masv, lst_ChanDung, lst_TheSV = get_Info()
    # Tìm kiếm
    if 'search_clicked' not in st.session_state:
        st.session_state.search_clicked = False
    if c1.button('Tìm kiếm'):
        st.session_state.search_clicked = True
    
    if st.session_state.search_clicked:
        
        Input_name = col_name.text_input("Tên")
        Input_Masv = col_Masv.text_input("Mã sinh viên")
        
        Input_name = remove_accents(Input_name)
        Input_Masv = remove_accents(Input_Masv)
        lst_id = []
        lst_name = normalize_Name()
        lst_TheSV = normalize_TheSV(lst_TheSV)
        
        if st.button("Xong"):
        
            if Input_name != "":
                for i in range(len(lst_name)):
                    if Input_name in lst_name[i]:
                        lst_id.append(i + 1)
            if Input_Masv != "":
                for i in range(len(lst_TheSV)):
                    if Input_Masv in lst_TheSV[i]:
                        lst_id.append(i + 1)
            if Input_name == "" and Input_Masv == "":
                lst_id = np.arange(1, len(lst_name) + 1, 1)
            lst_id = np.array(lst_id)
            lst_id = np.unique(lst_id)
            for i in lst_id:
                disPlay_Info(i)
    
    # Thêm
    if 'add_clicked' not in st.session_state:
        st.session_state.add_clicked = False
    if c2.button('Thêm'):
        st.session_state.add_clicked = True
    if st.session_state.add_clicked:
        Input_name = col_name.text_input("Tên")
        Input_Masv = col_Masv.text_input("Mã sinh viên")
        
        AnhChanDung_upload = st.file_uploader("Tải ảnh chân dung", type=["png", "jpg", "jpeg"])
        TheSV_upload = st.file_uploader("Tải ảnh thẻ sv", type=["png", "jpg", "jpeg"])
        if st.button('Xong'):
            if Input_name == "" and Input_Masv == "":
                st.markdown("##### Chú ý: Bạn phải nhập đầy đủ **Tên** và **Mã sinh viên**")
            else:
                Add_Student(Ten=Input_name, Masv=Input_Masv, url_ChanDung="", url_TheSV="")
                id = len(lst_TheSV) + 1
                Name_1 = remove_accents(Input_name)
                Name_1 = Name_1.replace(" ", "") + "AnhChanDung.jpg"
                
                Name_2 = remove_accents(Input_Masv)
                Name_2 = Name_2.replace(" ", "") + "TheSV.jpg"

                Add_Image(AnhChanDung_upload, Name_1, id, 1)
                Add_Image(TheSV_upload, Name_2, id, 2)
    
    # Sửa
    # if 'update_clicked' not in st.session_state:
    #     st.session_state.update_clicked = False
    # if c2.button('Sửa'):
    #     st.session_state.update_clicked = True
    # if st.session_state.update_clicked:

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--target', '-t', type=str,
                    help='Usage: Set path to the input image 1 (target face).')
parser.add_argument('--query', '-q', type=str,
                    help='Usage: Set path to the input image 2 (query).')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Usage: Set model path, defaults to face_recognition_sface_2021dec.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(img1, faces1, img2, faces2, matches, scores, target_size=[512, 512]): # target_size: (h, w)
    out1 = img1.copy()
    out2 = img2.copy()
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR

    # Resize to 256x256 with the same aspect ratio
    padded_out1 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h1, w1, _ = out1.shape
    ratio1 = min(target_size[0] / out1.shape[0], target_size[1] / out1.shape[1])
    new_h1 = int(h1 * ratio1)
    new_w1 = int(w1 * ratio1)
    resized_out1 = cv.resize(out1, (new_w1, new_h1), interpolation=cv.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h1) // 2
    bottom = top + new_h1
    left = max(0, target_size[1] - new_w1) // 2
    right = left + new_w1
    padded_out1[top : bottom, left : right] = resized_out1

    # Draw bbox
    bbox1 = faces1[0][:4] * ratio1
    x, y, w, h = bbox1.astype(np.int32)
    cv.rectangle(padded_out1, (x + left, y + top), (x + left + w, y + top + h), matched_box_color, 2)

    # Resize to 256x256 with the same aspect ratio
    padded_out2 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h2, w2, _ = out2.shape
    ratio2 = min(target_size[0] / out2.shape[0], target_size[1] / out2.shape[1])
    new_h2 = int(h2 * ratio2)
    new_w2 = int(w2 * ratio2)
    resized_out2 = cv.resize(out2, (new_w2, new_h2), interpolation=cv.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h2) // 2
    bottom = top + new_h2
    left = max(0, target_size[1] - new_w2) // 2
    right = left + new_w2
    padded_out2[top : bottom, left : right] = resized_out2

    # Draw bbox
    assert faces2.shape[0] == len(matches), "number of faces2 needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    for index, match in enumerate(matches):
        bbox2 = faces2[index][:4] * ratio2
        x, y, w, h = bbox2.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv.rectangle(padded_out2, (x + left, y + top), (x + left + w, y + top + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        cv.putText(padded_out2, "{:.2f}".format(score), (x + left, y + top - 5), cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    return np.concatenate([padded_out1, padded_out2], axis=1)

def YuNet_and_Sface():
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath='./services/face_verification/face_recognition_sface_2021dec.onnx',
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='./services/face_verification/face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)

    # image1 = st.file_uploader("Tải ảnh chân dung", type=["png", "jpg", "jpeg"])
    # image2 = st.file_uploader("Tải ảnh thẻ sv", type=["png", "jpg", "jpeg"])
    # img1 = cv.imread(image1.name)
    # img2 = cv.imread(image2.name)
    
    # # img1 = cv.imread(args.target)
    # # img2 = cv.imread(args.query)

    # # Detect faces
    # detector.setInputSize([img1.shape[1], img1.shape[0]])
    # faces1 = detector.infer(img1)
    # assert faces1.shape[0] > 0, 'Cannot find a face in {}'.format(args.target)
    # detector.setInputSize([img2.shape[1], img2.shape[0]])
    # faces2 = detector.infer(img2)
    # assert faces2.shape[0] > 0, 'Cannot find a face in {}'.format(args.query)

    # # Match
    # scores = []
    # matches = []
    # for face in faces2:
    #     result = recognizer.match(img1, faces1[0][:-1], img2, face[:-1])
    #     scores.append(result[0])
    #     matches.append(result[1])

    # # Draw results
    # image = visualize(img1, faces1, img2, faces2, matches, scores)

    # # # Save results if save is true
    # # if args.save:
    # #     print('Resutls saved to result.jpg\n')
    # #     cv.imwrite('result.jpg', image)
    # st.image(image)
    
def App():
    st.markdown("#### 1. Thông tin sinh viên")
    get_Info()
    CRUD()
    Table_of_Data()
    YuNet_and_Sface()
    
App()
import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
from google.cloud import firestore

import cv2 as cv
st.title("🎈Thông tin sinh viên CNTT")

# Khởi tạo Firestore Client bằng credentials từ file JSON
db = firestore.Client.from_service_account_json("./images/Private_Key/face-detection-2024-firebase-adminsdk-pevrv-476f6abf74.json")
doc_ref = db.collection("1").document("1")
doc = doc_ref.get()
doc_data = doc.to_dict()

options = ['Hoàng Hào', 'Ngô Văn Hải', 'Trương Đoàn', 'Nguyễn Phước Bình', 'Nguyễn Vũ Hoàng	Chương', 'Trần Thị Thanh Huệ',
           'Lê Bá Nhật Minh', 'Lê Trần Khánh Tùng', 'Lê Minh Tú']
select_options = st.selectbox("Chọn tên người bạn muốn xem thông tin", options)

# Khởi tạo Firebase Admin với cùng một credentials
if not firebase_admin._apps:
    cred = credentials.Certificate('./images/Private_Key/face-detection-2024-firebase-adminsdk-pevrv-476f6abf74.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'face-detection-2024.appspot.com' 
    })
bucket = storage.bucket()

doc_ref = 0
blob_1 = 0
blob_2 = 0

# try:
#     blobs = bucket.list_blobs()  # Lấy danh sách blob trong bucket
#     print("Danh sách tệp trong bucket:")
#     for blob in blobs:
#         print(blob.name)  # In tên của từng blob
#     print("Kết nối đến Firebase Storage thành công!")
# except Exception as e:
#     print(f"Có lỗi khi kết nối đến Firebase Storage: {e}")

# Liệt kê các blob trong bucket
# try:
#     blob_1 = bucket.blob('Hao_Hoang_21T1020347_ChanDung.jpg')
#     blob_1.make_public()
#     image_url = blob_1.public_url
#     st.image(image_url, caption="Ảnh chân dung")
# except Exception as e:
#     print(1)
image_ChanDung = 0
image_TheSV = 0
if select_options == 'Hoàng Hào':
    doc_ref = db.collection("1").document("1")
    blob_1 = bucket.blob('Hao_Hoang_21T1020347_ChanDung.jpg')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    blob_2 = bucket.blob('Hao_Hoang_21T1020347_TheSV.jpg')
    blob_2.make_public()
    image_TheSV = blob_2.public_url
    
    
elif select_options == 'Ngô Văn Hải':
    doc_ref = db.collection("1").document("2")
    
    blob_1 = bucket.blob('Hai_NgoVan_21T1020340_ChanDung.jpg')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    blob_2 = bucket.blob('Hai_NgoVan_21T1020340_TheSV_1.jpg')
    blob_2.make_public()
    image_TheSV = blob_2.public_url
    
elif select_options == 'Trương Đoàn':
    doc_ref = db.collection("1").document("3")
    
    blob_1 = bucket.blob('Doan_Truong_21T1020306_ChanDung.HEIC')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    blob_2 = bucket.blob('Doan_Truong_21T1020306_TheSV.HEIC')
    blob_2.make_public()
    image_TheSV = blob_2.public_url
    
elif select_options == 'Nguyễn Phước Bình':
    doc_ref = db.collection("1").document("4")
    
    blob_1 = bucket.blob('Binh_Nguyen_Phuoc_21T1020117_ChanDung.jpg')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    blob_2 = bucket.blob('Binh_Nguyen_Phuoc_21T1020117_TheSV.jpg')
    blob_2.make_public()
    image_TheSV = blob_2.public_url
    
elif select_options == 'Nguyễn Vũ Hoàng	Chương':
    doc_ref = db.collection("1").document("5")
    
    blob_1 = bucket.blob('Chuong_NguyenVuHoang_21T1020267_ChanDung.jpg')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    blob_2 = bucket.blob('Chuong_NguyenVuHoang_21T1020267_TheSV.jpg')
    blob_2.make_public()
    image_TheSV = blob_2.public_url
    
elif select_options == 'Trần Thị Thanh Huệ':
    doc_ref = db.collection("1").document("6")
    
    blob_1 = bucket.blob('Hue_TranThiThanh_21T1020031_ChanDung.jpg')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    # blob_2 = bucket.blob('Hue_TranThiThanh_21T1020031_ChanDung.jpg')
    # blob_2.make_public()
    # image_TheSV = blob_2.public_url
    
elif select_options == 'Lê Bá Nhật Minh':
    doc_ref = db.collection("1").document("7")
    
    blob_1 = bucket.blob('Minh_LeBaNhat_21T1020047_ChanDung.JPG')
    blob_1.make_public()
    image_ChanDung = blob_1.public_url
    
    blob_2 = bucket.blob('Minh_LeBaNhat_21T1020047_TheSV.JPG')
    blob_2.make_public()
    image_TheSV = blob_2.public_url
    
elif select_options == 'Lê Trần Khánh Tùng':
    doc_ref = db.collection("1").document("8")
    
elif select_options == 'Lê Minh Tú':
    doc_ref = db.collection("1").document("9")
    
doc = doc_ref.get()
doc_data = doc.to_dict()
Ten = doc_data.get('Ten')
Masv = doc_data.get('Ma sinh vien')
Nganh = doc_data.get('Nganh')
Khoa = doc_data.get('Khoa')
Ngaysinh = doc_data.get('Ngay sinh')




c1, c2, c3 = st.columns(3)
c1.write(f" -   Tên : {Ten}")
c1.write(f" -   Mã sinh viên : {Masv}")
c1.write(f" -   Ngành : {Nganh}")
c1.write(f" -   Khóa : {Khoa}")
c1.write(f" -   Ngày sinh : {Ngaysinh}")

if image_ChanDung != 0:
    c2.image(image_ChanDung, caption= "Ảnh chân dung")
if image_TheSV != 0:
    c3.image(image_TheSV, caption="Ảnh thẻ sv")
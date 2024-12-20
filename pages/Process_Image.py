import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Process Image Application")

def flip_image_opencv(img, flip_code):
    flipped_img = cv.flip(img, flip_code)
    return flipped_img
  

def Flip(image_upload):
    if image_upload is not None:
        c = st.columns([2.5, 2.5, 4])
        image_ul = np.array(Image.open(image_upload))
        image = None
        with c[0]:
            st.markdown("**Ảnh gốc**")
            st.image(image_ul)
            cc = st.columns([4, 4, 2])
            with cc[0]:
                if st.button("Lật ngang"):
                    image = flip_image_opencv(image_ul, 1)
            with cc[1]:
                if st.button("Lật dọc"):
                    image = flip_image_opencv(image_ul, 0)
        
        if image is not None:
            with c[1]:
                st.markdown("**Ảnh sau khi lật**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download", byte_im, 'flip_result.png', "image/png")

def rotate_image(image, angle):
    angle = -angle
    (h, w) = image.shape[:2]
    
    center = (w // 2, h // 2)
    
    # Tạo ma trận xoay với góc angle
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Xoay ảnh
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

def Rotation(image_upload):
    if image_upload is not None:
        angel = st.slider("Chọn góc xoay", 0, 360, 0)
        cc = st.columns([2.5, 2.5, 4])
        image = None
        with cc[0]:
            image_ul = np.array(Image.open(image_upload))
            st.markdown("**Ảnh gốc**")
            st.image(image_ul)
            c = st.columns([1.5, 1.5, 7])
            if st.button("Apply"):
                image = rotate_image(image_ul, angel)
        if image is not None:
            with cc[1]:
                st.markdown("**Ảnh sau khi xoay**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download", byte_im, 'rotation_result.png', "image/png")

def convert_colorspace(image, color_type):
    # if color_type == "BGR":
    #     image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if color_type == "Grayscale":
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if color_type == "HSV":
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return image

def Colorspace(image_upload):
    color_type = st.selectbox(
                'Chọn không gian màu',
                ('BGR', 'Grayscale', 'HSV')
            ) 
    cc = st.columns([2.5, 2.5, 4])
    if image_upload is not None:
        image = None  
        with cc[0]:
            image_ul = np.array(Image.open(image_upload))
            st.markdown("**Ảnh gốc**")
            st.image(image_ul)
            if st.button("Apply"):
                image = convert_colorspace(image_ul, color_type)
        if image is not None:
            with cc[1]:
                st.markdown("**Ảnh sau khi điều chỉnh màu sắc**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download", byte_im, 'colorspace_result.png', "image/png")

def translate_image(image, tx, ty):

  (rows, cols) = image.shape[:2]

  M = np.float32([[1, 0, tx], [0, 1, ty]])

  translated_image = cv.warpAffine(image, M, (cols, rows))

  return translated_image

def Translation(image_upload):
    if image_upload is not None:
        image_ul = np.array(Image.open(image_upload))
        tx = st.slider("Di chuyển theo chiều ngang", -image_ul.shape[1], image_ul.shape[1], 0)
        ty = st.slider("Di chuyển theo chiều dọc", -image_ul.shape[1], image_ul.shape[1], 0)
        cc = st.columns([2.5, 2.5, 4])
        image = None
        with cc[0]:
            st.markdown("**Ảnh gốc**")
            st.image(image_ul)
            if st.button("Apply"):
                image = translate_image(image_ul, tx, ty)
        if image is not None:
            with cc[1]:
                st.markdown("**Ảnh sau khi di chuyển**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download", byte_im, 'translation_result.png', "image/png")
    
def Cropping(image_upload):
    st.markdown("**Kéo thả chuột để chọn vùng ảnh cần cắt**")
    cc = st.columns([2.5, 2.5, 4])
    if image_upload is not None:
        image_ul = Image.open(image_upload)
        with cc[0]:
            width, height = image_ul.size
            stroke_width = 3
            stroke_color = "red"
            drawing_mode = "rect"
            canvas_result = st_canvas(
                fill_color = "rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color = stroke_color,
                background_image=image_ul,
                # update_streamlit=realtime_update,
                width = image_ul.width,
                height = image_ul.height,
                drawing_mode=drawing_mode,
                key=image_upload.name,
            )
            rect = None
            if canvas_result is not None and canvas_result.json_data is not None:
                list_rect = []
                for obj in canvas_result.json_data["objects"]:
                    x = obj["left"] 
                    y = obj["top"]
                    
                    width = obj["width"] 
                    height = obj["height"] 
                    min_x = int(x)
                    min_y = int(y) 
        
                    rect = (min_x, min_y, min_x + int(width), min_y + int(height))
            cropped_img = None
            if st.button("Apply"):
                if rect is not None:
                    cropped_img = image_ul.crop(rect)
                else:
                    st.warning("Vui lòng chọn vùng ảnh cần cắt!")
        if cropped_img is not None:
            with cc[1]:
                st.markdown("**Ảnh sau khi cắt**")
                st.image(cropped_img)
                cropped_img = np.array(cropped_img)
                result_image = Image.fromarray(cropped_img)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download", byte_im, 'crop_result.png', "image/png")

def Application():
    c = st.columns(2)
    select_box = st.selectbox("**Chọn kĩ thuật xử lí ảnh**", ("Flip", "Rotation", "Colorspace", "Translation", "Cropping"))
    image_upload = st.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg"])
    if select_box == "Flip":
        Flip(image_upload)
    elif select_box == "Rotation":
        Rotation(image_upload)
    elif select_box == "Colorspace":
        Colorspace(image_upload)
    elif select_box == "Translation":
        Translation(image_upload)
    elif select_box == "Cropping":
        Cropping(image_upload)

def Introduce():
    st.markdown(
                """
                - Ứng dụng này dùng để xử lí hình ảnh bằng các kĩ thuật như: **Flip, Rotation, Colorspace, Translation và Cropping**
                """
    )
def App():
    Introduce()
    Application()
App()
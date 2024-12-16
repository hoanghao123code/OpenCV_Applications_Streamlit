import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Process Image Application")

def Introduce():
    st.header("1. Giới thiệu")
    st.markdown(
                """
                - Chào mừng đến với ứng dụng chỉnh sửa ảnh mạnh mẽ của chúng tôi! Ứng dụng này cho phép bạn dễ dàng biến đổi hình ảnh của mình với một loạt các công cụ chỉnh sửa chuyên nghiệp. 
                Bạn có thể cắt (**Cropping**), xoay (**Rotation**), lật (**Flip**), điều chỉnh màu sắc(**Colorspace**) và di chuyển hình ảnh (**Translation**) chỉ với vài thao tác đơn giản. Hãy cùng khám phá các tính năng tuyệt vời này nhé!
                """
    )

def flip_image_opencv(img, flip_code):
    flipped_img = cv.flip(img, flip_code)
    return flipped_img
  

def Flip():
    st.markdown("#### 2.1 Flip (Lật ảnh)")
    st.markdown(
                """
                - **Hướng dẫn sử dụng:**
                    - Tải ảnh cần **Flip (lật ảnh)** ở phần **Browser.**
                    - Nhấn vào nút **Lật Ngang** để lật ảnh theo chiều ngang. Nhấn vào nút **Lật Dọc** để lật ảnh theo chiều dọc.
                """
    )
    cc = st.columns(2)
    with cc[0]:
        image_upload = st.file_uploader("Tải ảnh cần lật (Flip)", type=["png", "jpg", "jpeg"])
        if image_upload is not None:
            image_ul = np.array(Image.open(image_upload))
            image = None
            if st.button("Lật ngang"):
                image = flip_image_opencv(image_ul, 1)
            if st.button("Lật dọc"):
                image = flip_image_opencv(image_ul, 0)
            
            if image is not None:
                st.markdown("**Ảnh sau khi lật**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download ảnh sau khi xử lí", byte_im, 'flip_result.png', "image/png")

def rotate_image(image, angle):
    angle = -angle
    (h, w) = image.shape[:2]
    
    center = (w // 2, h // 2)
    
    # Tạo ma trận xoay với góc angle
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Xoay ảnh
    rotated_image = cv.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

def Rotation():
    st.markdown("#### 2.2 Rotation (Xoay ảnh)")
    st.markdown(
                """
                - **Hướng dẫn sử dụng:**
                    - Tải ảnh cần **Rotation (xoay ảnh)** ở phần **Browser.**
                    - Sử dụng thanh trượt để chọn góc xoay. Hoặc nhập giá trị góc xoay vào ô bên cạnh. Nhấn nút **Xoay Trái** hoặc **Xoay Phải** để xoay nhanh **90** độ."
                """
    )
    cc = st.columns(2)
    with cc[0]:
        image_upload_rotate = st.file_uploader("Tải ảnh cần xoay (Rotation)", type=["png", "jpg", "jpeg"])
        if image_upload_rotate is not None:
            image_ul = np.array(Image.open(image_upload_rotate))
            image = None
            angel = 0
            angel = st.slider("Chọn góc xoay", 0, 360, 0)
            # c = st.columns([1.5, 1.5, 7])
            if st.button("Xoay Trái"):
                angel = -90
            if st.button("Xoay Phải"):
                    angel = 90
            image = rotate_image(image_ul, angel)
            if image is not None:
                st.markdown("**Ảnh sau khi xoay**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download ảnh sau khi xử lí", byte_im, 'rotation_result.png', "image/png")

def convert_colorspace(image, color_type):
    # if color_type == "BGR":
    #     image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if color_type == "Grayscale":
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if color_type == "HSV":
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return image

def Colorspace():
    st.markdown("#### 2.3 Colorspace (Điều chỉnh màu sắc)")
    st.markdown(
                """
                - **Hướng dẫn sử dụng:**
                    - Tải ảnh cần **Colorspace (Điều chỉnh màu sắc)** ở phần **Browser.**
                    - Chọn không gian màu mong muốn từ danh sách: **BGR, Đen trắng (Grayscale), HSV**. Nhấn nút **Áp dụng** để chuyển đổi.
                    - **Lưu ý:** "Chuyển đổi sang **Grayscale** sẽ loại bỏ thông tin màu sắc khỏi ảnh."
                """
    )
    cc = st.columns(2)
    with cc[0]:
        image_upload_color = st.file_uploader("Tải ảnh cần điều chỉnh màu sắc", type=["png", "jpg", "jpeg"])
        if image_upload_color is not None:
            image_ul = np.array(Image.open(image_upload_color))
            image = None
            color_type = st.selectbox(
                'Chọn một tùy chọn:',
                ('BGR', 'Grayscale', 'HSV')
)
            image = convert_colorspace(image_ul, color_type)
            if image is not None:
                st.markdown("**Ảnh sau khi điều chỉnh màu sắc**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download ảnh sau khi xử lí", byte_im, 'colorspace_result.png', "image/png")

def translate_image(image, tx, ty):

  (rows, cols) = image.shape[:2]

  M = np.float32([[1, 0, tx], [0, 1, ty]])

  translated_image = cv.warpAffine(image, M, (cols, rows))

  return translated_image

def Translation():
    st.markdown("#### 2.4 Di chuyển hình ảnh (Translation)")
    st.markdown(
                """
                - **Hướng dẫn sử dụng:**
                    - Tải ảnh cần **Translation (Di chuyển)** ở phần **Browser.**
                    - Nhập giá trị pixel bạn muốn di chuyển ảnh theo chiều ngang (trục X) và chiều dọc (trục Y) vào các ô tương ứng."
                """
    )
    cc = st.columns(2)
    with cc[0]:
        image_upload_color = st.file_uploader("Tải ảnh cần di chuyển", type=["png", "jpg", "jpeg"])
        if image_upload_color is not None:
            image_ul = np.array(Image.open(image_upload_color))
            image = None
            c = st.columns([1.5, 1.5, 7])
            tx = st.slider("Trục X", -image_ul.shape[1], image_ul.shape[1], 0)
            ty = st.slider("Trục Y", -image_ul.shape[1], image_ul.shape[1], 0)
            image = translate_image(image_ul, tx, ty)
            if image is not None:
                st.markdown("**Ảnh sau khi di chuyển**")
                st.image(image)
                result_image = Image.fromarray(image)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download ảnh sau khi xử lí", byte_im, 'translation_result.png', "image/png")
    
def Cropping():
    st.markdown("#### 2.5 Cắt ảnh (Cropping)")
    st.markdown(
                """
                - **Hướng dẫn sử dụng:**
                    - Tải ảnh cần **Cropping (Cắt ảnh)** ở phần **Browser.**
                    - Kéo các cạnh hoặc góc của khung cắt để điều chỉnh vùng chọn. Nhấn nút **Cắt** để cắt ảnh.
                """
    )
    cc = st.columns(2)
    with cc[0]:
        image_upload_crop = st.file_uploader("Tải ảnh cần cắt", type=["png", "jpg", "jpeg"])
        if image_upload_crop is not None:
            image_ul = Image.open(image_upload_crop)
            width, height = image_ul.size

            x_min = st.slider("X min", 0, width - 1, 0)

            y_min = st.slider("Y min", 0, height - 1, 0)

            x_max = st.slider("X max", x_min + 1, width, width)

            y_max = st.slider("Y max", y_min + 1, height, height)
            cropped_img = image_ul.crop((x_min, y_min, x_max, y_max))
            if cropped_img is not None:
                st.markdown("**Ảnh sau khi di cắt**")
                st.image(cropped_img)
                cropped_img = np.array(cropped_img)
                result_image = Image.fromarray(cropped_img)
                buf = BytesIO()
                result_image.save(buf, format = "PNG")
                byte_im = buf.getvalue()
                if byte_im is not None:
                    st.download_button("Download ảnh sau khi xử lí", byte_im, 'crop_result.png', "image/png")

def Application():
    st.header("2. Ứng dụng")
    c = st.columns(2)
    with c[0]:
        Flip()
        Rotation()
        Colorspace()
    with c[1]:
        Translation()
        Cropping()
def App():
    Introduce()
    Application()
App()
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os

from io import BytesIO
from PIL import Image
# from rembg import remove
from streamlit_drawable_canvas import st_canvas

st.title('🎈Hoang Hao GrabCut App')


def run():
    print(__doc__)
    image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if image_upload is not None:
        drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect")
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        # if drawing_mode == 'point':
        #     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        # stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        realtime_update = st.sidebar.checkbox("Update in realtime", True)
        
        if not os.path.exists('images'):
            os.makedirs('images')
        image = Image.open(image_upload)
        image.save('images/' + image_upload.name)
        
    
        # Tạo thành phần canvas
        img = cv.imread('images/' + image_upload.name)
        max_size = 350
        h = image.height
        w = image.width
        if max(h, w) > max_size:
            image.thumbnail((max_size, max_size))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('   <p style="text-indent: 100px;"> <span style = "color:red; font-size:22px;"> Ảnh gốc</span>', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                background_image=image,
                update_streamlit=realtime_update,
                height = image.height,
                width = image.width,
                drawing_mode=drawing_mode,
                key="canvas",
            )
            
        # canvas_result.json_data chứa thông tin các hình vẽ trên canvas
        image_ul = np.array(Image.open(image_upload))
        masks = np.zeros(image_ul.shape[:2], np.uint8)
        if canvas_result is not None and canvas_result.json_data is not None:
            list_rect = []
            for obj in canvas_result.json_data["objects"]:
                # Tọa độ x, y trái dưới
                # print(obj['type'])
                # rect, path, line
                x = obj["left"]
                y = obj["top"]
                
                # Chiều dài, chiều rộng
                width = obj["width"]
                height = obj["height"]
                min_x = x 
                min_y = y 
                max_x = x + width
                max_y = y + height
                rect = (min_x, min_y, max_x, max_y)
                list_rect.append(rect)
                
                # Xử lí foreground
                
                if obj['type'] == 'path':
                   path_data = obj['path']
                   print(type(path_data))
                #    for x, y in path_data:
                #        x = int(x)
                #        y = int(y)
                #        print(x, y)
                #        if 0 <= x < image.height and 0 <= y < image.width:
                #            masks[y, x] = 1
                if obj['type'] == 'line':
                    path_data = obj['line']
                    # for x, y in path_data:
                    #     x = int(x)
                    #     y = int(y)
                    #     if 0 <= x < image.height and 0 <= y < image.width:
                    #         masks[y, x] = 0
                # mask, back_ground, foreground
            rect = 0
            if len(list_rect) > 0:
                rect = list_rect[0]
                # masks = np.zeros(image_ul.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Áp dụng grapCut
                
                cv.grabCut(img, masks, rect, bgd_model, fgd_model, 1, cv.GC_INIT_WITH_RECT)
                
                # Sửa đổi mask để các pixel được gán nhãn là foreground là 1, còn lại là 0 
                mask2 = np.where((masks == 2) | (masks == 0), 0, 1).astype('uint8')
                
                # Áp masks vào ảnh gốc
                grabcut_result = image_ul * mask2[:, :, np.newaxis]
                
                # In ảnh sau khi xử lí
                
                if st.button("Submit"):
                    with c2:
                        c2.markdown(' <p style="text-indent: 60px;"> <span style = "color:red; font-size:22px;">    Ảnh sau khi xử lí</span>', unsafe_allow_html=True)
                        result = cv.resize(grabcut_result, (image.width, image.height))
                        if 'processed_image' not in st.session_state:
                            st.session_state.processed_image = result
                        st.image(st.session_state.processed_image)
                        result_image = Image.fromarray(st.session_state.processed_image)
                        buf = BytesIO()
                        result_image.save(buf, format = "PNG")
                        byte_im = buf.getvalue()
                        if byte_im is not None:
                            st.download_button("Download ảnh sau khi xử lí", byte_im, 'grabcut_result.png', "image/png")
            
undo_symbol = "↩️"
trash_symbol = "🗑️"
def How_to_Use():
    # st.markdown('<span style="color:blue;">This is blue text</span>', unsafe_allow_html=True)
    st.markdown('<span style = "color:blue; font-size:24px;">Cách sử dụng</span>', unsafe_allow_html=True)
    st.write("  - Chọn ảnh cần xử lí ở mục **Browse files**")
    st.write("  - Vẽ hình chữ nhật xung quanh đối tượng cần tách ra khỏi Background")
    st.write(f"  - Khi cần hoàn tác thao tác vừa thực hiện, **Click** chuột vào {undo_symbol} ở dưới ảnh")
    st.write(f"  - Khi cần Reset lại từ đầu các thao tác, **Click** chuột vào {trash_symbol} ở dưới ảnh")
    st.write("  - Sau đó nhấn nút **Submit** ở bên dưới để nhận kết quả")
    
How_to_Use()
run()  
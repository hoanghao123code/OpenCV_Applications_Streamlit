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


if __name__ == '__main__':
    print(__doc__)
    image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if image_upload is not None:
        st.write("Ảnh gốc")
        st.image(image_upload)
        drawing_mode = st.sidebar.selectbox("Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        realtime_update = st.sidebar.checkbox("Update in realtime", True)
        
        if not os.path.exists('images'):
            os.makedirs('images')
        image = Image.open(image_upload)
        image.save('images/' + image_upload.name)
        
        # background_image=Image.open(image_upload.name)
        # background_image.save(image_upload.name)
        # Tạo thành phần canvas
        img = cv.imread('images/' + image_upload.name)
        
        # st.write("spdops")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            # fill_color="",  # Fixed fill color with some opacity
            
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            # background_color=bg_color,
            background_image=image,
            update_streamlit=realtime_update,
            height = image.height,
            width = image.width,
            drawing_mode=drawing_mode,
            # point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )
        # canvas_result.json_data chứa thông tin các hình vẽ trên canvas
        image_ul = np.array(Image.open(image_upload))
        if canvas_result is not None and canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                
                # Tọa độ x, y trái dưới
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
                
                # mask, back_ground, foreground
                masks = np.zeros(image_ul.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Áp dụng grapCut
                
                cv.grabCut(img, masks, rect, bgd_model, fgd_model, 1, cv.GC_INIT_WITH_RECT)
                
                # Sửa đổi mask để các pixel được gán nhãn là foreground là 1, còn lại là 0 
                mask2 = np.where((masks == 2) | (masks == 0), 0, 1).astype('uint8')
                
                # Áp masks vào ảnh gốc
                
                grabcut_result = image_ul * mask2[:, :, np.newaxis]
                
                # In ảnh sau khi xử lí
                
                st.write('Ảnh sau khi xử lí')
                st.image(grabcut_result)
                
                # result_image = Image.fromarray(grabcut_result)
                # buf = BytesIO()
                # result_image.save(buf, format = "PNG")
                # byte_im = buf.getvalue()
                # if byte_im is not None:
                #     st.download_button("Download ảnh sau khi xử lí", byte_im, 'grabcut_result.png', "image/png")
                
                # ori_image = cv.imread('images/' + image_upload.name)
                
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

st.title('🎈 GrabCut App')


# st.write('Hello world!')

#!/usr/bin/env python
'''
===============================================================================
# Grabcut

A simple program for interactively removing the background from an image using
the grab cut algorithm and OpenCV.

This code was derived from the Grab Cut example from the OpenCV project.

## Operation

At startup, two windows will appear, one for input and one for output.

To start, in input window, draw a rectangle around the object using mouse right
button.  For finer touch-ups, press any of the keys below and draw circles on
the areas you want.  Finally, press 's' to save the result.

## Keys
  * 0 - Select areas of sure background
  * 1 - Select areas of sure foreground
  * 2 - Select areas of probable background
  * 3 - Select areas of probable foreground
  * n - Update the segmentation
  * r - Reset the setup
  * s - Save the result
  * q - Quit
===============================================================================
'''

# Python 2/3 compatibility



if __name__ == '__main__':
    print(__doc__)
    image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if image_upload is not None:
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
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            # background_color=bg_color,
            background_image=Image.open(image_upload) if image_upload else None,
            update_streamlit=realtime_update,
            height=img.shape[0],
            width = img.shape[1],
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )
        st.download_button(
            "Download grabcut image", image_upload, "fixed.png", "image/png"
        )
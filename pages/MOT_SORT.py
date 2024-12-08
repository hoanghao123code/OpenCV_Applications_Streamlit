import streamlit as st
from PIL import Image, ImageOps
import cv2 as cv

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("SORT (Simple Online Realtime Tracking)")
def SORT_Algorithm():
    st.header("1. Thuật toán SORT")
    st.markdown("#### 1.1 Giới thiệu")
    c = st.columns(2)
    with c[0]:
        st.markdown(
                    """
                    - **SORT (Simple Online and Realtime Tracking)** là một thuật toán theo dõi đối tượng đa mục tiêu 
                    **(multi-object tracking)** phổ biến, được phát triển bởi **Alex Bewley** và cộng sự vào năm 2016. 
                    **SORT** được thiết kế để đơn giản, hiệu quả và phù hợp với các ứng dụng thời gian thực **(real-time applications)**.
                    """)
    with c[1]:
        st.markdown("Dưới đây là hình ảnh so sánh **Speed(Tốc độ)** và **Accuracy (Độ chính xác)** của **SORT** với các thuật toán theo dõi đối tượng khác.")
        # c = st.columns([2, 6, 2])
        st.image(cv.imread("./images/MOT_SORT/SORT.PNG"), channels="BGR", caption="So sánh Speed và Accuracy của các thuật toán theo dõi đối tượng")
    st.markdown("#### 1.2 Cách hoạt động")
    st.markdown("Dưới đây là hình ảnh về luồng xử lí của thuật toán **SORT**")
    c = st.columns([2, 6, 2])
    c[1].image(cv.imread("./images/MOT_SORT/pipeline_SORT.PNG"), channels="BGR", caption="Luồng xử lí của SORT")
    st.markdown("""
                **SORT** kết hợp giữa **phát hiện đối tượng (object detection)** và **lọc Kalman (Kalman filter)**, giải thuật **Hungarian** để theo dõi các đối tượng 
                qua các khung hình. Dưới đây là các bước chính:
                """)
    c = st.columns(2)
    with c[0]:
        st.markdown("##### 1. Phát hiện đối tượng (Object Detection)")
        st.markdown(
                    """
                    - **SORT** không tự thực hiện phát hiện đối tượng mà dựa vào các mô hình phát hiện **(detector)** như **YOLO, SSD, Faster R-CNN...**
                    - Output của **detector** là các **bounding box** biểu diễn vị trí và kích thước của các đối tượng.
                    """)
        st.markdown("##### 2. Dự đoán vị trí tiếp theo (Prediction)")
        st.markdown(
                    """
                    - **SORT** sử dụng [Kalman Filter](https://www.researchgate.net/profile/Chaw-Bing-Chang/publication/224680746_Kalman_filter_algorithms_for_a_multi-sensor_system/links/54a170c20cf257a636036eaf/Kalman-filter-algorithms-for-a-multi-sensor-system.pdf) 
                    để dự đoán vị trí và kích thước của các **bounding box** trong khung hình tiếp theo.
                    - Trạng thái của một đối tượng được biểu diễn bởi một vector trạng thái bao gồm:
                        - $X = [x, y, s, r, x', y', s']$
                    - **Trong đó**:
                        - $x, y$ : Toạ độ trung tâm **bounding box.**
                        - $s$ : Diện tích của **bounding box.**
                        - $r$ : Tỷ lệ khung hình (width / height).
                        - $x', y', s'$ : Lần lượt là các giá trị vận tốc tương ứng của $x, y, s$.
                    """)
    with c[1]:
        st.markdown("##### 3. Liên kết các đối tượng (Data Association)")
        st.markdown(
                    """
                    - **SORT** sử dụng thuật toán [Hungarian Algorithm](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9de0d77fed3781f98de743aba6ac4967688d711f)
                    để liên kết các bounding box dự đoán (từ **Kalman Filter**) với các **bounding box** phát hiện (từ **detector**).
                    - Liên kết dựa trên chỉ số **IoU (Intersection over Union)** giữa các **bounding box**.
                    """)
        st.markdown("##### 4. Cập nhật trạng thái")
        st.markdown(
                    """
                    - Sau khi liên kết được thực hiện, **Kalman Filter** cập nhật lại trạng thái của đối tượng dựa trên thông tin từ **bounding box** phát hiện.
                    """)
        st.markdown("##### 5. Quản lý ID và đối tượng mới")
        st.markdown(
                    """
                    - **SORT** theo dõi các đối tượng bằng cách gán **ID** duy nhất cho từng đối tượng.
                    - Các đối tượng mất liên kết sau một số khung hình (do không được phát hiện) sẽ bị loại bỏ.
                    - Đối tượng mới xuất hiện trong khung hình sẽ được gán **ID** mới.
                    """)


def Challenger_of_SORT():
    st.header("2. Một số thách thức của thuật toán SORT")
    st.markdown("#### 2.1 Nhạy cảm với việc mất dấu (Occlusion)")
    st.markdown(
                """
                - **SORT** sử dụng **Kalman Filter** để dự đoán vị trí tiếp theo của đối tượng và **Hungarian Algorithm** để gán đối tượng. 
                Tuy nhiên, khi xảy ra che khuất hoặc đối tượng bị tạm thời mất dấu, thuật toán dễ gán sai hoặc tạo **ID** mới cho đối tượng, 
                dẫn đến sự phân mảnh **(ID switching)**.
                """)
    st.markdown("#### 2.2 Đối tượng bị che khuất 1 phần hoặc toàn bộ")
    st.markdown(
                """
                - Khi 1 ID được gán cho 1 đối tượng, ID cần đảm bảo nhất quán trong suốt video, tuy nhiên, khi một đối tượng bị che khuất, 
                nếu chỉ dựa riêng vào object detection là không đủ để giải quyết vấn đề này.
                """)

def App():
    SORT_Algorithm()
    Challenger_of_SORT()
App()
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import streamlit as st
import cv2 as cv
import sys
import time

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Object Tracking Algorithm")
(major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
def CSRT_Tracking_Algorithm():
    st.header("1. Thuật toán CSRT")
    st.markdown("#### 1.1 Giới thiệu")
    st.markdown(
                """
                - **CSRT (Channel and Spatial Reliability Tracker)** là một thuật toán theo dõi đối tượng trong **OpenCV**, 
                được thiết kế để cải thiện độ chính xác và độ ổn định so với các thuật toán theo dõi khác như **KCF (Kernelized Correlation Filters)**.
                **CSRT** sử dụng thông tin kênh màu **(Channel)** và thông tin không gian **(Spatial)** để điều chỉnh độ tin cậy của các mẫu, giúp xử lý tốt 
                các tình huống biến đổi về hình dạng hoặc môi trường.
                """)
    st.markdown("#### 1.2 Quy trình hoạt động của CSRT")
    image = cv.imread("./images/Object_Tracking/pineline_CSRT.PNG")
    st.image(image, channels="BGR")
    c = st.columns(2)
    with c[0]:
        st.markdown("##### 1.2.1 Khởi tạo")
        st.markdown(
                    """
                    - Người dùng chọn một vùng ban đầu **(bounding box)** chứa đối tượng cần theo dõi.
                    - **CSRT** tạo một mô hình dựa trên các đặc trưng của đối tượng trong **bounding box**.
                    - Tính toán các đặc trưng từ **bounding box** bằng cách sử dụng:
                        - Kênh màu (Color Channels).
                        - Gradient hướng (HOG).
                        - Đặc trưng không gian và tần số.
                    """)
        st.markdown("##### 1.2.3 Cập nhật")
        st.markdown(
                    """
                    - Khi đối tượng thay đổi (về hình dạng hoặc kích thước), **CSRT** điều chỉnh mô hình bằng cách cập nhật thông tin từ các **frame** mới.
                    """)
    with c[1]:
        st.markdown("##### 1.2.2 Theo dõi")
        st.markdown(
                    """
                    - Ở mỗi frame mới:
                        - Trích xuất các đặc trưng từ vùng lân cận **bounding box** hiện tại.
                            - Xác định một vùng lân cận **(search window)** xung quanh vị trí của **bounding box** trong **frame** trước.
                            - Trích xuất các kênh đặc trưng từ vùng này
                        - Tính toán độ tin cậy của từng kênh và từng vùng không gian.
                            - Các kênh (ví dụ: màu, gradient) được đánh giá để xem kênh nào phù hợp nhất để phân biệt đối tượng.
                            - Các kênh không đáng tin cậy sẽ được giảm trọng số hoặc loại bỏ trong tính toán.
                            - Áp dụng một mặt nạ **(spatial reliability mask)** để xác định vùng nào trong **bounding box** đáng tin cậy nhất.
                            - Loại bỏ các vùng nhiễu hoặc không liên quan
                        - So khớp
                            - Sử dụng bộ lọc đã huấn luyện từ **frame** trước để tìm vị trí phù hợp nhất trong vùng lân cận.
                            - Tính toán một **hàm mất mát** để xác định vị trí của **bounding box** tối ưu.
                        - Cập nhật bộ lọc dự đoán vị trí tiếp theo của đối tượng.
                            - Điều chỉnh bộ lọc dự đoán để phản ánh các thay đổi về hình dạng, kích thước, hoặc môi trường của đối tượng.
                        - Đưa ra vị trí **bounding box** mới.
                            - Lấy vùng **bounding box** mới từ **frame hiện tại**.
                            - Cập nhật các đặc trưng **(kênh màu, gradient, tần số)**.
                    """)
    
def Example():
    undo_symbol = "↩️"
    trash_symbol = "🗑️"
    st.header("2. Ví dụ minh hoạ")
    st.markdown('<span style = "color:blue; font-size:24px;">Các bước thực hiện</span>', unsafe_allow_html=True)
    st.markdown(
                f"""
                - Vẽ một **bounding box** xung quanh đối tượng muốn theo dõi
                - Khi cần hoàn tác thao tác vừa thực hiện, **Click** chuột vào {undo_symbol} ở dưới ảnh
                - Khi cần Reset lại từ đầu các thao tác, **Click** chuột vào {trash_symbol} ở dưới ảnh
                - Sau đó **Click** vào nút theo dõi
                """)
    if int(minor_ver) < 3:
        tracker = cv.Tracker_create("CSRT")
    else:
        tracker = cv.TrackerCSRT_create()
    tracker_type = "CSRT"
    # Read video
    video = cv.VideoCapture("./images/Object_Tracking/39837-424360872_small.mp4")
    # Read first frame.
    ok, frame = video.read()
    drawing_mode = "rect"
    stroke_width = 1
    stroke_color = "blue"
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    max_size = 800
    w = min(img.width, max_size)
    h = w * img.height // img.width
    c = st.columns([1])
    with c[0]:
        canvas_result = st_canvas(
            fill_color = "rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color = stroke_color,
            background_image=img,
            width = w,
            height = h,
            drawing_mode=drawing_mode,
            key="Object Tracking Algorithm",
        )
    if st.button("Theo dõi"):
        if canvas_result is not None and canvas_result.json_data is not None:
            list_rect = []
            scale = img.width / w
            bbox = None
            for obj in canvas_result.json_data["objects"]:
                # Tọa độ x, y trái trên
                x = obj["left"] * scale
                y = obj["top"] * scale
                
                # Chiều dài, chiều rộng
                width = obj["width"] * scale
                height = obj["height"] * scale
                min_x = int(x)
                min_y = int(y) 

                bbox = (min_x, min_y, int(width), int(height))
            # Initialize tracker with first frame and bounding box
            if bbox is not None:
                ok = tracker.init(frame, bbox)
                cnt_frame = 0
                image_placeholder = st.empty()
                while True:
                    # Read a new frame
                    ok, frame = video.read()
                    if not ok:
                        break
                    
                    # Start timer
                    timer = cv.getTickCount()
            
                    # Update tracker
                    ok, bbox = tracker.update(frame)
            
                    # Calculate Frames per second (FPS)
                    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
            
                    # Draw bounding box
                    if ok:
                        # Tracking success
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                    else :
                        # Tracking failure
                        cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
                    # Display tracker type on frame
                    cv.putText(frame, tracker_type + " Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                
                    # Display FPS on frame
                    cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                    image_placeholder.image(frame, channels="BGR")
                    # st.image(frame, channels="BGR")


def Challenge_of_CSRT():
    st.header("2. Một số thách thức")
    st.markdown("#### 2.1 Occlusion (Che khuất)")
    c = st.columns(4)
    path_1 = "./images/Object_Tracking/TruocKhiCheKhuat.PNG"
    path_2 = "./images/Object_Tracking/TrongKhiCheKhuat.PNG"
    path_3 = "./images/Object_Tracking/TrongKhiCheKhuat2.PNG"
    path_4 = "./images/Object_Tracking/TrongKhiCheKhuat3.PNG"
    path_5 = "./images/Object_Tracking/SauKhiCheKhuat1.PNG"
    path_6 = "./images/Object_Tracking/SauKhiCheKhuat2.PNG"
    path_7 = "./images/Object_Tracking/SauKhiCheKhuat3.PNG"
    path_8 = "./images/Object_Tracking/SauKhiCheKhuat4.PNG"
    c[0].image(cv.imread(path_1), channels="BGR", caption="Ảnh 1")
    c[1].image(cv.imread(path_2), channels="BGR", caption="Ảnh 2")
    c[2].image(cv.imread(path_3), channels="BGR", caption="Ảnh 3")
    c[3].image(cv.imread(path_4), channels="BGR", caption="Ảnh 4")
    c[0].image(cv.imread(path_5), channels="BGR", caption="Ảnh 5")
    c[1].image(cv.imread(path_6), channels="BGR", caption="Ảnh 6")
    c[2].image(cv.imread(path_7), channels="BGR", caption="Ảnh 7")
    c[3].image(cv.imread(path_8), channels="BGR", caption="Ảnh 8")
    st.markdown("Qua các hình ảnh ví dụ ta có thể thấy:")
    st.markdown(
                """
                - Sau khi đối tượng bị che khuất thì **bounding box** đã theo dõi sai đối tượng ban đầu
                và chuyển sang theo dõi một đối tượng khác.
                    - Vì **CSRT** không tích hợp cơ chế xử lý mạnh mẽ để phát hiện hoặc bù đắp khi đối tượng bị che khuất một phần hoặc toàn bộ. 
                    Khi điều này xảy ra, **CSRT** dễ bị trôi **(drift)** theo các đối tượng nhiễu.
                """)
    st.markdown("#### 2.2 Background Clutters (Nền phức tạp, nhiễu)")
    path_11 = "./images/Object_Tracking/BackgroundClutter1.PNG"
    path_12 = "./images/Object_Tracking/BackgroundClutter2.PNG"
    path_13 = "./images/Object_Tracking/BackgroundClutter3.PNG"
    path_14 = "./images/Object_Tracking/BackgroundClutter4.PNG"
    c = st.columns(4)
    c[0].image(cv.imread(path_11), channels="BGR", caption="Ảnh 1")
    c[1].image(cv.imread(path_12), channels="BGR", caption="Ảnh 2")
    c[2].image(cv.imread(path_13), channels="BGR", caption="Ảnh 3")
    c[3].image(cv.imread(path_14), channels="BGR", caption="Ảnh 4")
    st.markdown("Qua các hình ảnh ví dụ ta có thể thấy:")
    st.markdown(
                """
                - Với nhiều quả bóng có hình dạng và màu sắc tương tự nhau bounding box đã theo dõi sai đối tượng ban đầu
                    - Vì **CSRT** phụ thuộc vào các đặc trưng trực quan như **HOG** và **Color Names**. Trong các khung cảnh có nền phức tạp
                    hoặc chứa nhiều yếu tố giống đối tượng, thuật toán có thể bị nhầm lẫn giữa đối tượng và nền, dẫn đến mất theo dõi.
                """)
    st.markdown("#### 2.3 Fast Motion (Chuyển động nhanh)")
    path_21 = "./images/Object_Tracking/Fast1.PNG"
    path_22 = "./images/Object_Tracking/Fast2.PNG"
    path_23 = "./images/Object_Tracking/Fast3.PNG"
    path_24 = "./images/Object_Tracking/Fast4.PNG"
    c = st.columns(4)
    c[0].image(cv.imread(path_21), channels="BGR", caption="Ảnh 1")
    c[1].image(cv.imread(path_22), channels="BGR", caption="Ảnh 2")
    c[2].image(cv.imread(path_23), channels="BGR", caption="Ảnh 3")
    c[3].image(cv.imread(path_24), channels="BGR", caption="Ảnh 4")
    st.markdown("Qua các hình ảnh ví dụ ta có thể thấy:")
    st.markdown(
                """
                - Khi tốc độ của đối tượng lớn dẫn đến nhầm lẫn đối tượng cần theo dõi
                    - Vì **CSRT** dựa trên việc cập nhật thông tin từ khung hình liền kề. Khi đối tượng di chuyển nhanh giữa các khung, 
                    thuật toán có thể không kịp theo dõi do thay đổi lớn về vị trí hoặc kích thước của đối tượng.
                """)
    st.markdown("#### 2.4 Illumination Variations (Sự thay đổi ánh sáng)")
    st.markdown(
                """
                -  **CSRT** không được thiết kế để xử lý tốt các thay đổi về ánh sáng. Khi độ sáng thay đổi, các đặc trưng màu sắc 
                và gradient của đối tượng cũng thay đổi, khiến thuật toán khó nhận diện đúng đối tượng.
                """)
    
def App():
    CSRT_Tracking_Algorithm()
    Example()
    Challenge_of_CSRT()
App()
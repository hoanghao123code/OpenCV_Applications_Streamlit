import streamlit as st
import cv2 as cv
from PIL import Image, ImageOps

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Handwriting Letter Recognize")

def Text():
    st.header("1. Giới thiệu dataset MNIST")
    st.markdown(
                """
                - **MNIST (Modified National Institute of Standards and Technology)** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong lĩnh vực học máy, 
                đặc biệt là trong các bài toán nhận dạng hình ảnh. **MNIST** được thiết kế để đánh giá các thuật toán phân loại hình ảnh, đặc biệt là các mô hình học sâu **(deep learning)**
                """)
    c = st.columns(2)
    with c[0]:
        st.markdown("#### 1.1 Tổng quan về dataset MNIST")
        st.markdown(
                """
                - Kích thước: Dataset **MNIST** gồm có $70.000$ hình ảnh, chia thành hai phần:
                    - $60.000$ hình ảnh huấn luyện **(training set)**.
                    - &10.000$ hình ảnh kiểm tra **(test set)**.
                """)
    with c[1]:
        st.markdown("#### 1.2 Đặc điểm dataset MNIST")
        st.markdown(
                """
                - Kích thước hình ảnh: Mỗi hình ảnh có kích thước **28x28** pixel, với mỗi pixel là một giá trị độ sáng **(grayscale)** từ $0$ đến $255$.
                - Định dạng dữ liệu: Dữ liệu có dạng 2D, mỗi ảnh là một ma trận **28x28**, với mỗi giá trị là độ sáng của pixel (được chuẩn hóa từ $0$ đến $1$ khi chia cho $255$).
                - Ký tự viết tay: Các hình ảnh trong bộ dữ liệu được thu thập từ các công dân Mỹ, bao gồm cả trẻ em và người trưởng thành. Các chữ số được viết tay trong nhiều phong cách khác nhau, 
                giúp mô hình học máy phải có khả năng nhận diện chữ số viết tay trong các tình huống đa dạng.
                - Lớp nhãn (Labels): Mỗi hình ảnh có một nhãn tương ứng (label) là một trong các chữ số từ 0 đến 9.
                """)
    st.markdown("#### 1.3 Dưới đây là một số ảnh minh họa dataset MNIST")
    c = st.columns([2, 6, 2])
    image = cv.imread("./images/Handwriting_Letter_Recognize/dataset_MNIST.png")
    c[1].image(image, channels="BGR", caption="Minh họa dataset MNIST")
    st.header("2 Phương pháp")
    st.markdown(
            """
            - Mô hình được sử dụng để huấn luyện các kí tự trong dataset **MNIST** là một mô hình CNN gồm $13$ lớp bao gồm nhiều lớp **convolutional (conv)** và **pooling**, 
            với các lớp **fully connected (FC)** ở cuối để thực hiện phân loại.  
            """)
    st.markdown("#### 2.1 Kiến trúc của mô hình")
    c = st.columns(2)
    with c[0]:
        st.markdown("##### 2.1.1 Convolutional Block 1")
        st.markdown(
                    """
                    - **Lớp 1**: **Lớp Conv2D** với $32$ bộ lọc (filters), mỗi bộ lọc có kích thước **3x3**, hàm kích hoạt ReLU, padding same (giữ kích thước đầu ra giống với đầu vào).
                    - **Lớp 2**: **Lớp Conv2D** thứ hai với $32$ bộ lọc, giống như lớp trước, với kích thước kernel **3x3** và padding same
                    - **Lớp 3**: **Lớp MaxPooling2D** với kernel **2x2** để giảm kích thước của đặc trưng (ảnh) một cách hiệu quả. Sau lớp pooling, chiều rộng và chiều cao của ảnh giảm một nửa
                    """   
        )
        st.markdown("##### 2.1.2 Convolutional Block 2")
        st.markdown(
                    """
                    - **Lớp 4 & 5**: Hai lớp **Conv2D** với $64$ bộ lọc, kích thước kernel **3x3**, hàm kích hoạt ReLU và padding same. Các lớp này giúp mô hình học được các đặc trưng phức tạp hơn của ảnh.
                    - **Lớp 6**: Lớp **MaxPooling2D** với kernel **2x2** để tiếp tục giảm kích thước đặc trưng và làm giảm độ phức tạp tính toán.
                    """   
        )
        st.markdown("##### 2.1.3 Convolutional Block 3")
        st.markdown(
                    """
                    - **Lớp 7 & 8**: Hai lớp **Conv2D** với $128$ bộ lọc và kích thước kernel **3x3**. Các lớp này giúp mô hình học được các đặc trưng chi tiết hơn ở mức độ cao hơn.
                    - **Lớp 9**: Lớp **MaxPooling2D** giúp giảm kích thước ảnh, giữ lại các đặc trưng quan trọng trong khi giảm độ phân giải không gian.
                    """   
        )
    with c[1]:
        st.markdown("##### 2.1.4 Convolutional Block 4")
        st.markdown(
                    """
                    - **Lớp 10**: Một lớp **Conv2D** với **256** bộ lọc, kích thước kernel **3x3**. Việc sử dụng nhiều bộ lọc giúp mô hình học các đặc trưng phong phú hơn và giúp cải thiện 
                    độ chính xác khi nhận diện các chữ số viết tay.
                    - **Lớp 11**: **Lớp MaxPooling2D** tiếp theo giúp giảm kích thước ảnh sau lớp **convolution**.
                    """   
        )
        st.markdown("##### 2.1.4 Fully Connected (FC) Block")
        st.markdown(
                    """
                    - **Lớp 12**: Lớp **Dense** với $512$ units và hàm kích hoạt ReLU. Lớp này sử dụng các đặc trưng học được từ các lớp **convolution** để thực hiện phân loại.
                    - **Lớp 13**: Lớp **Dense** cuối cùng với $10$ units (một unit cho mỗi chữ số từ $0$ đến $9$) và hàm kích hoạt softmax để phân loại và xác suất của mỗi lớp.
                    """   
        )
    st.markdown("#### 2.2 Lý do chọn kiến trúc này")
    st.markdown(
                """
                Mô hình **CNN** này sử dụng cấu trúc tương tự **VGG** với nhiều lớp **convolution** và **pooling**, thích hợp cho bài toán phân loại chữ số viết tay **MNIST**. 
                Việc áp dụng **padding="same"**, **max-pooling**, và **fully connected layers** giúp mô hình học các đặc trưng chi tiết từ ảnh và cải thiện độ chính xác phân loại, 
                đồng thời giảm thiểu nguy cơ **overfitting**. Cấu trúc này là một lựa chọn hợp lý và hiệu quả cho **MNIST**, đạt được kết quả tốt với độ chính xác cao trong bài toán phân loại chữ số viết tay."
                """
    )
    st.markdown("#### 2.3 Quá trình huấn luyện")
    st.markdown("##### 2.3.1 ")
       
def App():
    Text()
App()
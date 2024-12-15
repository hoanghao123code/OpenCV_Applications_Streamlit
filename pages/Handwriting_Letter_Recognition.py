import streamlit as st
import cv2 as cv
from PIL import Image, ImageOps
from tensorflow.keras import layers, models
import pickle
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="🎈Hoang Hao's Applications",
    page_icon=Image.open("./images/Logo/logo_welcome.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Handwriting Letter Recognize")

def predict_with_image(image):
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Fully Connected Block 1
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer
    model.add(layers.Dense(10, activation='softmax')) 

    loaded_weights = []
    with open("./data_processed/Handwriting_Letter_Recognize/weight.pkl", "rb") as file:
        loaded_weights = pickle.load(file)

    model.set_weights(loaded_weights)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Process image
    if len(image.shape) == 3: 
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_cpy = image.copy()
    image[image_cpy != 0] = 255
    # st.image(image)
    image = cv.resize(image, (28, 28))
    image = image = image.astype('float32') / 255.0
    # Reshape ((batch_size, height, width, channels))
    image = image.reshape((1, 28, 28, 1))
    predictions = model.predict(image)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def crop_and_center_mnist(img):
    img_cpy = img.copy()
    id_white = np.where(img_cpy > 0)

    x_min = np.min(id_white[1])  
    x_max = np.max(id_white[1])
    y_min = np.min(id_white[0])  
    y_max = np.max(id_white[0])

    height = y_max - y_min
    width = x_max - x_min

    # Bounding box
    x, y, w, h = (x_min, y_min, width, height)
    cropped_img = img[y : y + h, x : x+w]
    
    padding = int(h * 0.25)

    # 5. Thêm padding
    padded_img = cv.copyMakeBorder(
        cropped_img,
        padding,
        padding,
        padding,
        padding,
        cv.BORDER_CONSTANT,
        value=0,
    )

    # 6. Resize về 28x28
    if padded_img.shape[0] <= 0 or padded_img.shape[1] <= 0:
      st.warning("Không tìm thấy hình vẽ!")
      return np.zeros((28,28), dtype=np.uint8)

    final_img = cv.resize(padded_img, (28, 28), interpolation=cv.INTER_AREA)
    return final_img



def Applications():
    undo_symbol = "↩️"
    trash_symbol = "🗑️"
    st.markdown('<span style = "color:blue; font-size:24px;">Cách sử dụng</span>', unsafe_allow_html=True)
    st.write("  - Vẽ chữ số cần dự đoán **(0, 1, ...9)** lên hình chữ nhật màu đen ở dưới")
    st.write(f"  - Khi cần hoàn tác thao tác vừa thực hiện, **Click** chuột vào {undo_symbol} ở dưới ảnh")
    st.write(f"  - Khi cần Reset lại từ đầu các thao tác, **Click** chuột vào {trash_symbol} ở dưới ảnh")
    st.write("  - Sau đó nhấn nút **Submit** ở bên dưới để nhận kết quả dự đoán")
    st.write("**Lưu ý:** Chỉ vẽ một chữ số **duy nhất**")
    stroke_width = 3
    stroke_color = "red"
    drawing_mode = "freedraw"
    image = Image.new("RGB", (280, 280), "black")
    c = st.columns([3, 7])
    with c[0]:
        canvas_result = st_canvas(
            fill_color = "rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color = stroke_color,
            background_color="#000000",
            background_image=image,
            # update_streamlit=realtime_update,
            width = image.width,
            height = image.height,
            drawing_mode=drawing_mode,
            key="Handwriting Letter Recognize",
        )
        c[1].markdown("**Kết quả dự đoán**")
        if st.button("Submit"):
            if canvas_result.image_data is not None:
                image_canvas = canvas_result.image_data
                if image_canvas.dtype != np.uint8:
                    image_canvas = cv.normalize(image_canvas, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                image_canvas = cv.cvtColor(image_canvas, cv.COLOR_BGR2GRAY)
                image_cpy = image_canvas.copy()
                image_canvas[image_cpy != 0] = 255
                # st.image(image_canvas)
                image_crop = crop_and_center_mnist(image_canvas)
                # st.image(image_crop)
                results = predict_with_image(image_crop)
                c[1].markdown(f"<p style='font-size: 30px;'>{results[0]}</p>", unsafe_allow_html=True)
            else:
                st.warning("Vui lòng vẽ kí tự cần dự đoán trước khi **Submit!**")

def Results():
    c = st.columns(2)
    with c[0]:
        st.markdown("Dưới đây là biểu đồ biểu diễn độ chính xác trên tập huấn luyện (**train**) và tập xác thực (**validation**)")
        image_accuracy = cv.imread("./images/Handwriting_Letter_Recognize/train_val_acc.PNG")
        st.image(image_accuracy, caption="Training and Validation Acccuracy", channels="BGR")
        accuracy_test = 0.976
        st.markdown(f"Độ chính xác trên tập **test** là: **{accuracy_test}**")
    with c[1]:
        st.markdown("Dưới đây là biểu đồ biểu diễn độ mất mát trên tập huấn luyện (**train**) và tập xác thực (**validation**)")
        image_loss = cv.imread("./images/Handwriting_Letter_Recognize/train_val_loss.PNG")
        st.image(image_loss, caption="Training and Validation Loss", channels="BGR")

def Training():
    st.markdown("#### 2.3 Quá trình huấn luyện")
    st.markdown(
                """
                - Trong dataset **MNIST:**
                    - Chia tập **training set** thành 2 tập:
                        - $50000$ ảnh cho tập huấn luyện (train)
                        - $10000$ ảnh cho tập xác thực (validation)
                    - Sử dụng $10000$ ảnh tập **test set** làm tập test.
                - Các tham số sử dụng trong quá trình huấn luyện:
                    - Hàm tối ưu: **Adam**
                    - Hàm mất mát: **sparse_categorical_crossentropy**
                    - Độ chính xác: **Accuracy**
                    - Learning_rate: **0.001**
                    - Số lượng Epoch: **10**
                """
    )
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
                    - $10.000$ hình ảnh kiểm tra **(test set)**.
                """)
    with c[1]:
        st.markdown("#### 1.2 Đặc điểm dataset MNIST")
        st.markdown(
                """
                - Kích thước hình ảnh: Mỗi hình ảnh có kích thước **28x28** pixel, với mỗi pixel là một giá trị độ sáng **(grayscale)** từ $0$ đến $255$.
                - Định dạng dữ liệu: Dữ liệu có dạng **2D**, mỗi ảnh là một ma trận **28x28**, với mỗi giá trị là độ sáng của pixel (được chuẩn hóa từ $0$ đến $1$ khi chia cho $255$).
                - Ký tự viết tay: Các hình ảnh trong bộ dữ liệu được thu thập từ các công dân Mỹ, bao gồm cả trẻ em và người trưởng thành. Các chữ số được viết tay trong nhiều phong cách khác nhau, 
                giúp mô hình học máy phải có khả năng nhận diện chữ số viết tay trong các tình huống đa dạng.
                - Lớp nhãn (Labels): Mỗi hình ảnh có một nhãn tương ứng (label) là một trong các chữ số từ $0$ đến $9$.
                """)
    st.markdown("Dưới đây là một số ảnh minh họa dataset **MNIST**")
    c = st.columns([2, 6, 2])
    image = cv.imread("./images/Handwriting_Letter_Recognize/dataset_MNIST.png")
    c[1].image(image, channels="BGR", caption="Minh họa dataset MNIST")
    st.header("2 Phương pháp")
    st.markdown(
            """
            - Mô hình được sử dụng để huấn luyện các kí tự trong dataset **MNIST** là một mô hình CNN gồm $7$ lớp bao gồm nhiều lớp **convolutional (conv)** và **pooling**, 
            với các lớp **fully connected (FC)** ở cuối để thực hiện phân loại.  
            """)
    st.markdown("#### 2.1 Kiến trúc của mô hình")
    image_achitecture = cv.imread("./images/Handwriting_Letter_Recognize/Achitecture_model.PNG")
    cc = st.columns([3, 5, 2])
    cc[1].image(image_achitecture, caption="Kiến trúc của mô hình", channels="BGR")
    st.markdown("##### 2.1.1 Feature extraction")
    st.markdown("**Feature extraction (trích xuất đặc trưng)** bao gồm nhiều lớp **Convolutional(Conv)** và **Max Pooling**")
    
    c = st.columns(2)
    with c[0]:
        st.markdown("**2.1.1.1 Lớp Convolutional (Conv)**")
        st.markdown(
                    """
                    - **Lớp Convolutional (Conv)** là một thành phần quan trọng trong mạng nơ-ron tích chập **(CNN)**. Chức năng chính của nó là trích xuất các đặc trưng từ dữ liệu đầu vào (ảnh hoặc tín hiệu)
                    thông qua **phép tích chập** giữa các bộ lọc **(filters)** và dữ liệu.
                    """
        )
        image_conv = cv.imread("./images/Handwriting_Letter_Recognize/convolutional_layer.png")
        st.image(image_conv, caption="Ảnh minh hoạ phép tích chập", channels="BGR")
    with c[1]:
        st.markdown("**2.1.1.2 Lớp Max pooling**")
        st.markdown(
                    """
                    - **Lớp Max Pooling** là một kỹ thuật hiệu quả để giảm kích thước dữ liệu, tăng tính bất biến với các biến đổi nhỏ và tăng cường trường tiếp nhận trong **CNN**. Nó giúp mô hình học các đặc trưng
                     hiệu quả hơn, giảm chi phí tính toán và hạn chế **overfitting**. Mặc dù có một số kỹ thuật **pooling** khác như **Average Pooling**, nhưng **Max Pooling** vẫn là phương pháp phổ biến và thường cho kết quả 
                     tốt hơn trong nhiều bài toán thị giác máy tính.
                    """
        )
        image_maxpooling = cv.imread("./images/Handwriting_Letter_Recognize/MaxPooling_layer.png")
        st.image(image_maxpooling, caption="Ảnh minh hoạ phép Max pooling", channels="BGR")
    st.markdown("Dưới đây là hình ảnh minh hoạ quá trình **trích xuất đặc trưng (Feature extraction)** của mô hình")
    image_visualize = cv.imread("./images/Handwriting_Letter_Recognize/visualizeCNN.PNG")
    st.image(image_visualize, caption="Hình ảnh minh hoạ quá trình trích xuất đặc trưng của mô hình", channels="BGR")
    st.markdown("##### 2.1.1 Classification")
    st.markdown(
                """
                - Sau khi trích xuất đặc trưng **(feature extraction)** từ các lớp **Convolutional(Conv)** và **Max pooling**, chức năng của phần **Classification** trong mô hình học sâu là sử dụng những đặc trưng này để phân loại 
                các đối tượng hoặc mẫu đầu vào vào các lớp cụ thể. Phần này thực hiện thông qua các lớp **Fully Connected (Dense)**, nơi các đặc trưng trích xuất được đưa vào và thông qua 
                quá trình học, chúng được sử dụng để đưa ra kết quả phân loại.
                """
    )
    st.markdown("Dưới đây là hình ảnh minh hoạ lớp **Fully Connected**")
    c = st.columns([3, 5, 2])
    image_FC = cv.imread("./images/Handwriting_Letter_Recognize/FullyConnected.PNG")
    c[1].image(image_FC, caption="Fully Connected", channels="BGR")
    st.markdown("#### 2.2 Lý do chọn kiến trúc này")
    st.markdown(
                """
                - Kiến trúc này được lựa chọn vì tính hiệu quả trong việc trích xuất đặc trưng từ dữ liệu đầu vào, giảm số lượng tham số, 
                và khả năng học các đặc trưng cần thiết cho bài toán phân loại chữ số. Đây là một cấu trúc cơ bản nhưng mạnh mẽ, thường được 
                sử dụng như bước khởi đầu trong các bài toán liên quan đến phân loại ảnh nhỏ.
                """
    )
    c = st.columns(2)
    with c[0]:
        st.markdown("##### 2.2.1 Sử dụng các khối **Convolutional**")
        st.markdown(
                    """
                    - Lớp **Conv2D:**
                        - **Vai trò**: Trích xuất các đặc trưng từ hình ảnh đầu vào, như cạnh, góc, và các chi tiết cấu trúc cơ bản.
                        - **Bộ lọc**: Số lượng bộ lọc tăng dần từ $32$ đến $64$. Điều này giúp mô hình học được nhiều đặc trưng phức tạp hơn khi đi sâu vào mạng.
                        - **Kích thước bộ lọc (3, 3)**: Một lựa chọn phổ biến để nắm bắt các chi tiết nhỏ trong ảnh.
                        - **padding='same'**: Đảm bảo đầu ra của lớp **Conv2D** có cùng kích thước không gian với đầu vào, giữ thông tin ở rìa ảnh.
                    - Hai lớp **Conv2D** liên tiếp trong khối:
                        - Việc xếp chồng hai lớp **Conv2D** giúp mô hình học được các đặc trưng sâu và phức tạp hơn, thay vì chỉ học các đặc trưng cơ bản từ một lớp duy nhất (phù hợp với anh xám có kích thước nhỏ **(28x28)** như **MNIST**).
                    """
        )
        st.markdown("##### 2.2.2 Giảm kích thước không gian bằng **MaxPooling**")
        st.markdown(
                    """
                    - Lớp **MaxPooling2D**:
                        - **Vai trò**: Giảm kích thước không gian (chiều rộng và chiều cao) của các đặc trưng, giúp giảm số lượng tham số và tăng tốc độ tính toán.
                        - Kích thước vùng **pooling (2, 2)** là một lựa chọn phổ biến để giảm một nửa kích thước không gian.
                        - Loại bỏ các thông tin không quan trọng, giữ lại các đặc trưng nổi bật nhất trong mỗi vùng.
                    """
        )
    with c[1]:
        st.markdown("##### 2.2.3 Chuyển đổi dữ liệu thành vector phẳng")
        st.markdown(
                    """
                    - Lớp **Flatten()**:
                        - **Vai trò**: Chuyển đổi **tensor 3D** (kích thước không gian và số lượng bộ lọc) thành **vector 1D** để làm đầu vào cho các lớp **fully connected**.
                    """
        )
        st.markdown("##### 2.2.4  Sử dụng các lớp **Fully Connected** để phân loại")
        st.markdown(
                    """
                    - Lớp **Dense(128, activation='relu')**:
                        - **Vai trò**: Học các mối quan hệ phi tuyến tính giữa các đặc trưng đã trích xuất và ánh xạ chúng tới các lớp phân loại.
                        - Số lượng đơn vị ẩn là $128$, đủ lớn để học được các đặc trưng phức tạp mà vẫn giữ tính hiệu quả tính toán.
                    - Lớp **Dense(10, activation='softmax')**:
                        - **Vai trò**: Lớp đầu ra với $10$ đơn vị tương ứng với $10$ lớp.
                        - Hàm kích hoạt **softmax**: Chuyển đổi các giá trị thành xác suất cho từng lớp, giúp mô hình dễ dàng phân loại.
                    """
        )
    Training()
    st.markdown("#### 3. Kết quả")
    Results()
    st.markdown("#### 4. Ứng dụng")
    Applications()

def App():
    Text()
App()
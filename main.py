import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import streamlit as st
import tempfile
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path

from PIL import Image

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')



model = tf.keras.models.load_model('model.h5', custom_objects={'Dropout': tf.keras.layers.Dropout}, compile=False)



# Đường dẫn đến model và thư mục chứa ảnh
model_path = "model.h5"
image_folder = "D:\MINDX\CSI\PythonProject\extracted_frames"  # Thư mục chứa các frame ảnh
# Các biến để lưu thời gian bắt đầu, kết thúc của đoạn có lửa




def FrameCapture(video_path, output_folder="extracted_frames", frame_interval=1):
    # Tạo (và xóa các file cũ nếu có) thư mục lưu frame
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    folder_path = Path(output_folder)
    for image_file in folder_path.glob("*.*"):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
            try:
                image_file.unlink()
                print(f"Deleted image: {image_file}")
            except Exception as e:
                print(f"Error: {image_file}: {e}")
    print("Delete completed!")

    start_time = None  # Thời gian bắt đầu khi phát hiện fire
    end_time = None  # Thời gian kết thúc khi fire vẫn còn

    vidObj = cv.VideoCapture(video_path)
    if not vidObj.isOpened():
        print("Can not open the video!")
        return

    fps = vidObj.get(cv.CAP_PROP_FPS)
    width = int(vidObj.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vidObj.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Thiết lập thư mục và file video đầu ra
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_video_path = os.path.join(output_dir, "output_video.mp4")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("FPS:", fps)
    count = 0  # Đếm số khung hình đã lưu/dự đoán
    frame_count = 0  # Đếm tổng số frame đọc được
    predictions = []  # Lưu giá trị dự đoán của từng frame

    # Ngưỡng màu HSV cho lửa (có thể điều chỉnh)
    lower_fire = np.array([0, 120, 150])
    upper_fire = np.array([35, 255, 255])

    # Tạo placeholder hiển thị frame theo thời gian thực trên Streamlit
    frame_placeholder = st.empty()

    while True:
        success, image = vidObj.read()
        if not success:
            break

        # Tạo bản sao để vẽ bounding box
        frame_copy = image.copy()

        # Xử lý theo khoảng cách frame_interval
        if frame_count % frame_interval == 0:
            # Lưu frame vào thư mục (nếu cần lưu)
            frame_filename = os.path.join(output_folder, f"frame{count:04d}.jpg")
            cv.imwrite(frame_filename, image)

            # Chuyển đổi màu từ BGR sang RGB cho model (nếu model huấn luyện trên RGB)
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            resized_image = cv.resize(image_rgb, (200, 200))
            test_image = img_to_array(resized_image)
            test_image = np.expand_dims(test_image, axis=0)

            # Dự đoán với model
            prediction = model.predict(test_image)
            prediction_value = prediction[0][0]
            predictions.append(prediction_value)
            count += 1

            current_time = frame_count / fps
            print(f"Extracted frame {count} - Prediction: {prediction_value}")

            # Nếu khả năng có fire cao (theo ngưỡng), thực hiện bounding detection
            if (1 - prediction_value) > 0.8:
                hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv, lower_fire, upper_fire)
                contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv.contourArea(cnt) > 500:
                        x, y, w, h = cv.boundingRect(cnt)
                        cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv.putText(frame_copy, "Fire", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if start_time is None:
                    start_time = current_time
                    print("START")
                end_time = current_time
            else:
                # Nếu không có fire, reset thời gian
                start_time, end_time = None, None

            # Ghi frame đã xử lý (có bounding box nếu có fire) vào video đầu ra
            out.write(frame_copy)
            # Cập nhật hiển thị frame theo thời gian thực trên giao diện Streamlit
            frame_placeholder.image(frame_copy, channels="BGR")

        frame_count += 1

    vidObj.release()
    out.release()
    print(f"\nExtracted {count} frame from video.")

    # Tổng hợp kết quả: sử dụng trung bình của các giá trị dự đoán
    if predictions:
        avg_prediction = sum(predictions) / len(predictions)
        print("Average prediction:", avg_prediction)
        if avg_prediction >= 0.8:
            print("Final result: No fire", avg_prediction)
            return st.success(f"No fire,Confi is : {round( avg_prediction,2)}")
        else:
            print("Final result:", 1 - avg_prediction)
            return st.success(f"Fire detected from {round(start_time, 2)}s to {round(end_time, 2)}s,  Confi is : {round( 1 - avg_prediction,2)}")
            return st.video("output/output_video.mp4")

    else:
        print("Error: No frames were processed!")
        st.warning("No frames were processed. Please check your video file.")


def predict_img(image, model) :
    # Kiểm tra nếu ảnh được tải thành công
    if image is None:
        print("Error: Unable to load image")
        return None

    # Kiểm tra số kênh ảnh và chuyển về định dạng BGR nếu cần
    if len(image.shape) == 2:  # Ảnh grayscale
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # Ảnh có kênh Alpha (RGBA)
        image = cv.cvtColor(image, cv.COLOR_RGBA2BGR)

    # Resize ảnh về kích thước 200x200
    resized_image = cv.resize(image, (200, 200))

    image_copy = image.copy()
    # Chuyển đổi ảnh copy sang không gian màu HSV
    hsv = cv.cvtColor(image_copy, cv.COLOR_BGR2HSV)

    # Định nghĩa phạm vi màu của lửa (có thể cần điều chỉnh lại giá trị)
    lower_fire = np.array([0, 135, 85])
    upper_fire= np.array([255, 180, 135])


    # Tạo mask lọc vùng lửa
    mask = cv.inRange(hsv, lower_fire, upper_fire)
    # Hiển thị mask để debug
    st.image(mask, caption="Fire Mask", use_container_width=True)

    # Tìm contours (đoạn biên) trong mask
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(f"Total contours detected: {len(contours)}")

    # Nếu có contours, vẽ bounding box
    if len(contours) > 0:
        # Tạo bản sao ảnh để vẽ lên
        image_copy = image.copy()

        for cnt in contours:
            thickness = 1
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), thickness)

            cv.putText(image_copy, "Fire", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness)

        # Hiển thị ảnh với bounding box
        st.image(image_copy, caption="Fire Detection Result", use_container_width=True)

    # Chuyển ảnh đã resize sang mảng (array) cho mô hình dự đoán
    test_image = img_to_array(resized_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Dự đoán với mô hình
    prediction = model.predict(test_image)
    prediction_value = prediction[0][0]

    print('Confidence:', prediction_value)

    if prediction_value >= 0.85:
        return st.write(f"No fire, Confi is: {prediction_value:.2f}")

    else:

        return st.success(f"Fire detected, Confi is: {(1 - prediction_value):.2f}")








#Giao diện web với Streamlit



st.title("🔥 AI FireGuard: Real-time Fire Detection System")

# Tải video lên
uploaded_file = st.file_uploader("Upload one video", type=["mp4", "mov", "avi", "mkv"])
uploaded_file_img = st.file_uploader("Upload one photo", type=["jpg","png"])


if uploaded_file is not None:
    # Hiển thị video đã tải lên
    st.video(uploaded_file)


    # Lưu video tạm thời để xử lý
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Nút chạy mô hình A
    if st.button("Run Video Model"):
        if __name__ == '__main__':
            # video_path = "D:\MINDX\CSI\PythonProject\Fire, Burn, Flames. Free Stock - Pixabay.mp4"
            output_folder = "extracted_frames"  # Tên thư mục bạn muốn lưu ảnh vào
            #frame_interval = 1  # Trích xuất mỗi 5 khung hình

            FrameCapture(video_path,output_folder)



    # Xóa file tạm sau khi xử lý (tùy chọn)
    os.unlink(video_path)
if uploaded_file_img is not None:
    # Mở ảnh bằng PIL
    image = Image.open(uploaded_file_img)

    # Hiển thị ảnh trên giao diện Streamlit
    st.image(image, caption="Uploaded photo")

    # Chuyển đổi ảnh sang numpy array để đưa vào predict_img()
    image_np = np.array(image, dtype=np.uint8)  # Chuyển về numpy

    # Chạy mô hình khi nhấn nút
    if st.button("Run Image Model"):
        predict_img(image_np, model)   # Truyền ảnh numpy vào





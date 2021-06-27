Trong bài này sẽ thực hiện face detection với thư viện `Dlib` bằng cách sử dụng HOG + Linear SVM và CNNs.
* **HOG + Linear SVM face detector** chính xác và hiệu năng tính toán tốt. `dlib.get_frontal_face_detector()`. Cái này không nhận parameters nào
* **Max-Margin (MMOD) CNN face detector vừa chính xác vừa nhanh, có khả năng phát hiện khuôn mặt với nhiều góc nhìn, điều kiện sáng và occlusion (gần nhau). MMOD face detector có thể chạy trên NVIDIA GPU. `dlib.cnn_face_detection_model_v1(modelPath)`. Model path là path đến pre-trained `mmod_human_face_detector.dat` nằm trong ổ cứng (mình tải về trước).

OpenCv và dlib có cách biểu diễn bounding boxes khác nhau:
* Trong OpenCV ta nghĩ bounding box gồn 4 tọa độ x_min, y_min, x_mãx, y_max
* Trong Dlib bounding box được biểu diễn bằng `rectangle` object với left, top, right, bottom properties.
Nhiều khi bounding boxes mà Dlib trả về nằm ngoài giới hạn của ảnh, do đó trong `helpers.py` chứa hàm giúp chuyển bounding boxes twuf Dlib về OpenCV và cắt bỏ các tọa độ bounding box nằm ngoài kích thước ảnh.

Trong bài này cần xác định số lần upsample (tăng kích thước ảnh) giúp phát hiện các khuôn mặt nhỏ. Việc này sẽ làm chậm quá trình phát hiện (tăng số ảnh trong image pyramid) (ở đây dùng sliding window).

Nhận thấy HOG + Linear SVM mất khoảng 0.1 s để thực hiện dự đoán, điều này có nghĩa chúng ta có thể xử lý 10 frames/seconds trên video. Tuy nhiên thử với đám đông nó không phát hiện được một số khuôn mặt nghiêng.

Nhận thấy MMOD CNN face detector có thể phát hiện được rất nhiều khuôn mặt tỏng đám đông. Tuy nhiên tốc độ xử lý của nó chậm hơn so với HOG+SVM Linear. Nếu chúng ta có NVIDIA GPU thì tốc độ sẽ được cải thiện hơn nhiều mà độ chính xác vẫn được đảm bảo.

### Tài liệu tham khảo
1. https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/







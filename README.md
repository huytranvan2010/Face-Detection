Chúng ta sẽ đi đánh giá 4 phương pháp face detection được thể hiện trong này.
Trong thư mục này tổng hợp 4 face detector rất phổ biến hiện nay bao gồm:
1. OpenCV and Haar cascades
2. OpenCV's deep learning-bases face detector (pre-trained model)
3. Dlib's HOG _ Linear SVM
4. Dlib's CNN face detector

#### OpenCV and Haar cascades
**Pros**
* Rất nhanh chạy real-time được
* Cần ít tài nguyên tunhs toán, dễ dàng chạy trên các thiết bị như Raspberry, Jetson Nano...
* Model size nhỏ (vài trăm KB)

**Cons**
* Pháp hiện nhiều False-positive detection
* Phải tune các parameters trong `detectMultiscale`

Nên dùng khi tài nguyên tính toán ít, cần real-time, chấp nhận độ chính xác không cao.

### OpenCV's deep learning-bases face detector (pre-trained model)
OpenCV's deep learning-bases face detector dựa trên SSD với ResNet làm backbone.
**Pros**
* Độ chính xác cao
* Sử dụng DL algorithm
* Không phải tune parameters
* Có thể chạy real-time trên laptops, desktops
* Kích thước model vừa phải (10 MB)
* Dựa trên `cv2.dnn`
* **Có thể thực hiện nhanh hơn trên thiết bị nhúng bằng cách sử dụng OpenVINO và Movidius NCS**

**Cons**
* Chính xác hơn Haar cascades và HOG + Linear SVM nhưng không chính xác bằng dlib's CNN MMOD face detector
* Có một số bias trong training set như khó nhận mặt da sẫm màu hơn

### Dlib's HOG + Linear SVM
Dlib's HOG + Linear SVM dựa trên image pyramids và sliding windows để detect objects/faces trong ảnh. Đây là phương pháp cổ điển, tuy nhiên vẫn được dùng nhiều cho đến nay.

**Pros**
* Chính xác hơn Haar cascades
* Ổn định hơn Haar cascades (ít phải tune parameters)
* Dựa trên `dlib`

**Cons**
* Chỉ hoạt động với mặt nhìn trực diện, khi thay đổi góc nhìn khó phát hiện
* Cần cài thêm `dlib`
* Cần nhiều tài nguyên tính toán  (do có immage pyramids)
* Không chính xác như deep learning based face detector

### dlib’s CNN face detector
**Pros**
* Chính xác nhất trong số 4 loại
* Kích thước model nhỏ (dưới 1 MB)
* Dựa trên `dlib`

**Cons**
* Code khá loằng ngoằng ví dụ cần chuyển đổi bounding boxes
* Cần cài thêm `dlib`
* Không chạy được real-time nếu không có GPU
* Không tương thích với trình tăng tốc thông qua OpenVINO, Movidius NCS, NVIDIA Jetson Nano or Google Coral.

Khi xây dựng [training sét for face recognition]() có thể dùng dlib's CNN face detector trước khi training (để tách các khuôn mặt ra). Tuy nhiên khi triển khai thực tế hơi khó, lúc này lại nghĩ đến OpenCV' deep learning based face detector.

Chung quy lại thấy OpenCV's DNN face detector có thể làm tốt hơn:
* Đạt được cân bằng giữa tốc độ và độ chính xác,
* Chính xác hơn Haar cascades, HOG + Linear SVM
* Chạy real-time đủ tốt trên CPUs
* Có thể tăng tốc bằng cách sử dụng Movidius NCS
* Không cần các thư viện khác ngoài OpenCV thông qua `cv2.dnn`.

### Tài liệu tham khảo
1. https://www.pyimagesearch.com/2021/04/26/face-detection-tips-suggestions-and-best-practices/
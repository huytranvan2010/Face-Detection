Trong bài này chúng ta cùng thực hiện face detectionn với OpenCV có sử dụng pre-trained deep learning face detector model. OpenCV có module `dnn` hỗ trợ một số frameworks như Tensorflow, Caffe, PyTorch.

Caffe-based face detector có thể được tìm thấy tại đây [dnn](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

Để phát hiện khuôn mặt cới pretrained model từ Caffe chúng ta cần 2 file:
* `.prototxt` chứa định nghĩa về model architecture (layers...). File này chúng ta có thể tìm thấy ở link trên
* `caffemodel` chứa các weights của các layers. File này không có trong link trên

Caffe face detector sử dụng ở đây dựa trên Signle Shot Detector (SSD) với ResNet làm backbone.

Tìm hiểu thêm về `cv2.dnn.blobFromImage()` tại đây https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/ 

Ở bên phần code cũng đã chú thích ảnh đầu vào được resize về `300x300`. Bounding box được dựa đoán là xác tọa độ `x_min, y_min, x_max, y_max` tương đối so với ảnh đầu vào `300x300`. Ví dụ nhận được bounding box mặt người với các gía trị [0.1, 0.1, 0.2, 0.2], khi đó tọa độ của bounding box trong ảnh đầu vào là [30, 30, 60, 60]. Tuy nhiên tọa độ bounding box trong ảnh thực tế là [60, 60, 120, 120]. Đó là lý do trước khi đưa ảnh vào model chúng ta lấy kích thước ảnh gốc. Và khi có tọa độ bounding box dự đoán chúng ta lại nhân với kích thước ảnh gốc (thay vì kích thước ảnh đầu vào).

Muốn hiển thị trên ảnh đầu vào thì thay 4 câu lệnh này vào 4 câu lệnh tương ứng trong source code.
```python
box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
cv2.rectangle(resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.putText(resized, text, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
cv2.imshow("Face", resized)
```
Nhận thấy face detector này phát hiện được cả các khuôn mặt bị nghiêng. Điều này gần như không thể với Haar cascade. Thử đối với ảnh đám đông `crowd.jpg` thì không phát hiện được một số khuôn mặt. 

### Tài liệu tham khảo
1. https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ 
# USAGE
# python detect_faces.py --image crowd.jpg --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import các thư viện
import numpy as np
import argparse
import cv2

# tạo parse để truyền các tham số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")    # loại bỏ detection có pro thấp hơn
args = vars(ap.parse_args())

# load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load image, tạo blob từ image để đưa vào model
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]    # lấy kích cỡ ảnh ban đầu
resized = cv2.resize(image, (300, 300))     # size (300, 300) thường cho SSD và Faster-RCNN
blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 177.0, 123.))  # image, scalefactor, size vào model, mean values của RGB, nếu cần swapRB thì thêm True

# đưa blob vào mạng để lấy pređiction
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()      # 4 dimensions, chiều thứ 3 là số detections

# duyệt qua các detections
for i in range(detections.shape[2]):
    # laays confidence (probability) liên quan đến detection
    confidence = detections[0, 0, i, 2]     # detections[0, 0, i, :2] là batchID và classID 
    # loại bỏ bớt các detection có confidence thấp
    if confidence < args["confidence"]:
        continue
    
    # do kết quả trả về là tỉ lệ so với "input image" đưa vào mạng, do muốn vẽ bounding lên ảnh gốc thì phải nhân với kích thước ảnh gốc
    # đó là lý do vì sao ảnh được resize khi đưa vào model nhưng khi vẽ vẫn cho kết quả đúng trên ảnh gốc
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    # chuyển về int (theo pixel)
    (x_min, y_min, x_max, y_max) = box.astype("int")

    # vẽ box cùng với confidence
    text = "{:.2f}%".format(confidence * 100)
    y = y_min - 10 if y_min - 10 > 10 else y_min + 10
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, text, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imshow("Face", image)
cv2.waitKey(0)

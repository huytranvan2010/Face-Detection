# USAGE
# python cnn_face_detect.py --image images/crowd.jpg --model mmod_human_face_detector.dat

from hammiu.helpers import convert_anh_trim_bb
import argparse
import imutils
import time
import dlib 
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, default="mmod_human_face_detector.dat", help="path to dlib's CNN face detector model")
# nếu ko upsample thì để 0 (ko tăng kích thước ảnh)
ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")     # số lần upsample trước khi detect, giúp detect mặt nhỏ
args = vars(ap.parse_args())

# load dlib's HOG + Linear SVM face detector
print("[INFO] loading CNN face detector...")
detector = dlib.cnn_face_detection_model_v1(args["model"])

# load image, resize, chuyển về RGB (dlib cần)
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)    # để chạy nhanh hơn
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform face detection using dlib's face detector
start = time.time()
print("[INFO] performing face detection with dlib...")
results = detector(rgb, args["upsample"])
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))

# chuyển dlib boxesvề OpenCV boxes
# chú ý MMOD CNN object detector trả về list of result objects, mỗi cái lại có rectangle của nó
boxes = [convert_anh_trim_bb(image, r.rect) for r in results]      # do có thể trả về nhiều khuôn mặt

for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv2.imshow("Output", image)
cv2.waitKey(0)
import cv2
import face_recognition

image0 = "images/juan-gabriel/fake-jg"
image1 = "images/juan-gabriel/jg0"

img = cv2.imread(image0)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread(image1)
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)
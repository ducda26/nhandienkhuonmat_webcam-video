import cv2
import face_recognition
import os
import numpy as np

# Step 01 Load ảnh từ kho ảnh nhận dạng

path = "pic2"
images = []
classNames = []
myList = os.listdir(path)
# ['Donal Trump.jpg', 'elon musk .jpg', 'Joker.jpg', 'tokuda.jpg']
print(myList)

for cl in myList:
    print(cl)  # pic2/Donal Trump.jpg
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
print(len(images))
print(classNames)

# Step 02: Xac dinh vi tri và ma hoa


def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  # 0: lấy 1 bức ảnh
        encodeList.append(encode)
    return encodeList


encodeListKnow = Mahoa(images)
print("Mã hóa thành công")
print(len(encodeListKnow))


# Khởi động webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame, (0, 0), None, fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    # Xác định vị trí khuôn mặt trên Cam và encode hình ảnh trên cam
    # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS)

    # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại theo cặp
    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)

    cv2.imshow("Anh Duc", frame)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cap.release()
cv2.destroyAllWindows()

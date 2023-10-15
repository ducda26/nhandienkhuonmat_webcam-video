import cv2
import face_recognition
import os
import numpy as np

#Step 01 Load ảnh từ kho ảnh nhận dạng

path = "pic2"
images = []
classNames = []
myList = os.listdir(path)
print(myList)# ['Donal Trump.jpg', 'elon musk .jpg', 'Joker.jpg', 'tokuda.jpg']

for cl in myList:
    print(cl) # pic2/Donal Trump.jpg
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
print(len(images))
print(classNames)

#Step 02: Xac dinh vi tri và ma hoa
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] #0: lấy 1 bức ảnh
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)
print("Mã hóa thành công")
print(len(encodeListKnow))
    
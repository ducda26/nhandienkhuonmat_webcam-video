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
    
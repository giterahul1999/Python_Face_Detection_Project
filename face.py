import cv2
import numpy as np
import face_recognition_models as fr
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video= cv2.VideoCapture(0)

T=True
while T:

    check,frame=video.read()


    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    kernal = np.ones((10, 10), np.float32) / 100
    blur = cv2.filter2D(frame,-1,kernal)

    faces=face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y + h, x:x + w]
        img_item="my_image.png"
        cv2.imwrite(img_item,roi_gray)


        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),3)


    cv2.imshow("Normal",frame)
    #cv2.imshow("B and W",gray)
    cv2.imshow("blur",blur)
    print(frame)


    key=cv2.waitKey(1)
    if key==ord('a'):
        T=False


video.release()
cv2.destroyAllWindows()

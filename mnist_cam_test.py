import numpy as np
import cv2
import pandas as pd
import os
from keras.models import load_model

model = load_model('model.h5')

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    x, y, w, h = 0, 0, 300, 300
    ret, im = cap.read()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

    ret, thresh1 = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hier= cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                newImage = thresh1[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage = np.array(newImage)
                im1=pd.DataFrame(newImage,dtype='float64')
                im2=im1.values.reshape(-1,28,28,1)
                im2-=np.mean(im2,axis=1)
                nbr = model.predict(im2)
                nb = np.argmax(nbr)


    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(im, str(nb), (10,380),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv2.putText(im, str(nbr), (5,300),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    

    cv2.imshow("Contours", thresh1)
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    k = cv2.waitKey(10)
    if k == 27:
       break
    
cap.release()
cv2.destroyAllWindows()
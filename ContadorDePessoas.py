import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fgbg = cv2.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("nao abriu")
    exit()
    
windowName = "webcam"    
    

while 1:
    ret, frame = cap.read()
    
    if not ret:
        print("Sem frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = fgbg.apply(gray)
    
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
     
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    dilatation = cv2.dilate(opening, kernel, iterations = 8)
    
    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations = 8)
    
    countours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in countours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        cv2.rectangle(frame,(x,y), (x + w, y + h), (0,255,9), 2)
    cv2.imshow("frame", frame)
    cv2.imshow("fgmask", fgmask)
    cv2.imshow(windowName, gray)
    cv2.imshow("th", th)
    
    
    k = cv2.waitKey(1)
    
    if k == ord('s'):
        break
    
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        break
    
cv2.destroyAllWindows()
cap.release()


    

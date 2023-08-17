import numpy as np
import cv2

def center (x, y, w,h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x +x1
    cy = y + y1
    return cx, cy
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fgbg = cv2.createBackgroundSubtractorMOG2()


if not cap.isOpened():
    print("nao abriu")
    exit()
    
windowName = "webcam"

posL = 250
offset = 30

xy1 = (0, posL)  # Extremidade esquerda da tela

detects = []

total = 0

up = 0

down = 0


while 1:
    ret, frame = cap.read()
    
    if not ret:
        print("Sem frame")
        break
    
    xy2 = (frame.shape[1], posL)  # Extremidade direita da tela (largura do quadro)s
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.line(frame, xy1, xy2, (255, 0, 0), 3)
     
    cv2.imshow("frame", frame)
    
    fgmask = fgbg.apply(gray)
    
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
     
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    dilatation = cv2.dilate(opening, kernel, iterations = 8)
    
    closing = cv2.morphologyEx(dilatation, cv2.MORPH_CLOSE, kernel, iterations = 8)
    
    countours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, xy1, xy2,(255,0,0), 3)
    
    cv2.line(frame,(xy1[0],posL-offset),(xy2[0],posL-offset),(255,255,0),2)
    
    cv2.line(frame,(xy1[0],posL+offset),(xy2[0],posL+offset),(255,255,0),2)
    
    i = 0
    for cnt in countours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        detect = []
        
        if int(area) < 3000:
           centro = center(x, y, w, h)
        
           cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
           cv2.circle(frame, centro, 4, (0, 0, 255), -1)
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 9), 2)
        
           if len(detects) <= i:
               detects.append([])
           if centro[1]> posL-offset and centro[1]< posL+offset:
               detects[i].append(centro)
           else:
               detects[i].clear()    
               
        
           i += 1        
    if i == 0:
        detect.clear()
 
    
    if len(countours) == 0:
       for detect in detects:
          detects.clear()  # Limpar todas as listas em detects
    else:
       for detect in detects:
           if len(detect) > 0:
               for (c, i) in enumerate(detect):
                if detect[c - 1][1] < posL and detect[c][1] > posL:
                    detect.clear()
                    up += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                    continue
                 
                if detect[c - 1][1] > posL and detect[c][1] < posL:
                    detect.clear()
                    down += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                    continue
                
                if c > 0 :
                    cv2.line(frame, detect[c - 1],detect[c - 1],(0,0,255),1)
                    
    cv2.putText(frame, "total: "+str(total),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)   
    cv2.putText(frame, "subindo: "+str(up),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)    
    cv2.putText(frame, "descendo: "+str(down),(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)               
                       
        
    cv2.imshow("frame", frame)
    cv2.imshow(windowName, gray)
    cv2.imshow("closing", closing)
    
    
    k = cv2.waitKey(1)
    
    if k == ord('s'):
        break
    
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        break
    
cv2.destroyAllWindows()
cap.release()
cap.release()


    

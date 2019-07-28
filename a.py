import numpy as np
import cv2, time


face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

cap  = cv2.VideoCapture(0)
time.sleep(2)

while(True):

    ret, frame = cap.read()

    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        #cordenadas de la cara junto con el alto y ancho print(x,y,w,h)

        #region de insteres
        roi_gray = gray[y:y+h, x:x+w] #(yCord_start, yCord_end)
        roi_color = frame[y:y+h, x:x+w]
        #nombre de la imagen de la region de interes
        img_item ="wea.png"
        cv2.imwrite(img_item,roi_gray)#esto guarda la imagen

        #rectangulo de coloracion
        color_rect = (0,0,255) #BGR
        stroke = 2
        end_cord_x = x+w
        end_cord_y =y +h

        cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y),color_rect, stroke)

    cv2.imshow('narnia',frame)

    #espera 20 sec para leer una entrada de teclado para salir

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
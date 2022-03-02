import cv2
import numpy as np
import time

h, w = 480, 640
black = np.zeros((h,w,3), dtype=np.uint8)
foto = black

folder_data = "photo/"

#image frame
ymin, ymax = h//2 - 125, h//2 + 125
xmin, xmax = w//2 - 125, w//2 + 125

n_image = 0 
capture_delay = 0.5 # second

filename = ''
label_frame = ''

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,w)
cap.set(4,h)

last_time = time.time()

def close():
    cap.release()
    cv2.destroyAllWindows()
    
count = 0
while(True):
    try :
        ret, frame = cap.read()

        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,wf,hf) in faces:
            # create box on detected face
            roi = frame[y:y+hf, x:x+wf]
            cv2.imwrite("E:\\tester\capture\\"+str(count)+'.jpg', roi)
            count += 1
            print (count)
        
        cv2.imshow('frame',frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            close()
            break
            
    except Exception as e: 
        print(e)     
        close()
        break
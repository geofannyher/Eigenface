import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

##untuk raspi
#import RPi.GPIO as GPIO

#RELAY = 17
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(RELAY, GPIO.OUT)
#GPIO.output(RELAY, GPIO.LOW)

def show_dataset(images_class, label):
    # show data for 1 class
    plt.figure(figsize=(14,5))
    k = 0
    for i in range(1,6):
        plt.subplot(1,5,i)
        try :
            plt.imshow(images_class[k][:,:,::-1])
        except :
            plt.imshow(images_class[k], cmap='gray')
        plt.title(label)
        plt.axis('scaled')
        plt.tight_layout()
        k += 1
    #plt.show()

dataset_folder = "dataset/"

names = []
images = []
for folder in os.listdir(dataset_folder):
    for name in os.listdir(os.path.join(dataset_folder, folder))[:9]: # limit only 9 face per class
        img = cv2.imread(os.path.join(dataset_folder + folder, name))
        images.append(img)
        names.append(folder)
labels = np.unique(names)

labels
for label in labels: 
    ids = np.where(label== np.array(names))[0]
    images_class = images[ids[0] : ids[-1] + 1]
    show_dataset(images_class, label)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img, idx):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    try :
        x, y, w, h = faces[0]

        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (100, 100))
    except :
        #print("Face not found in image index", i)
        img = None
    return img

croped_images = []
for i, img in enumerate(images) :
    img = detect_face(img, i)
    if img is not None :
        croped_images.append(img)
    else :
        del names[i]

for label in labels:
    
    ids = np.where(label== np.array(names))[0]
    images_class = croped_images[ids[0] : ids[-1] + 1] # select croped images for each class
    show_dataset(images_class, label)

name_vec = np.array([np.where(name == labels)[0][0] for name in names])
print(name_vec)

model = cv2.face.EigenFaceRecognizer_create()
model.train(croped_images, name_vec)
model.save("Eigenface.yml")
model.read("Eigenface.yml")

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            
            idx, confidence = model.predict(face_img)
            #label_text = "%s (%.2f %%)" % (labels[idx], confidence)

            if (confidence < 3000):
                label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                #id = label[idx]
                #confidence = " {0}%".format(round(100 - confidence))
                frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
### untuk raspi
                #Membuka Kunci Pintu
                print("Door Unlock")
                #GPIO.output(RELAY, GPIO.HIGH)
                #prevTime = time.time()
                #doorUnlock = True

                #Menutup Kunci Pintu

                #if doorUnlock == True and time.time() - prevTime > 5:
                    #doorUnlock = False
                    #GPIO.output(RELAY, GPIO.LOW)
            else:
                label_text = "unknown"
                frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
                print("Acces Denied")
                #confidence = " {0}%".format(round(100 - confidence))
            
            #frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
       
        cv2.imshow('Detect Face', frame)
    else :
        break
    if cv2.waitKey(10) == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
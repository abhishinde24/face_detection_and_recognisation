import cv2
import numpy as numpy
import matplotlib.pyplot as plt
# for face detection we use viola and jones method

#############################ML-setup-for-face-recognisation#################
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)###importing faces training data set
##making piple for workflow
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150,whiten=True,random_state=42)
svc = SVC(kernel='rbf',class_weight='balanced')
model = make_pipeline(pca,svc)
###Extracting features with gridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C':[1,5,10,50],
              'svc__gamma': [0.0001,0.0005,0.001,0.005]}
grid = GridSearchCV(model,param_grid)
grid.fit(faces.data,faces.target)
###

model = grid.best_estimator_


def face_reconginasation(img_gray):
    faces = fetch_lfw_people(min_faces_per_person=60)
    #this for converting (563,429) size img to (62,47)
    img_new=cv2.resize(img_gray,(62,47))

    # plt.imshow(img_new,cmap='gray')
    #this for converting (62,47) shape img to (1,2914)
    img_new=img_new.reshape(1,2914)
    
    # img_new=img_new[np.newaxis,:,:]
    yfits = model.predict(img_new)
    # plt.imshow(img_new.reshape(47,62),cmap='bone')
    return(faces.target_names[yfits[0]])




######################Face-detection-part####################################
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
cap.set(3,640)
#3 id for width
cap.set(4,480)
#4 id for height
cap.set(10,10)
#10 id for brightness

#while as video is continous images
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    success,imga=cap.read()
    #img is variable where img is store and success is boolean type variable stores img is store properly or not
    # faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgaGray=cv2.cvtColor(imga,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgaGray,1.1,4)
    for (x,y,w,h) in faces:
        # temp=imgaGray.copy()
        # name=face_reconginasation(temp[y:y+h,x:x+w])
        cv2.rectangle(imga,(x,y),(x+w,y+h),(255,0,0),2)
        ###################calling face-recognisation function defined above################################
        cv2.putText(imga,face_reconginasation(imgaGray[y:y+h,x:x+w]),
                         (x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),1)

    cv2.imshow("video",imga)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
################################
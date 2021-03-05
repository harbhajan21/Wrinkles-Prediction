import cv2
import os
from mtcnn import MTCNN
import argparse

### Path to input image ###
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
source_dir = cv2.imread(args["image"])

#image = cv2.imread(args["image"])
##For Face Cropping
detector = MTCNN()
mode=1   
face_source_dir = source_dir#"./face/"
dest_dir = "./cropped_face/"

#Function to crop the face box
def crop_face_image(face_source_dir, dest_dir, mode):
    if os.path.isdir(dest_dir)==False:
        os.mkdir(dest_dir)
    detector = MTCNN()
    img = source_dir#cv2.imread(source_dir)
    data=detector.detect_faces(img)
    if mode==1:  #detect the box with the largest area
        for i, faces in enumerate(data): # iterate through all the faces found
            box=faces['box']  # get the box for each face                
            biggest=0                    
            area = box[3]  * box[2]
            if area>biggest:
                biggest=area
                bbox=box 
        bbox[0]= 0 if bbox[0]<0 else bbox[0]
        bbox[1]= 0 if bbox[1]<0 else bbox[1]
        img=img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]] 
        cv2.imwrite("./cropped_face/cropped_face.png", img)

    return 1

#Function Call
crop_face_image(face_source_dir, dest_dir, mode)

#Eye HaarCascade Model
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_source_dir = source_dir
eye_dest_dir = "./cropped_eye/"

#Function to crop the EYE box
def crop_eye_image(eye_source_dir, eye_dest_dir, mode):
    if os.path.isdir(eye_dest_dir)==False:
        os.mkdir(eye_dest_dir)

    img = source_dir#cv2.imread("./face/cropped_raw_img.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
        cv2.rectangle(img, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height + 10), (0, 255, 0), 2)

    if mode==1:  #detect the box with the largest area
        for i, eye in enumerate(eyes): # iterate through all the eye found
            biggest=0                    
            area = eye_height  * eye_width
            if area>biggest:
                biggest=area
                
        eye_startX= 0 if eye_startX<0 else eye_startX
        eye_startY= 0 if eye_startY<0 else eye_startY
        #img=img[eye_startY: eye_startY+eye_height+10, eye_startX: eye_startX+ eye_width]
        img=img[eye_startY+eye_height-17: (eye_startY)+(eye_height+8), eye_startX: eye_startX+ eye_width]
        cv2.imwrite("./cropped_eye/cropped_eye.png", img)
            
    return 1

#Function Call
crop_eye_image(eye_source_dir, eye_dest_dir, mode)

#Directories for forehead 
forehead_source_dir = source_dir
forehead_dest_dir = "./cropped_forehead/"
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Function to crop the Forehead Part
def crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode):
    if os.path.isdir(forehead_dest_dir)==False:
        os.mkdir(forehead_dest_dir)

    imgs = source_dir#cv2.imread("./face/cropped_raw_img.png")
    gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
        cv2.rectangle(imgs, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height), (0, 255, 0), 2)

    if mode==1:  #detect the box with the largest area
        for i, eye in enumerate(eyes): # iterate through all the eye found
            biggest=0                    
            area = eye_height * eye_width
            if area>biggest:
                biggest=area
                
        eye_startX= 0 if eye_startX<0 else eye_startX
        eye_startY= 0 if eye_startY<0 else eye_startY
        #img=img[eye_startY: eye_startY+eye_height+10, eye_startX: eye_startX+ eye_width]
        imgs=imgs[eye_startY-40: eye_startY-14, eye_startX+10: eye_startX+60]
        cv2.imwrite("./cropped_forehead/cropped_forehead.png", imgs)
            
    return 1

#Function Call
crop_forehead_image(forehead_source_dir, forehead_dest_dir, mode)

#Directories for cheeks 
cheek_source_dir = source_dir
cheek_dest_dir = "./cropped_cheek/"
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Function to crop the Forehead Part
def crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode):
    if os.path.isdir(cheek_dest_dir)==False:
        os.mkdir(cheek_dest_dir)
    
    imgsc = source_dir#cv2.imread("./face/cropped_raw_img.png")
    gray = cv2.cvtColor(imgsc, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for (eye_startX, eye_startY, eye_width, eye_height) in eyes:
        cv2.rectangle(imgsc, (eye_startX, eye_startY), (eye_startX + eye_width, eye_startY + eye_height), (0, 255, 0), 2)

    if mode==1:  #detect the box with the largest area
        for i, eye in enumerate(eyes): # iterate through all the eye found
            biggest=0                    
            area = eye_height * eye_width
            if area>biggest:
                biggest=area
                
        eye_startX= 0 if eye_startX<0 else eye_startX
        eye_startY= 0 if eye_startY<0 else eye_startY
        #img=img[eye_startY: eye_startY+eye_height+10, eye_startX: eye_startX+ eye_width]
        imgsc=imgsc[eye_startY+eye_height+17: eye_startY+eye_height+42, eye_startX-20: eye_startX + 15]
        cv2.imwrite("./cropped_cheek/cropped_cheek.png", imgsc)
            
    return 1

#Function Call
crop_cheek_image(cheek_source_dir, cheek_dest_dir, mode)

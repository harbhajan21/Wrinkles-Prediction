import cv2
import os
import numpy as np
import streamlit as st
from mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
###########################################################################
st.header("Wrinkles & Dark Circles Prediction App")
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpeg", "jpg"])
if st.checkbox("Show Image"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    else:
        st.error("Please, Upload a valid file!")

############################################################################################################################
st.subheader("To check percentage of Wrinkles and Dark circles Prediction. Please click in below box.")
if st.checkbox("Check Prediction"):
    try:
        source_dir = img_array#cv2.imread(img)#cv2.imread(uploaded_file)#cv2.imread("./face/test.jpg")#cv2.imread(args["image"])
        
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
        
        
        ######### Dark Circle Prediction ##########
        #Beolw Eye Part
        img_drk_crkl = cv2.imread('./cropped_eye/cropped_eye.png')
        img_drk_crkl = cv2.cvtColor(img_drk_crkl, cv2.COLOR_BGR2RGB)
        img_drk_crkl = cv2.GaussianBlur(img_drk_crkl, (5,5), 0)
        
        img_drk_crkl_g = cv2.cvtColor(img_drk_crkl, cv2.COLOR_RGB2GRAY)
        sobely_img_drk_crkl = cv2.Sobel(img_drk_crkl_g, cv2.CV_8UC1, 0, 1, ksize=3)
        drkC_hy, drkC_wy = sobely_img_drk_crkl.shape
        param_img_drk_crkl_1 = cv2.sumElems(sobely_img_drk_crkl)[0]/(drkC_hy * drkC_wy)
        st.write("Value of Eye Part is ", param_img_drk_crkl_1)
        
        
        ##Forehead Part
        img_forehead = cv2.imread('./cropped_forehead/cropped_forehead.png')
        img_forehead = cv2.cvtColor(img_forehead, cv2.COLOR_BGR2RGB)
        img_forehead = cv2.GaussianBlur(img_forehead, (5,5), 0)
        
        img_forehead_g = cv2.cvtColor(img_forehead, cv2.COLOR_RGB2GRAY)
        sobely_img_forehead = cv2.Sobel(img_forehead_g, cv2.CV_8UC1, 0, 1, ksize=3)
        fix_hy, fix_wy = sobely_img_forehead.shape
        param_img_forehead_1 = cv2.sumElems(sobely_img_forehead)[0]/(fix_hy * fix_wy)
        st.write("Value of Forehead Part is ", param_img_forehead_1)
        
        ##cheeks Part
        img_cheeks = cv2.imread('./cropped_cheek/cropped_cheek.png')
        img_cheeks = cv2.cvtColor(img_cheeks, cv2.COLOR_BGR2RGB)
        img_cheeks = cv2.GaussianBlur(img_cheeks, (5,5), 0)
        
        img_cheeks_g = cv2.cvtColor(img_cheeks, cv2.COLOR_RGB2GRAY)
        sobely_img_cheeks = cv2.Sobel(img_cheeks_g, cv2.CV_8UC1, 0, 1, ksize=3)
        chk_hy, chk_wy = sobely_img_cheeks.shape
        param_img_cheeks_1 = cv2.sumElems(sobely_img_cheeks)[0]/(chk_hy * chk_wy)
        st.write("Value of Cheeks Part is ", param_img_cheeks_1)
        
        avg_skin_value = (param_img_forehead_1 + param_img_cheeks_1)/2
        st.write("Average skin value is ", avg_skin_value)
        
        #Loading Model
        model  = load_model("model_new.h5")
        
        img_fix = cv2.imread('Young.jpg')
        img_fix = cv2.cvtColor(img_fix, cv2.COLOR_BGR2RGB)
        img_fix = cv2.GaussianBlur(img_fix, (5,5), 0)
        img_fix_g = cv2.cvtColor(img_fix, cv2.COLOR_RGB2GRAY)
        sobely_fix = cv2.Sobel(img_fix_g, cv2.CV_8UC1, 0, 1, ksize=3)
        fix_hy, fix_wy = sobely_fix.shape
        param_fix_1 = cv2.sumElems(sobely_fix)[0]/(fix_hy * fix_wy)
        sobelx_fix = cv2.Sobel(img_fix_g, cv2.CV_8UC1, 1, 0, ksize=3)
        fix_hx, fix_wx = sobelx_fix.shape
        param_fix_2 = cv2.sumElems(sobelx_fix)[0]/(fix_hx * fix_wx)
        
        #image Path
        imagepath = "./cropped_face/cropped_face.png"
        
        # load the input image from disk
        images = cv2.imread(imagepath)
        
        orig = images.copy()
        (h, w) = images.shape[:2]
        
        face = orig
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
        (Adults, Children, MiddleAges, Old, Youth) = model.predict(face)[0]
        #print(model.predict(face)[0])
        
        if (Children < Adults) and (Children < MiddleAges) and (Children < Old) and (Children < Youth):
            label = "0-14"
        elif (Youth > Children) and (Youth < Adults) and (Youth < MiddleAges) and (Youth < Old):
            label = "15-24"
        elif (Adults > Children) and (Adults > Youth) and (Adults < MiddleAges) and (Adults < Old):
            label = "25-39"
        elif (MiddleAges > Children) and (MiddleAges > Adults) and (MiddleAges > Youth) and (MiddleAges < Old):
            label = "40-59"
        else:
            label="60+"
        
        color = (255, 0, 0)
        label = "Age Group:{}".format(label)#, max(Adults, Children, MiddleAges, Old, Youth) * 100)
        
        st.write(label)
        
        img = orig
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (5,5), 0)
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        
        sobely = cv2.Sobel(img_g, cv2.CV_8UC1, 0, 1, ksize=3)
        #plt.imshow(sobely,  cmap='gray')
        #plt.show()
        #cv2.sumElems(sobely)[0]
        hy, wy = sobely.shape
        param_1 = cv2.sumElems(sobely)[0]/(hy * wy)
        
        sobelx = cv2.Sobel(img_g, cv2.CV_8UC1, 1, 0, ksize=3)
        
        #cv2.sumElems(sobelx)[0]
        hx, wx = sobelx.shape
        param_2 = cv2.sumElems(sobelx)[0]/(hx * wx)
        
        Param_horizontal = (param_1 - param_fix_1)
        Param_vertical = (param_2 - param_fix_2)
        #print(Param_horizontal, Param_vertical)
        #print(param_1, param_fix_1, param_2, param_fix_2)
        
        if label == "0-14":
            st.write("No Wrinkles")
            
        else:
            if (Param_horizontal) < 0 and (Param_vertical) < 0:
                st.write("No Wrinkles")
            else:
                if Param_horizontal > Param_vertical:
                    st.write("Horizontal", (param_1 - param_fix_1)*2, "%")
                else:
                    st.write("Vertical", (param_2 - param_fix_2)*2, "%")
        
    #Show Face of Input Image
    except:
        st.warning("Please, upload another file!")
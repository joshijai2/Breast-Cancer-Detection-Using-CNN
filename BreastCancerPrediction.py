print("Please wait while we import some important libraries and models to run the software.")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tkinter import Tk

from tkinter.filedialog import askopenfilename

from time import sleep

import pandas as pd

import numpy as np

import cv2

from keras.models import load_model

annmodel = load_model(r'C:\git\Breast-Cancer-Detection-Using-CNN\results\ANN.h5')

base_cnnmodel = load_model(r'C:\git\Breast-Cancer-Detection-Using-CNN\results\base_CNN.h5')

final_cnnmodel = load_model(r'C:\git\Breast-Cancer-Detection-Using-CNN\results\final_CNN.h5')

print("The libraries and models are imported successfully.")

os.system('cls')
n=150
print("*"*n)
print("|"," "*(n-4),"|")
print("|"," "*(n-4),"|")
print("|"," "*(n//4),"Welcome to the Breast Cancer Detection Software", " "*(n//3+10),"|")
print("|"," "*(n//4+5),"made by 18BIT0223, 18BIT0239 and 18BIT0292"," "*(n//3+21),"|")
print("|"," "*(n-4),"|")
print("|"," "*(n-4),"|")
print("*"*n)

status = 1
while(status!=2):    
    print("Please select an image to detect:\n")
    sleep(1)
    
    Tk().withdraw()
    img_path = askopenfilename()
    sleep(0.5)
    
    #img_name = input("Enter image name: ")
    #root_path = "C:\\git\\Breast-Cancer-Detection-Using-CNN\\results"
    #img_path = root_path + img_name
    
    img = cv2.imread(img_path)
    print("The image is loaded successfully.")
    sleep(2)
    print("\nPress")
    sleep(0.25)
    opt = input("1: View image.\n2: Detect Breast Cancer.\n")
    sleep(0.5)
    if opt == "1":
        print("\nDisplaying image.\n")
        sleep(1)
        cv2.imshow('Image Specimen', cv2.resize(img,(250,250)))
        sleep(0.5)
        print("Close image to proceed.\n\n")
        cv2.waitKey(0)
        
    
    sleep(1)
    print("\nProcessing...\n")
    sleep(1)
    img = cv2.resize(img, (50,50), interpolation=cv2.INTER_CUBIC)
    
    test_input = img/255.0
    test_input = np.array([test_input,])

    annpred = annmodel.predict(test_input).argmax()
    base_cnnpred = base_cnnmodel.predict(test_input).argmax()
    final_cnnpred = final_cnnmodel.predict(test_input).argmax()

    label = img_path[-5]
    if label not in ["0","1"]:
        label = "unknown"
    
    print("\nHere are our predictions:")
    sleep(1)
    print('\nPredicted Value using ann model =',annpred)
    sleep(1)
    
    print('\nPredicted Value using base cnn model =',base_cnnpred)
    sleep(1)
    
    print('\nPredicted Value using final cnn model =',final_cnnpred)
    sleep(1)
    
    print("\nTrue Value =",label)
    sleep(1)
    
    if label == "unknown":
        print("True value is unknown.")
        result = "benign" if final_cnnpred == "0" else "malignant"
        print("\nPredicted value",final_cnnpred,"means the sample tested is",result)
    else:
        result = "benign" if label == "0" else "malignant"
        print("\n",label,"means the sample tested is",result)
    sleep(1)
    
    status = input("\nPress \n  1 : Select a new image. \n  2 : Exit.\n")
    while status not in ["1", "2"]:
        status = input("\nWrong input.\nPress \n  1 : Select a new image. \n  2 : Exit.\n")
    status = 2 if status!= "1" else 1
    print(status)
    os.system('cls')
            

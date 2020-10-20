print("Please wait while we import some important libraries and models to run the software.")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("*")
from tkinter import Tk
print("*")
from tkinter.filedialog import askopenfilename
print("*")
from time import sleep
print("*")
import pandas as pd
print("*")
import numpy as np
print("*")
import cv2
print("*")
from keras.models import load_model
print("*")
annmodel = load_model(r'C:\Users\joshi\OneDrive\B Tech\Sem 5\ITE2010-Artificial Intelligence\J-component\results\ANN.h5')
print("*")
base_cnnmodel = load_model(r'C:\Users\joshi\OneDrive\B Tech\Sem 5\ITE2010-Artificial Intelligence\J-component\results\base_CNN.h5')
print("*")
final_cnnmodel = load_model(r'C:\Users\joshi\OneDrive\B Tech\Sem 5\ITE2010-Artificial Intelligence\J-component\results\final_CNN.h5')
print("*")
print("The libraries and models are imported successfully.")
sleep(4)
os.system('cls')
n=100
print("*"*n)
print("|"," "*(n-4),"|")
print("|"," "*23,"Welcome to the Breast Cancer Detection Software", " "*24,"|")
print("|"," "*30,"made by Jai, Arishti and Tushar"," "*33,"|")
print("|"," "*(n-4),"|")
print("*"*n)
sleep(2)
input("\n\nPress Enter to start.\n")
    
status = 1
while(status!=2):
    os.system('cls')
    sleep(0.75)
    
    print("Please select an image to detect:\n")
    sleep(1)
    
    Tk().withdraw()
    img_path = askopenfilename()
    sleep(1)
    
    #img_name = input("Enter image name: ")
    #root_path = "C:\\Users\\joshi\\OneDrive\\B Tech\\Sem 5\\ITE2010-Artificial Intelligence\\J-component\\"
    #img_path = root_path + img_name
    
    img = cv2.imread(img_path)
    print("The image is loaded successfully.")
    sleep(2)
    print("\nPress")
    sleep(0.25)
    opt = input("1: View image.\n2: Detect Breast Cancer.\n3: Select a new image\n")
    sleep(0.5)
    if opt == "1":
        print("\nDisplaying image.\n")
        sleep(1)
        cv2.imshow('Image Specimen', cv2.resize(img,(250,250)))
        sleep(1.25)
        print("Close image to proceed.\n\n")
        cv2.waitKey(0)
    while(opt=="1"):
        print("\nPress")
        sleep(0.25)
        opt = input("1: View image again.\n2: Continue to detect Breast Cancer.\n3: Select a new image\n")
        
        sleep(0.5)
        if opt == "1":
            print("\nDisplaying image.\n")
            sleep(1)
            cv2.imshow('Image Specimen', cv2.resize(img,(250,250)))
            sleep(1)
            print("Close image to proceed.\n\n")
            cv2.waitKey(0)
        if opt not in ["1","2","3"]:
            opt="1"
        
        
    if opt != "3":
        sleep(2)    
        print("Processing...\n")
        sleep(4)
        img = cv2.resize(img, (50,50), interpolation=cv2.INTER_CUBIC)
        
        test_input = img/255.0
        test_input = np.array([test_input,])

        annpred = annmodel.predict(test_input).argmax()
        base_cnnpred = base_cnnmodel.predict(test_input).argmax()
        final_cnnpred = final_cnnmodel.predict(test_input).argmax()

        label = img_path[-5]
        if label not in ["0","1"]:
            label = "unknown"
        input("Press Enter to view Results")
        sleep(2)
        print("\nHere are our predictions:")
        sleep(2)
        print('\nPredicted Value using ann model =',annpred)
        sleep(2)
        
        print('\nPredicted Value using base cnn model =',base_cnnpred)
        sleep(2)
        
        print('\nPredicted Value using final cnn model =',final_cnnpred)
        sleep(2)
        
        print("\nTrue Value =",label)
        sleep(1)
        
        if label == "unknown":
            print("True value is unknown.")
            result = "benign" if final_cnnpred == "0" else "malignant"
            print("\nPredicted value",final_cnnpred,"means the sample tested is",result)
        else:
            result = "benign" if label == "0" else "malignant"
            print("\n",label,"means the sample tested is",result)
        sleep(5)
        
        status = input("\nPress \n  1 : Select a new image. \n  2 : Exit.\n")
        while status not in ["1", "2"]:
            status = input("\nWrong input.\nPress \n  1 : Select a new image. \n  2 : Exit.\n")
        status = 2 if status!= "1" else 1
        print(status)
            

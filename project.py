'''import pyautogui
import webbrowser as wb
import time
wb.open('web.whatsapp.com')
time.sleep(10)
for i in range(100):
    pyautogui.write('subscribe my channel i will give you diamonad and dj alok')
    pyautogui.press('enter')'''



'''import instaloader
ig=instaloader.Instaloader()
dp=input("enter the username: ")
ig.download_profile(dp,profile_pic_only=True)'''


'''from email.mime import audio
import imp
import cv2
import numpy as np
from sqlalchemy import false
img=cv2.imread("")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray=cv2.medianBlur(gray,5)
edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
color=cv2.bilateralFilter(img,9,250,250)
cartoon=cv2.bitwise_and(color,color,color,mask=edges)
cv2.imshow("image",img)
cv2.imshow("edges",edges)
cv2.imshow("cartoon",cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


from gtts import gTTS
from playsound import playsound
audio='speech.mp3'
language='en'
sp=gTTS(text="ayush bahut bada gaandu hai ye roj kisi na kisi se gand marwata rahta hai isko  bina gand maraye chain na aata ",lang=language,slow=False)
sp.save(audio)
playsound(audio)


'''from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
board=np.tile([1,0],[8,4])
for i in range(board.shape[0]):
    board[i]=np.roll(board[i],i%2)
cmap=ListedColormap(['#779556','#ebecd0'])
plt.matshow(board,cmap=cmap, )
plt.xticks([])
plt.yticks([])
plt.show()


from turtle import width
import requests
import cv2
import numpy as np
import imutils
url='http://100.75.228.127:8080/shot.jpg'
while True:
    img_resp=requests.get(url)
    img_ar=np.array(bytearray(img_resp.content),dtype=np.uint8)
    img=cv2.imdecode(img_ar,-1)
    img=imutils.resize(img,width=1000,height=1800)
    cv2.imshow('android_cam',img)
    if cv2.waitkey(1)==27:
        break
cv2.destroyAllWindows()'''























'''import numpy as np
import argparseclear
import cv2
import os
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input", type=str, required=True, help="path to input video")
ap.add_argument("-o","--output", type=str, required=True, help="path to output directory of cropped faces")
ap.add_argument("-d","--detector", type=str, required=True, help="path to opencv's deep learning face detector")
ap.add_argument("-c","--confidence", type=float, default=0.5, help="minimum probability to filter weak detection")
ap.add_argument("-s","--skip", type=int, default=16, help="# of frames to skip before applying face detaction")
args=vars(ap.parse_args())
print("[info] loading face detactor...")
protopath=os.path.sep.join([args["detector"], deploy.prototxt])
modelpath=os.path.join([args["detector"],"res10_300*300_ssd_iter_140000.caffemodel"])
net=cv2.dnn.readNetFromCaffe(protopath,modelpath)
vs=cv2.VideoCapture(args["input"])
read=0
saved=0
while True:
    (grabbed,Frame)=vs.read()
    if not grabbed:
        break
    read+=1
    if read%args["skip"]!=0:
        continue
    (h,w)=Frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(Frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detection=net.forward()
    if len(detection)>0:
        i=np.argmax(detection[0,0,:,2])
        confidence=detection[0,0,i,2]'''
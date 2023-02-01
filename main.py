import streamlit as st
import spotipy
import sys 
from spotipy.oauth2 import SpotifyClientCredentials
import cv2
import numpy as np
import os 


path = 'BaseImages'

#free to use 
orb = cv2.ORB_create(nfeatures=1000)

imagesls = []
classNames = []
mylist = os.listdir(path)

print(len(mylist))

for cl in mylist:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    imagesls.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])


def finDes(images):
    deslist = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist

def findID(img, desList, thres=15):
    kp1, ds1 = orb.detectAndCompute(img, None)
    #match between descriptors to find matches with K-Nearest neighbors
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,ds1, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75* n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    if len(matchList) !=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal




mydeslist = finDes(imagesls)

img1 = st.camera_input("Take a picture of your album!")
if img1:
    st.title('This is your image!')
    st.image(img1)

# img1 = cv2.imread('testIMages/Lionel.jpeg')
st.text('If you like the picture you took press the button below to find your album')
if st.button('Find album Match!'):
    bytes_data = img1.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    id = findID(cv2_img, mydeslist)

    if id != -1:
        st.balloons()
        st.text(f'this album is {classNames[id]}')
        print(f'this album is {classNames[id]}')
        # spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
        # spotify.search(q= str(classNames[id]), type = 'album')
    else:
        st.error('sorry I dont know that album yet')
        print('sorry I dont know that album yet')







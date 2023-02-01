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

img1 = cv2.imread('testIMages/Lionel.jpeg')


id = findID(img1, mydeslist)
print(classNames[id])

if id != -1:
    print(f'this album is {classNames[id]}')
else:
    print('sorry I dont know that album yet')








# img2 = cv2.imread('BaseImages/Diana Ross Greatest Hits.jpeg')
# kp,des = orb.detectAndCompute(img1,None)
# kp2,des2 = orb.detectAndCompute(img2,None)

# bf2 = cv2.BFMatcher()
# matches = bf2.knnMatch(des,des2,k=2)
# goodgood = []
# for m,n in matches:
#     if m.distance < 0.75* n.distance:
#         goodgood.append([m])
# img3 = cv2.drawMatchesKnn(img1,kp,img2,kp2,goodgood,None,flags=2)
# print(len(goodgood))
# cv2.imshow('image3', img3)
# # cv2.imshow('diana2', cv2.drawKeypoints(img2,kp2,None))
# cv2.waitKey(0)
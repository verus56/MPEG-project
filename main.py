import cv2
import numpy as np
from math import inf

img1 = cv2.imread('d2.png')
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:,:,0]
print(img1.shape)
img2 = cv2.imread('fi.png')
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:,:,0]

imageN = np.zeros((1080,1920)).astype(np.uint8)
imagepad = np.zeros((1080+128,1920+128)).astype(np.uint8)
imagepad[64:1144,64:1984]= grayImg2
cv2.imshow('imagepad', imagepad)
cv2.waitKey(0)

def Mse(block1, block2):
   err = np.sum((block1- block2)**2)
   err = err / (block1.shape[0]*block1.shape[1])
   return err

def search(step, b1cor, bloc1, i,j):
    newblock = b1cor
    while step>=1:
        #print("new block"+str(newblock))
        cord =[]
        cord.append([newblock[0]-step,newblock[1]-step,newblock[2]-step,newblock[3]-step])
        cord.append([newblock[0]-step,newblock[1]-step,newblock[2],newblock[3]])
        cord.append([newblock[0]-step,newblock[1]-step,newblock[2]+step,newblock[3]+step])
        cord.append([newblock[0],newblock[1],newblock[2]-step,newblock[3]-step])
        cord.append([newblock[0],newblock[1],newblock[2],newblock[3]])
        cord.append([newblock[0],newblock[1],newblock[2]+step,newblock[3]+step])
        cord.append([newblock[0]+step,newblock[1]+step,newblock[2]-step,newblock[3]-step])
        cord.append([newblock[0]+step,newblock[1]+step,newblock[2],newblock[3]])
        cord.append([newblock[0]+step,newblock[1]+step,newblock[2]+step,newblock[3]+step])
        mini = inf
        for k in cord:
            # print(k)
            # continue
            voisin = imagepad[k[0]:k[1],k[2]:k[3]]
            loss = Mse(voisin, bloc1)
            if loss < mini: 
                mini = loss
                newblock = k
        step= step//2
    if mini > 50: 
        imageN[i:i+16,j:j+16] = imagepad[newblock[0]:newblock[1], newblock[2]: newblock[3]]



for i in range(0,grayImg1.shape[0]-16,16):
    for j in range(0,grayImg1.shape[1]-16,16):
        #image1 blocks 
        bloc1 = grayImg1[i:(i+16), (j):(j+16)]
        b1cor = [i+64,(i+16)+64,j+64,(j+16)+64]
        step = 64
        search(step, b1cor, bloc1, i,j)
cv2.imwrite('image072_residue.png', imageN)
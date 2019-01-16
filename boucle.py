# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 18:37:39 2015

@author: Patricia
"""
#######To clear the working memory###########
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
#############################################
#clearall()

import numpy as np
from skimage.color import rgb2gray
from skimage import data, measure,io
import skimage as sk
import matplotlib.pyplot as plt
from dct2 import dct2, idct2

#############################################

def decoupage_en_blocks(img, n=8):
    img_size = img.shape
    xMax = img_size[0]
    yMax = img_size[1]
    res = []

    for x in range(0, xMax, n):
        resbis = []
        for y in range(0, yMax, n):

            block = np.zeros((n, n))
            block = img[x:x+n,y:y+n]

            resbis.append(block)
        res.append(resbis)

    return res

def recollage_des_blocks(tab, n=8):
    xMax = len(tab)
    yMax = len(tab[0])

    img = np.zeros((xMax*n, yMax*n))

    for x in range(0, xMax):
        for y in range(0, yMax):

            for i in range(0, n):
                for j in range(0,n):

                    img[x*n+i,y*n+j] = tab[x][y][i][j]

    return img

#############################################

def dct_blocks(tab, n=8):
    res = []

    for x in tab:
        resbis = []
        for block in x:

            new_block = np.zeros((n, n))
            new_block = dct2(block)

            resbis.append(new_block)
        res.append(resbis)

    return res

def idct_blocks(tab, n=8):
    res = []

    for x in tab:
        resbis = []
        for block in x:

            new_block = np.zeros((n, n))
            new_block = idct2(block)

            resbis.append(new_block)
        res.append(resbis)

    return res

#############################################

def creer_tab_quantif(compression, n=8):
    res = np.zeros((n, n))

    for i in range(0,n):
        for j in range(0,n):
            res[i][j] = 1 + (1 + i + j)*compression

    return res

def quantification(tab, compression, n=8):
    res = []
    tab_quantif = creer_tab_quantif(compression,n)

    for x in tab:
        resbis = []
        for block in x:

            new_block = np.zeros((n, n))
            new_block = np.round(block/tab_quantif)

            resbis.append(new_block)
        res.append(resbis)

    return res

def dequantification(tab, compression, n=8):
    res = []
    tab_quantif = creer_tab_quantif(compression,n)

    for x in tab:
        resbis = []
        for block in x:

            new_block = np.zeros((n, n))
            new_block = block*tab_quantif

            resbis.append(new_block)
        res.append(resbis)

    return res

#############################################

plt.close('all')
n=8
c=50


#Recuperation et configuration de l'image initiale
img0=io.imread('sequence video/taxi_00.bmp')
img0_gray = rgb2gray(img0)
img0_gray=sk.img_as_float(img0_gray)*255
plt.figure(0)
plt.title('Image initiale')
plt.imshow(img0_gray, cmap='gray')


#Recuperation et configuration de l'image initiale + 1
img1=io.imread('sequence video/taxi_01.bmp')
img1_gray = rgb2gray(img1)
img1_gray=sk.img_as_float(img1_gray)*255
plt.figure(1)
plt.title('Image initiale+1')
plt.imshow(img1_gray, cmap='gray')

img_size=img0_gray.shape

##############################################

def diff(block1, block2):

    res = 0
    
    for i in range(0, block1.shape[0]):
        for j in range(0, block1.shape[1]):

            res += np.abs(block1[i][j] - block2[i][j])
    return res

def calculDiff(block1,block2):
    
    res = np.zeros((block1.shape[0], block1.shape[1]))
    for i in range(0, block1.shape[0]):
        for j in range(0, block1.shape[1]):

            res[i][j] = np.abs(block1[i][j] - block2[i][j])
            
    return res


def find_block(img, block1, x, y, maxD):

    sizeblock = block1.shape
    sizeimg = img.shape

    min = 256*16*16
    minI = -1
    minJ = -1

    for i in range(x-maxD, x+maxD, 1):
        for j in range(y-maxD, y+maxD, 1):
            

            if i >= 0 and j >= 0 and i + sizeblock[0]-1 <= sizeimg[0] - 1 and j + sizeblock[1]-1 <= sizeimg[1] - 1:

                block2 = img[i:i+sizeblock[0], j:j+sizeblock[1]]
                tmp = diff(block1, block2)

                if tmp < min :
                    min = tmp
                    minI = i
                    minJ = j
    return {
        'block': img[minI:minI+sizeblock[0], minJ:minJ+sizeblock[1]],
        'i': minI,
        'j': minJ
    }


taille_block= 16
maxD=15


imPred = np.zeros((img_size[0], img_size[1]))

sizex = (img_size[0]//16) * 16
sizey = (img_size[1]//16) * 16
#imErreur = np.zeros((img_size[0], img_size[1]))
imErreur = np.zeros((sizex,sizey))

for x in range(0, img_size[0]-taille_block+1, taille_block):
    for y in range(0, img_size[1]-taille_block+1, taille_block):

        block = img1_gray[x:x+taille_block, y:y+taille_block]
        res = find_block(img0_gray, block, x, y, maxD)
        print(res['i'], res['j'])
        imPred[x:x+taille_block, y:y+taille_block] = res['block'] 
        imErreur[x:x+taille_block, y:y+taille_block] = calculDiff(block, res['block'])

plt.figure(6)
plt.title('Image Predite')
plt.imshow(imPred, cmap='gray')




plt.figure(7)
plt.imshow(imErreur, cmap='gray')

## A partir d'ici, on a l'image prédite et l'image d'erreur, on fait dct, quantification et inverse sur l'image d'erreur
n = 8
blocks = decoupage_en_blocks(imErreur, n)
blocks = dct_blocks(blocks, n)
blocks = quantification(blocks, c, n)
blocks = dequantification(blocks, c, n)
blocks = idct_blocks(blocks, n)
new_img = recollage_des_blocks(blocks, n)

#On ajoute l'image d'erreur à l'image prédite
imRecompose = np.zeros((new_img.shape[0], new_img.shape[1]))
for x in range(0,new_img.shape[0],1):
    for y in range(0,new_img.shape[1],1):
        imRecompose[x,y] = new_img[x,y] + imPred[x,y]
        
plt.figure(8)
plt.title('Image recomposee')
plt.imshow(imRecompose, cmap='gray')

#On refait pour i2

#Recuperation et configuration de l'image initiale
img0_gray = rgb2gray(imRecompose)
img0_gray=sk.img_as_float(img0_gray)*255
plt.figure(9)
plt.title('Image initiale (image recomposee)')
plt.imshow(img0_gray, cmap='gray')

#Recuperation et configuration de l'image initiale + 2
img1=io.imread('sequence video/taxi_02.bmp')
img1_gray = rgb2gray(img1)
img1_gray=sk.img_as_float(img1_gray)*255
plt.figure(10)
plt.title('Image initiale+2')
plt.imshow(img1_gray, cmap='gray')

img_size=img0_gray.shape

taille_block= 16
maxD=15


imPred = np.zeros((sizex, sizey))
imErreur = np.zeros((sizex,sizey))

for x in range(0, img_size[0]-taille_block+1, taille_block):
    for y in range(0, img_size[1]-taille_block+1, taille_block):

        block = img1_gray[x:x+taille_block, y:y+taille_block]
        res = find_block(imRecompose, block, x, y, maxD)
        print(res['i'], res['j'])
        imPred[x:x+taille_block, y:y+taille_block] = res['block'] 
        imErreur[x:x+taille_block, y:y+taille_block] = calculDiff(block, res['block'])
       
        
plt.figure(11)
plt.title('Image Predite(2)')
plt.imshow(imPred, cmap='gray')




plt.figure(12)
plt.title('Image Erreur (2)')
plt.imshow(imErreur, cmap='gray')

## A partir d'ici, on a l'image prédite et l'image d'erreur, on fait dct, quantification et inverse sur l'image d'erreur
n = 8
blocks = decoupage_en_blocks(imErreur, n)
blocks = dct_blocks(blocks, n)
blocks = quantification(blocks, c, n)
blocks = dequantification(blocks, c, n)
blocks = idct_blocks(blocks, n)
new_img = recollage_des_blocks(blocks, n)

#On ajoute l'image d'erreur à l'image prédite
imRecompose = np.zeros((new_img.shape[0], new_img.shape[1]))
for x in range(0,new_img.shape[0],1):
    for y in range(0,new_img.shape[1],1):
        imRecompose[x,y] = new_img[x,y] + imPred[x,y]
        
plt.figure(13)
plt.title('Image recomposee (2)')
plt.imshow(imRecompose, cmap='gray')
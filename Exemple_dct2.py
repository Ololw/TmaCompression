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
    
import os        
import numpy as np
from skimage.color import rgb2gray
from skimage import data, measure,io
import skimage as sk
import matplotlib.pyplot as plt
from dct2 import dct2, idct2
import PIL as pil #pour utiliser la librairie d'écriture de fichier jpeg

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


plt.close('all')
n=8
c=50
nb_image = 1


#Recuperation et configuration de l'image initiale
img=io.imread('horse.bmp')
img_gray = rgb2gray(img)
img_gray=sk.img_as_float(img_gray)*255
plt.figure(nb_image)
plt.title('Image initiale')
plt.imshow(img_gray, cmap='gray')
nb_image += 1


img_size=img_gray.shape

##############################################

#Decoupage en blocs, dct et quantification
blocks = decoupage_en_blocks(img_gray, n)
blocks = dct_blocks(blocks, n)
blocks = quantification(blocks, c, n)


#Partie pour afficher l'image après avoir effectué une déquantification
blocks = dequantification(blocks, c, n)
blocks = idct_blocks(blocks, n)
new_img = recollage_des_blocks(blocks, n)
plt.figure(nb_image)
plt.title('Image finale')
plt.imshow(new_img, cmap='gray')
nb_image+=1

##############################################

#Pil pour différentes qualités
monIm = pil.Image.fromarray(np.ubyte(np.round(img_gray,0)))
try:
    os.mkdir('Pil')
except:
    pass    
    
psnr = []
ssim = []
tabCompression = []

for nb_quality in range(1,101,1):
    monIm.save('Pil/horse_pil_qualite{}.jpeg'.format(nb_quality), quality = nb_quality)
    monImlu = io.imread("Pil/horse_pil_qualite{}.jpeg".format(nb_quality))
    #Calcul du psnr
    psnrCalc = measure.compare_psnr(img_gray, monImlu,1.0)
    psnr.append(psnrCalc)
    #Calcul du ssim
    ssimCalc = measure.compare_ssim(img_gray, monImlu)
    ssim.append(ssimCalc)
    #Calcul compression
    compress = 1.0*img_size[0]*img_size[1]/os.path.getsize("Pil/horse_pil_qualite{}.jpeg".format(nb_quality))
    tabCompression.append(compress)
    
    if(nb_quality == 1 or nb_quality == 50 or nb_quality == 100):
        plt.figure(nb_image)
        plt.title('Cheval avec la qualite {}'.format(nb_quality))
        plt.imshow(monImlu)
        nb_image += 1
    
    
    
    
print('100 Images de cheval avec des qualités différentes ont été mises dans le dossier Pil')

plt.figure(nb_image)
plt.title('Psnr en fonction de la qualité')
plt.plot(range(1,101),psnr)  
nb_image+= 1

plt.figure(nb_image)
plt.title('Ssim en fonction de la qualité')
plt.plot(range(1,101), ssim)
nb_image +=1

plt.figure(nb_image)
plt.title('Psnr en fonction de la compression')
plt.plot(tabCompression, psnr)
nb_image +=1

plt.figure(nb_image)
plt.title('Ssim en fonction de la compression')
plt.plot(tabCompression, ssim)
nb_image +=1





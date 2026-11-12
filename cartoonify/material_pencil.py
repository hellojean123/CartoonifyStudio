"""Persistent tonal valleys for wrinkles/folds, not semantic segmentation."""
import cv2
import numpy as np


def material_structure(rgb,skin):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    responses=[]
    for sigma in (1.2,2.4,4.8):
        blur=cv2.GaussianBlur(gray,(0,0),sigma)
        xx=cv2.Sobel(blur,cv2.CV_32F,2,0,ksize=3)/4
        yy=cv2.Sobel(blur,cv2.CV_32F,0,2,ksize=3)/4
        xy=cv2.Sobel(blur,cv2.CV_32F,1,1,ksize=3)/4
        root=np.sqrt((xx-yy)**2+4*xy*xy)
        normal=(xx+yy+root)/2;tangent=(xx+yy-root)/2
        elongated=np.clip((normal-np.abs(tangent))/(np.abs(normal)+1e-5),0,1)
        valley=np.maximum(normal-.0012,0)*sigma**1.3*elongated
        responses.append(valley.astype(np.float32))
    # Cross-scale persistence rejects isolated pores and single-pixel speckle.
    fine=np.sqrt(responses[0]*responses[1]);coarse=np.sqrt(responses[1]*responses[2])
    blend=cv2.GaussianBlur(skin.astype(np.float32),(0,0),4)
    relief=np.clip(fine*blend*20+coarse*(1-blend)*24,0,.65)
    # A soft tonal surface carries volume; detected grooves remain selective accents.
    form=cv2.bilateralFilter(np.round(gray*255).astype(np.uint8),9,24,7).astype(np.float32)/255
    return relief,form

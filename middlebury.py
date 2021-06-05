## Middlebury color representation
##
## according to Matlab source code of Deqing Sun (dqsun@cs.brown.edu)
## Python translation by Dominique Bereziat (dominique.bereziat@lip6.fr)

import numpy as np
import os

def computeColor( W, norma=False):
    """ Compute a colormap according the Middlebury convention """
    def colorWheel():
        """ internal function """
        RY,YG,GC = 15,6,4
        CB,BM,MR = 11,13,6

        ncols = RY+YG+GC+CB+BM+MR
        colwheel = np.zeros((ncols,3),dtype=np.uint8)

        col = 0
        colwheel[:RY,0] = 255
        colwheel[:RY,1] = np.floor(255*np.arange(RY)/RY)
        col += RY

        colwheel[col:col+YG,0] = 255 - np.floor(255*np.arange(YG)/YG)
        colwheel[col:col+YG,1] = 255
        col += YG

        colwheel[col:col+GC,1] = 255
        colwheel[col:col+GC,2] = np.floor(255*np.arange(GC)/GC)
        col += GC

        colwheel[col:col+CB,1] = 255 - np.floor(255*np.arange(CB)/CB)
        colwheel[col:col+CB,2] = 255
        col += CB

        colwheel[col:col+BM,2] = 255
        colwheel[col:col+BM,0] = np.floor(255*np.arange(BM)/BM)
        col += BM

        colwheel[col:col+MR,2] = 255 - np.floor(255*np.arange(MR)/MR)
        colwheel[col:col+MR,0] = 255

        return colwheel

    
    U = W[:,:,0].copy()
    V = W[:,:,1].copy()

    nanIdx = np.isnan(U) | np.isnan(V)
    U[nanIdx] = 0
    V[nanIdx] = 0

    colwheel = colorWheel()
    ncols = colwheel.shape[0]

    rad = np.sqrt(U**2+V**2)
    if norma:
        U /= np.max(rad)
        V /= np.max(rad)
        rad = np.sqrt(U**2+V**2)
    
    a = np.arctan2(-V,-U)/np.pi

    fk = (a+1)/2 * (ncols-1) # -1~1 maped to 0~ncols-1
    k0 = np.uint8(np.floor(fk))
    k1 = (k0+1)%ncols
    f = fk - k0
    img = np.empty((U.shape[0],U.shape[1],3),dtype=np.uint8)

    for i in range(3):
        tmp = colwheel[:,i]
        col0 = tmp[k0]/255;
        col1 = tmp[k1]/255;
        col = (1-f)*col0 + f*col1
        idx = rad <=1
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] *= 0.75
        #img[:,:,i] = np.uint8(np.floor(255*col*(1-nanIdx)))
        img[:,:,2-i] = np.uint8(np.floor(255*col*(1-nanIdx)))
    return img




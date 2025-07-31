"""
Projet Segmentation fossile 
GILET Eliott 
main 
"""
## ----------------------------------------[ Zone d'import ]---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import Lib_outils as lo
import Lib_contours as lc
import Lib_rayon as lr

from PIL import Image

## ----------------------------------------[ Zone de déclaration ]---------------------------------------------------------------------------------------------------------------------
sigma = 1
r_voisin = 10
img_test = ['img_test.tif']
img_stack = ["CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c"]

## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    img_lvg,img_bin = lc.segment_1_pic_function(sigma,img_test)

    print(" segment : check ")

    print ( " Matrice niveau de gris : ")
    print(img_lvg.shape)
    print ( " ")
    print ( " Matrice contours :")
    print(img_bin.shape)
    print(" ")

    img_r = lr.approx_radius_map(img_bin,r_voisin)
    img_r_V2 = lr.approx_radius_map_V2(img_bin,r_voisin)
    print (" rayon : check")

    lo.affiche_rayon(img_r,"Rayons de courbures","magma")
    lr.show_histo(img_r)
    lo.affiche_rayon(img_r_V2,"Rayons de courbures V2","magma")
    lr.show_histo(img_r_V2)


    plt.show()
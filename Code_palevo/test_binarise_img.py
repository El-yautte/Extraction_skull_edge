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

from PIL import Image

## ----------------------------------------[ Zone de déclaration ]---------------------------------------------------------------------------------------------------------------------
sigma = 1
save = 0 

## Donnée de test 

img_test = ['img_test_2.tif']
img_stack = ["CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c"]

## Paramètre de traitement

v_param = [ 3.0 , 3.0 , 3.0 , 2.5 , 0.9 , 300 ]

## Choix de l'affichage 

af_s1 = [ 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ]
af_s4 = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
af_post = [ 0 , 0 , 0 , 1 , 1 ]
affichage = [ af_s1 , af_s4 , af_post ]



## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    img = lo.open_tif(img_test[0])

    img_lv = lo.smart_threshold_clean(img)

    Contours_extrait = lc.Full_extract(img_lv,v_param,affichage,save)


    plt.show()
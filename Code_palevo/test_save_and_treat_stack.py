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

## Donnée de test 

img_test = ['img_test_2.tif']
img_stack = ["CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c"]

## Paramètre de traitement

v_param = [ 1.0 , 3.0 , 1.0 , 2.5 , 0.9 , 300 ]

## Choix de l'affichage 

af_s1 = [ 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ]
af_s4 = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
af_post = [ 0 , 0 , 0 , 1 , 1 ]
affichage = [ af_s1 , af_s4 , af_post ]



## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Archivage des vieilles données 

    lo.archive_file_saved()

    # Extraction des contours 

    img_lvg_stack = lo.open_tif_stack(img_stack[0])

    Contours_stack = lc.Full_extract_stack(img_lvg_stack,v_param,affichage)

    lo.save_image_stack(Contours_stack,".tif")

    lo.afficher_pile(Contours_stack,"cool")

    plt.show()
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

v_param = [ 6.0 , 3.0 , 3.0 , 2.5 , 0.9 , 300 ]

## Choix de l'affichage 

af_s1 = [ 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ]
af_s4 = [ 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ]
af_post = [ 1 , 1 , 0 , 1 , 1 ]
affichage = [ af_s1 , af_s4 , af_post ]



## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Archivage des vieilles données 

    lo.archive_file_saved()

    save=lo.ask_save()

    # Extraction des contours 

    Contours_extrait = lc.Full_extract(img_test,v_param,affichage,save)

    lo.save_space(save)


    plt.show()



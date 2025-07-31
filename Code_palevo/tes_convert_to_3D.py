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
cont_stack = ["contour_stack/boite_&_dents_390_stack"]

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

    ## récupération des données 

    contour = lo.open_image_stack(cont_stack[0],".tif")

    ## alignment 

    contour_aligné, centre = lo.stack_alignement_by_lenth(contour)

    lo.save_image_stack(contour_aligné,".png","img_align_stack")

    # convertion en 3D 

    # cont_3D = lo.save_binary_stack_as_mesh_obj(contour,"contour_full_scan.obj","3D_file")

    ## affichage

    # lo.visualiser_fichier_3D("3D_file")

    plt.show()
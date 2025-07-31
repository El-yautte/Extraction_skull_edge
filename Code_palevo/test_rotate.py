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

## Donnée de test 

img_test = ['contours_test_2.png']
img_stack = ["CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c"]
cont_stack = ["contour_stack/boite_7"]

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

    ## image seul --------------------------------

    image =lo.open_image(img_test[0])

    img_align , center = lo.align_skull(image)

    lo.afficher_image(img_align)

    angle,dist_min,dist_max = lr.radial_profile(img_align, center)

    lo.plot_radial(angle,dist_min,dist_max,'linear')
    lo.plot_radial(angle,dist_min,dist_max)

    img_up , smoth = lr.approximate_radial_profile_poly(img_align,center, dist_max,angle)

    lo.afficher_image(img_up)
    lo.plot_radial(angle,dist_max,smoth,'linear','both','dist_max','dist_max_lisser')

    ## pile d'image---------------------------------

    # contours = lo.open_image_stack(cont_stack[0])

    # cont_aligne , center = lo.align_skull_stack(contours)

    # angle_stack,dist_min_stack,dist_max_stack = lr.radial_profile_stack(cont_aligne,center)

    # lr.plot_radial_stack(angle_stack,dist_min_stack,dist_max_stack,'min')
    # # lr.plot_radial_stack(angle_stack,dist_min_stack,dist_max_stack,'both')



    plt.show()






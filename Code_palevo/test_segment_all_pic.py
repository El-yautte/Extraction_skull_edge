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


img_test = ['img_test.tif']
img_stack = ["CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c"]

## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------





## ----------------------------------------[ Zone du main ]---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
     
    print (" Run : Check ")

    ## Recupération dimension image 
    img_lvg = lo.open_tif(img_test[0])
    nbr_lgn,nbr_col=img_lvg.shape

    img_lvg_stack = lo.open_tif_stack(img_stack[0])

    print(" open : check ")

    ## Création des filtres 

    f_v = lc.vertical_F(sigma,img_lvg_stack[:,:,0])
    f_h = lc.horizontal_F(sigma,img_lvg_stack[:,:,0])
    print(" Filtre : check ")

    ## derivé 
    img_dx_stack,img_dy_stack = lc.derive_stack(img_lvg_stack,f_v,f_h)
    print("derivé : check")

    ## Calcul de Acor

    Acor_stack = lc.Acor_matrice_stack(img_dx_stack,img_dy_stack)
    print(" Acor : check")

    ## Calcul de lambda

    Delta_stack,Lambda_stack = lc.Delta_Lambda_stack(Acor_stack)
    print(" Delta / Lambda : check ")

    ## Calcul de la norme 

    Chatoux_stack = lc.chatoux_stack(Lambda_stack)
    print(" Chatoux : check ")

    ## Binarisation 

    img_binair_stack = lc.binarise_stack(Chatoux_stack)

    ## affichage 

    lo.afficher_pile(img_lvg_stack)

    # lo.afficher_derive_stack(img_dx_stack,img_dy_stack,' Dx ',' Dy ')

    lo.afficher_pile(img_binair_stack)


    plt.show()
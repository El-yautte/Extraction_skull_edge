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

save_name = " Test_save "

'''
Paramètre d'affichage premier contours :

af_sX [0] : Raw data 
      [1] : Image en niveau de gris 
      [2] : Filtres horizontal et vertical 
      [3] : Dérivée en x et y 
      [4] : Matrice Acor ( couche α (XX) / couche β (XY) / couche γ (YY))
      [5] : Matrice Delta et Lambda 
      [6] : Norme de Chatoux 
      [7] : Contours extrait binaire 
'''
af = [ 0 , 1 , 0 , 0 , 1 , 0 , 1 , 1 ]
data = []

## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    lo.archive_file_saved()

 ## Ouverture de l'image 

    img_lvg = lo.open_tif(img_test[0])
    nbr_lgn,nbr_col=img_lvg.shape
    if af[0] == 1 or af[1] == 1 :
        data.append(img_lvg)
    print(" Open : check ")

    ## Création des filtres 
    f_v , f_h = lc.vertical_and_horizontal_f(sigma,img_lvg)
    if af[2] == 1 :
        data.append(f_v)
        data.append(f_h)
    print(" Filtre : check ")

    ## Calcul de la dérivée 

    img_dx,img_dy = lc.all_derive(img_lvg,f_v,f_h)
    if af[3] == 1 :
        data.append(img_dx)
        data.append(img_dy)
    print(" Derive : check")

    ## Matrice Acor
    
    Acor = lc.Acor_Matrice(img_dx,img_dy,nbr_lgn,nbr_col)
    if af[4] == 1 :
        data.append(Acor)
    print(" Acor :check ")

    ## Matrice lambda et delta 

    Delta,Lambda = lc.Lambda_matrice(Acor,nbr_lgn,nbr_col)
    if af[5] == 1 :
        data.append(Delta)
        data.append(Lambda)
    print("lambda / Delta : check")

    ## Norme de chatoux

    n_chatoux = lc.Chatoux(Lambda,nbr_lgn,nbr_col)
    if af[6] == 1 :
        data.append(n_chatoux)
    
    print("norme : check")

    ## Binarisation 

    n_bin = lc.binarise( n_chatoux,nbr_lgn,nbr_col)
    if af[7] == 1 :
        data.append(n_bin)
    print(" binair : check ")

    # lo.afficher_image(img_lvg,"raw","gray",save_name)
    # lo.afficher_image(img_lvg,"pic","gray",save_name)

    # lo.affiche_filtres(f_v,f_h,"Filtre H",'Filtre V','hot',save_name)

    # lo.affiche_filtres(img_dx,img_dy,'Dx','Dy','inferno',save_name)

    # lo.affiche_acor(Acor,'Matrice Acor','inferno',save_name)

    # lo.affiche_delta_lambda(Delta , Lambda ,'Matrice delta et lambda','plasma','magma',save_name)
    
    # lo.affiche_chatoux(n_chatoux,"Norme ( Chatoux ) ", "coolwarm",save_name)

    lo.affiche_segment(af,data,"save_TEST")




    plt.show()
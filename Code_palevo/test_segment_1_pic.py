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


img_test = ['img_test_2.tif']

af_s4 = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]


## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------





## ----------------------------------------[ Zone du main ]---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
     
    print (" Run : Check ")

    ## Ouverture de l'image 

    img_lvg = lo.open_tif(img_test[0])
    nbr_lgn,nbr_col=img_lvg.shape
    print(" Open : check ")

    ## Création des filtres 
    f_v , f_h = lc.vertical_and_horizontal_f(sigma,img_lvg)

    f_v_a = np.fft.fftshift(f_v)
    f_h_a = np.fft.fftshift(f_h)
    print(" Filtre : check ")

    ## Calcul de la dérivée 

    img_dx,img_dy = lc.all_derive(img_lvg,f_v,f_h)
    print(" Derive : check")

    ## Matrice Acor
    
    Acor = lc.Acor_Matrice(img_dx,img_dy,nbr_lgn,nbr_col)
    print(" Acor :check ")

    ## Matrice lambda et delta 

    Delta,Lambda = lc.Lambda_matrice(Acor,nbr_lgn,nbr_col)
    print("lambda / Delta : check")

    ## Norme de chatoux

    n_chatoux = lc.Chatoux(Lambda,nbr_lgn,nbr_col)
    print("norme : check")

    ## Binarisation 

    n_bin = lc.binarise( n_chatoux,nbr_lgn,nbr_col)
    print(" binair : check ")

    ## norme 

    # norm,arg_nabs = lc.grad_norme_arg(img_dx,img_dy)
    # arg = np.abs(arg_nabs)

    # ## angle 

    # img_theta = lc.Angle_theta(Lambda,Acor,nbr_lgn,nbr_col)
    # img_theta_abs = np.abs(img_theta)

    # ## sigma = 4 

    # data = lc.base_segmente(0,4 , img_test,af_s4)

    # img_conf = lc.conf_contours(n_bin,data[9],norm,arg)
    # img_conf_fort =lc.binarize_confidence(img_conf, 0.1)


    ## affichage 

    # lo.afficher_image(img_lvg,"raw")
    # lo.afficher_image(img_lvg,"pic")

    lo.affiche_filtres(f_v_a,f_h_a,'Filtre H','Filtre V','hot')

    # lo.affiche_filtres(img_dx,img_dy,'Dx','Dy','inferno')

    # lo.affiche_acor(Acor,'Matrice Acor','inferno')

    # lo.affiche_delta_lambda(Delta , Lambda ,'Matrice delta et lambda','plasma','magma')
    
    # lo.affiche_chatoux(n_chatoux, titre="Norme ( Chatoux ) ", cmap="coolwarm")

    # lo.affiche_chatoux(n_bin, titre="Contours ", cmap="Purples")

    # lo.affiche_chatoux(norm, titre="normes du gradient ", cmap="viridis")
    # lo.affiche_chatoux(arg_nabs, titre="argument du gradient ", cmap="viridis")
    # lo.affiche_chatoux(arg, titre="argument du gradient ", cmap="viridis")
    # lo.affiche_rayon(img_conf, "Contours ", "cool")
    # lo.affiche_rayon(img_conf_fort, "Contours confiances ", "cool")


    




    plt.show()

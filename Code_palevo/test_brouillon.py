"""
Projet Segmentation fossile 
GILET Eliott 
main 
"""
## ----------------------------------------[ Zone d'import ]---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import math

import Lib_outils as lo
import Lib_contours as lc
import Lib_rayon as lr

from PIL import Image

## ----------------------------------------[ Zone de déclaration ]---------------------------------------------------------------------------------------------------------------------
sigma = 4

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
def gaussian_filters_1D(sigma):
    """
    Retourne les filtres 1D : gaussien (LP) et dérivée de gaussienne (HP)
    """
    size = int(2 * math.ceil(3 * sigma) + 1)
    demi_size = int(math.ceil(3.0 * sigma))

    LP_filter = np.zeros((size, 1), dtype=np.float32)
    HP_filter = np.zeros((size, 1), dtype=np.float32)

    t = np.arange(-demi_size, demi_size + 1)

    LP_filter[:, 0] = np.exp(-t**2 / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))
    HP_filter[:, 0] = (-t * np.exp(-t**2 / (2.0 * sigma**2))) / (sigma**3 * np.sqrt(2.0 * np.pi))

    return t, LP_filter[:, 0], HP_filter[:, 0]

if __name__ == '__main__':
    sigmas = [1.0, 4.0]  # Valeurs de sigma à comparer

    plt.figure(figsize=(12, 5))

    # Graphique du filtre passe-bas (Gaussien)
    plt.subplot(1, 2, 1)
    for sigma in sigmas:
        t, LP, _ = gaussian_filters_1D(sigma)
        plt.plot(t, LP, label=f'σ = {sigma}')
    plt.title('Filtre passe-bas (Gaussien)')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Graphique du filtre passe-haut (Dérivée de Gaussienne)
    plt.subplot(1, 2, 2)
    for sigma in sigmas:
        t, _, HP = gaussian_filters_1D(sigma)
        plt.plot(t, HP, label=f'σ = {sigma}')
    plt.title('Filtre passe-haut (dérivée de Gaussienne)')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
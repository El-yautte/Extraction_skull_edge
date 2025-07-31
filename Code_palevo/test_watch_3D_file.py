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



## ----------------------------------------[ Zone d'initialisation ]---------------------------------------------------------------------------------------------------------------------





## ----------------------------------------[ Zone du main ]---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
     
    print (" Run : Check ")

    lo.visualiser_fichier_3D("3D_file")
    # lo.aligner_fichiers_3D("3D_file")
     
    # lo.afficher_volume_tif("CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c")
    # lo.visualiser_volume_binaire("CT_Scan/Clarisa Sutherland_XCT_S3428_ 86_01_c")

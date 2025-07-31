'''
Librairie CONTOURS  

2025, GILET Eliott , PALEVOPRIM/X-LIM

    Résumé : Cette librairie contint les fonctions 
et sous-fonctions nécéssaires a l'extraction des
contours d'une images.

## -------[ Structure de la librairie ]------- ##

## [ Fonctions globale ] ##
#  [ Segmentation complète ]
#  [ Etapes de segmentations ]

## [ Dérivée ] ##
#  [ créations des filtres ]
#  [ Dérivation ]

## [ Norme de chatoux ] ##
#  [ Matrice Acor ]
#  [ Matrice des Lambda et de Delta ]
#  [ Calcul de Chatoux ]

## [ Etude du gradient ] ##
#  [ Norme ]
#  [ Angle ]

## [ Calcul de confiance des contours ] ##
#  [ Globale  ]
#  [ Calcul de la confiance ]
#  [ Impact de l'angle ]
#  [ Corrections ]

## [ Seuillage ] ##
#  [ Norme ]
#  [ Confiance ]

'''
#################################################################################################################################################################################
## ----------------------------------------[ Zone d'import ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################
import os
import pyvista as pv
import math
import matplotlib.pyplot as plt
import cv2
import Lib_outils as lo
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio 
import math

from matplotlib.widgets import Slider
from skimage.io import imread
from skimage.morphology import remove_small_objects,skeletonize,disk
from skimage.draw import line as skimage_line
from scipy.ndimage import binary_dilation,generic_filter,map_coordinates,maximum_filter,convolve,label


#################################################################################################################################################################################
## ----------------------------------------[ Zone de déclaration ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################

## Paramètres d'affichages
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
af_s1 = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
af_s4 = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]

'''
Paramètre d'affichage post traitement :

af_sX [0] : Norme du gradient ( sigma = 1 )
      [1] : Argument / Angle du gradient ( sigma = 1 )
      [2] : Valeurs absolue de l'argument 
      [3] : Confiance des contours 
      [4] : Contours extrait
'''
af_post = [ 0 , 0 , 0 , 0 , 0 ]

affichage = [ af_s1 , af_s4 , af_post ]

## Parametrage 
'''
Paramètre de post_traitements des contours :

v_param [0] : poid du contours pour sigma fort (4)
        [1] : poid de la norme du gradient 
        [2] : poid de l'angle du gradient 
        [3] : Coefficient de boost de confiance 
        [4] : Coefficient de réduction de confiance 
        [5] : Distance min pour considérer un éléments continue ( nombre de pixel )

'''
v_param = [ 1.0 , 3.0 , 1.0 , 2.5 , 0.9 , 300 ]


#################################################################################################################################################################################
## ----------------------------------------[ Zone de fonction ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################

## ====================================================[ Fonctions globale ]==================================================== ##

## -------[ Segmentation complète ]------- ##

def Full_extract ( img_test,parameter,af = affichage ,save=0,orientation='vertical'):              ## extraction complete des contours 

    af_1 = af[0]
    af_4 = af[1]
    af_p = af[2]  ## Séparation des paramètres d'affichage 

    if save != 0 :
        name_1 = save + " : Base_segment_S1"
        name_2 = save + " : Base_segment_S4"
        name_3 = save + " : Final_segment"
    else :
        name_1 = 0
        name_2 = 0
        name_3 = 0

    if isinstance(img_test, list):
        img = lo.open_image(img_test[0]) 
    else:
        img = img_test

    ## ----------------[ Segmentation basique ]----------------##   
    
    data_s1 = base_segmente(name_1,1,img,af_1)      ## Sigma = 1 -> + de détails et + précise
    data_s4 = base_segmente(name_2,7,img,af_4)      ## Sigma = 4 -> + Global + fiable et - précise 

    ## ----------------[ Amélioration de la base ]----------------## 

    Data_evolved = evolve_segmente(name_3,data_s1,data_s4,parameter,af_p)

    return Data_evolved[4]

def Full_extract_stack ( stack_test,parameter,af = affichage ,save=0):              ## extraction complete des contours pour une pile d'image

    ## ----------------[ préparation  ]----------------##        
    H, W, nb_images = stack_test.shape                          # Récupération des dimensions 
    segment_stack = np.zeros((H, W, nb_images), dtype=np.float64)

    ## ----------------[ segmentation ]----------------##  

    for i in range(nb_images):
        segment_stack[:,:,i]=Full_extract(stack_test[:,:,i],parameter)
        print(f" Extracte : {i}/{nb_images}")
    
    return segment_stack

## -------[ Etapes de segmentations ]------- ##

def base_segmente (name,sigma,img_lvg,af=affichage[0]):                    ## Segmentation simple en fonction de Sigma 
    
    ## ----------------[ Traitement ]----------------##

    nbr_lgn,nbr_col=img_lvg.shape                          ## Récupération de la taillede l'image
    f_v , f_h = vertical_and_horizontal_f(sigma,img_lvg)    ## Création des filtres
    img_dx,img_dy = all_derive(img_lvg,f_v,f_h)             ## Calcul de la dérivée
    Acor = Acor_Matrice(img_dx,img_dy,nbr_lgn,nbr_col)      ## Matrice Acor
    Delta,Lambda = Lambda_matrice(Acor,nbr_lgn,nbr_col)     ## Matrice lambda et delta
    n_chatoux = Chatoux(Lambda,nbr_lgn,nbr_col)             ## Norme de chatoux
    n_bin = binarise( n_chatoux,nbr_lgn,nbr_col)            ## Binarisation

    ## ----------------[ Stockage Affichage ]----------------##

    data_show = []                              ## Staockage des données dans une liste
    if af[0] == 1 or af[1] == 1 :
        data_show.append(img_lvg)
    if af[2] == 1 : 
        f_v_affiche = gaussian_F(sigma)         ## Calcul adaptée a l'affichage
        f_h_affiche = gaussian_F(sigma).T       ## Calcul adaptée a l'affichage
        data_show.append(f_v_affiche)
        data_show.append(f_h_affiche)
    if af[3] == 1 :
        data_show.append(img_dx)
        data_show.append(img_dy)
    if af[4] == 1 :
        data_show.append(Acor)
    if af[5] == 1 :
        data_show.append(Delta)
        data_show.append(Lambda)
    if af[6] == 1 :
        data_show.append(n_chatoux)
    if af[7] == 1 :
        data_show.append(n_bin)

    lo.affiche_segment(af,data_show,name)       ## Affichage 

    ## ----------------[ Stockage des données ]----------------##

    data_save = []                  ## Staockage des données dans une liste
    data_save.append(img_lvg)       # data_save [0] : Image en niveau de gris
    data_save.append(f_v)           # data_save [1] : Filtre vertical
    data_save.append(f_h)           # data_save [2] : Filtre Horizontal
    data_save.append(img_dx)        # data_save [3] : Dérivée dX
    data_save.append(img_dy)        # data_save [4] : Dérivée dY
    data_save.append(Acor)          # data_save [5] : Matrice Acor
    data_save.append(Delta)         # data_save [6] : Matrice Delta
    data_save.append(Lambda)        # data_save [7] : Matrices des Lambda ( + et - )
    data_save.append(n_chatoux)     # data_save [8] : Norme de Chatoux
    data_save.append(n_bin)         # data_save [9] : Contours binaires 

    return data_save

def evolve_segmente(name,data_1,data_4,param=v_param, af = af_post) :       ## Amélioration de segmentation en multi-echelle 

    ## ----------------[ Traitement ]----------------##

    norm,arg_nabs = grad_norme_arg(data_1[3],data_1[4])     ## Calcul de la Norme et de l'Argument / Angle du gradient 
    arg = np.abs(arg_nabs)                                  ## Calcul de la valeurs absolue de l'Angle
    img_conf = conf_contours(data_1[9],data_4[9],norm,arg)  ## Estimation de la confiance des pixels de contours
    img_conf_fort =binarize_confidence(img_conf, 0.1)       ## Seuillage de la Confiance ( + clean des erreurs )

    ## ----------------[ Stockage Affichage ]----------------##

    data_show = []                                      ## Staockage des données dans une liste
    if af[0] == 1 :
        data_show.append(norm)
    if af[1] == 1 :
        data_show.append(arg_nabs)
    if af[2] == 1 : 
        data_show.append(arg)
    if af[3] == 1 :
        data_show.append(img_conf)
    if af[4] == 1 :
        data_show.append(img_conf_fort)

    lo.affiche_segment_evoleved(af,data_show,name)      ## Affichage 

    ## ----------------[ Stockage des données ]----------------##

    data_evolved = []                   ## Staockage des données dans une liste
    data_evolved.append(norm)           #  data_evolved [0] : Norme du gradient ( sigma = 1 )
    data_evolved.append(arg_nabs)       #  data_evolved [1] : Argument / Angle du gradient ( sigma = 1 )
    data_evolved.append(arg)            #  data_evolved [2] : Valeurs absolue de l'argument
    data_evolved.append(img_conf)       #  data_evolved [3] : Confiance des contours
    data_evolved.append(img_conf_fort)  #  data_evolved [4] : Contours extrait

    return data_evolved

## ====================================================[ Dérivée ]==================================================== ##

## -------[ créations des filtres ]------- ##

def gaussian_F(sigma):
    """
    Génère un filtre 2D basé sur la dérivée d'une gaussienne, utilisé pour détecter les gradients 
    dans les images (similaire au filtre Sobel mais en version lissée).

    Paramètre :
        sigma (float) : Écart-type de la gaussienne (contrôle la largeur et la douceur du filtre)

    Retour :
        filter_2d (np.ndarray) : 
            - Filtre 2D de forme (N, N) avec N = 2 * ceil(3 * sigma) + 1
            - Représente la dérivée d’une gaussienne selon l’axe horizontal (dx), avec un lissage 
              gaussien dans l’axe vertical
            - Utilisable pour extraire des gradients dans les domaines spatial ou fréquentiel (ex. via FFT)
    """

    ## ---------[ Étape 1 : Définir la taille du filtre ]--------- ##
    size = int(2 * math.ceil(3 * sigma) + 1)     # Taille du filtre = ±3 sigma + centre
    demi_size = int(math.ceil(3.0 * sigma))      # Moitié de la taille

    ## ---------[ Étape 2 : Initialiser les filtres 1D ]--------- ##
    LP_filter = np.zeros((size, 1), dtype=np.float32)  # Filtre passe-bas (gaussien)
    HP_filter = np.zeros((size, 1), dtype=np.float32)  # Filtre passe-haut (dérivée)

    t = np.arange(-demi_size, demi_size + 1)            # Coordonnées symétriques centrées sur 0

    # Passe-bas : distribution gaussienne standard
    LP_filter[:, 0] = np.exp(-t**2 / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))

    # Passe-haut : dérivée de la gaussienne (antisymétrique)
    HP_filter[:, 0] = (-t * np.exp(-t**2 / (2.0 * sigma**2))) / (sigma**3 * np.sqrt(2.0 * np.pi))

    ## ---------[ Étape 3 : Produit extérieur → filtre 2D ]--------- ##
    filter_2d = LP_filter @ HP_filter.T  # Produit de convolution croisé : lissage vertical + gradient horizontal

    return filter_2d

def vertical_and_horizontal_f(sigma, theshape):
    """
    Génére deux filtres de dérivée gaussienne 2D (vertical et horizontal), puis les transforme 
    dans le domaine fréquentiel via la FFT (utilisable pour détection de contours / gradients).

    Paramètres :
        sigma (float) :
            - Écart-type de la gaussienne
            - Contrôle la douceur du filtre et la sensibilité au bruit
        theshape (np.ndarray) :
            - Image ou tableau dont les dimensions seront utilisées pour adapter la taille
              des filtres à l’entrée (via `theshape.shape`)

    Retour :
        F_v (np.ndarray) :
            - Filtre de dérivée gaussienne verticale, converti via `fft2`
            - De même taille que `theshape`, prêt à être utilisé en domaine fréquentiel

        F_H (np.ndarray) :
            - Filtre de dérivée gaussienne horizontale (transposé du précédent), converti également en `fft2`
            - Permet d’extraire les gradients horizontaux par multiplication fréquentielle
    """

    ## ---------[ Créer les filtres dérivés en spatial ]--------- ##
    f_vertical = gaussian_F(sigma)             # dérivée selon x, lissée selon y
    f_horizontal = f_vertical.T                # dérivée selon y, lissée selon x

    ## ---------[ Transformer en domaine fréquentiel ]--------- ##
    F_v = np.fft.fft2(f_vertical, s=theshape.shape)     # filtre vertical
    F_H = np.fft.fft2(f_horizontal, s=theshape.shape)   # filtre horizontal

    return F_v, F_H

## -------[ Dérivation ]------- ##

def all_derive(img, f_v, f_h):
    """
    Calcule les dérivées (ou gradients) d'une image 2D selon les axes vertical et horizontal
    en utilisant des filtres de dérivée gaussienne dans le domaine fréquentiel (FFT).

    Paramètres :
        img (np.ndarray) :
            - Image 2D en niveaux de gris (float32 recommandé)
            - Peut provenir d'une image lissée ou brute

        f_v (np.ndarray) :
            - Filtre de dérivée gaussienne verticale (pour dx), déjà transformé avec fft2
            - Appliqué via multiplication fréquentielle

        f_h (np.ndarray) :
            - Filtre de dérivée gaussienne horizontale (pour dy), fft2 déjà effectuée

    Retour :
        img_dx (np.ndarray) : 
            - Dérivée selon x (i.e., changement dans la direction horizontale)
        img_dy (np.ndarray) : 
            - Dérivée selon y (i.e., changement dans la direction verticale)

    Remarques :
        - Le produit dans le domaine de Fourier équivaut à une convolution dans le domaine spatial
        - Plus rapide pour grandes images que la convolution directe
        - Le résultat est ramené dans le domaine spatial avec ifft2, puis la partie réelle est conservée
    """

    ## ---------[ Transformée de Fourier de l’image ]--------- ##
    img_fft = np.fft.fft2(img)

    ## ---------[ Calcul des dérivées via multiplication fréquentielle ]--------- ##
    img_dx = np.real(np.fft.ifft2(img_fft * f_v))  # Gradient horizontal (dérivée selon x)
    img_dy = np.real(np.fft.ifft2(img_fft * f_h))  # Gradient vertical (dérivée selon y)

    return img_dx, img_dy
    
def derive_stack(img_stack, f_v, f_h):
    """
    Calcule les dérivées (ou gradients) de chaque image dans une pile 3D selon les directions 
    verticale et horizontale, en utilisant des filtres de dérivée gaussienne en domaine fréquentiel.

    Paramètres :
        img_stack (numpy.ndarray) : Pile d'images 2D en niveaux de gris, 
                                    sous forme d'une matrice 3D de taille (hauteur, largeur, nombre_images).
        f_v (numpy.ndarray) : Filtre de dérivée gaussienne verticale (FFT2), de même taille que chaque image.
        f_h (numpy.ndarray) : Filtre de dérivée gaussienne horizontale (FFT2), de même taille que chaque image.

    Retour :
        deriv_x_stack (numpy.ndarray) : Pile 3D des dérivées selon l'axe vertical (gradients horizontaux), 
                                        de même taille que img_stack.
        deriv_y_stack (numpy.ndarray) : Pile 3D des dérivées selon l'axe horizontal (gradients verticaux), 
                                        de même taille que img_stack.

    Notes :
        Chaque image de la pile est transformée dans le domaine fréquentiel, filtrée, puis reconstruite pour extraire
        ses dérivées selon les deux axes. Le traitement est vectorisé sur chaque plan indépendamment.
    """
    # Récupération des dimensions 
    H, W, nb_images = img_stack.shape
    # Vérification des dimensions des filtres
    if f_v.shape != (H, W) or f_h.shape != (H, W):
        raise ValueError(f"Dimensions incompatibles : "
                         f"f_v={f_v.shape}, f_h={f_h.shape}, image={(H, W)}")

    # Initialisation des piles pour les dérivées
    deriv_x_stack = np.zeros((H, W, nb_images), dtype=np.float64)
    deriv_y_stack = np.zeros((H, W, nb_images), dtype=np.float64)


    # Calcul des dérivées pour chaque image de la pile
    for i in range(nb_images):
        dx, dy = all_derive(img_stack[:, :, i], f_v, f_h)
        deriv_x_stack[:, :, i] = dx
        deriv_y_stack[:, :, i] = dy
        print(f"derivé {i+1}/{nb_images}")


    return deriv_x_stack, deriv_y_stack

## ====================================================[ Norme de chatoux ]==================================================== ##

## -------[ Matrice Acor ]------- ##

def Acor_Matrice(dx, dy, ligne, colonne):
    """
    Calcule la matrice d'auto-corrélation locale (Acor) à partir des gradients d'une image multicanale.
    Cette matrice est utilisée pour caractériser les variations locales de l'image (ex : détection de structures, orientation).

    Paramètres :
        dx (np.ndarray) :
            - Gradient (dérivée) selon X, de taille (ligne, colonne)
        dy (np.ndarray) :
            - Gradient (dérivée) selon Y, de taille (ligne, colonne)
        ligne (int) :
            - Hauteur (nombre de lignes) de l'image
        colonne (int) :
            - Largeur (nombre de colonnes) de l'image

    Retour :
        Acor (np.ndarray) :
            - Matrice d'auto-corrélation de taille (ligne, colonne, 3)
            - Contient pour chaque pixel : [dx², dx*dy, dy²]
    """

    ## ---------[ Initialisation de la matrice résultat ]--------- ##
    Acor = np.zeros((ligne, colonne, 3), dtype=np.float32)

    ## ---------[ Calcul vectorisé des composantes ]--------- ##
    dx_dx = dx * dx        # ∂I/∂x au carré
    dx_dy = dx * dy        # produit croisé ∂I/∂x * ∂I/∂y
    dy_dy = dy * dy        # ∂I/∂y au carré

    ## ---------[ Remplissage de la matrice ]--------- ##
    Acor[:, :, 0] = dx_dx
    Acor[:, :, 1] = dx_dy
    Acor[:, :, 2] = dy_dy

    return Acor


def Acor_matrice_stack(img_dx, img_dy):
    """
    Calcule la matrice d'autocorrélation (ou tenseur structurel) pour chaque pixel 
    à travers une pile d'images 3D de gradients dx et dy.

    Paramètres :
        img_dx (np.ndarray) :
            - Pile 3D de dérivées selon x (axe horizontal), de forme (H, W, N)
            - H : hauteur, W : largeur, N : nombre d’images

        img_dy (np.ndarray) :
            - Pile 3D de dérivées selon y (axe vertical), même forme que img_dx

    Retour :
        Acor_stack (np.ndarray) :
            - Pile 4D des tenseurs d'autocorrélation par pixel et image
            - Forme : (H, W, 3, N), où les 3 composantes sont :
                  [ Gxx = dx², Gyy = dy², Gxy = dx*dy ]

    Détails :
        - Chaque pixel est associé à une matrice symétrique 2x2 :
            | Gxx  Gxy |
            | Gxy  Gyy |
        - Cette matrice est représentée sous forme compressée par un vecteur [Gxx, Gyy, Gxy]
        - Le traitement est effectué image par image pour éviter des calculs volumineux

    Affiche :
        - Une barre de progression textuelle (image par image)
    """

    H, W, nb_images = img_dx.shape

    ## ---------[ Allocation du tenseur résultat ]--------- ##
    Acor_stack = np.zeros((H, W, 3, nb_images), dtype=np.float64)

    ## ---------[ Boucle sur chaque image de la pile ]--------- ##
    for i in range(nb_images):
        Acor_stack[:, :, :, i] = Acor_Matrice(img_dx[:, :, i], img_dy[:, :, i], H, W)
        print(f"✅ Acor calculée pour image {i+1}/{nb_images}")

    return Acor_stack


## -------[ Matrice des Lambda et de Delta ]------- ##

def Lambda_matrice(Acor, ligne, colonne):
    """
    Calcule les valeurs propres (λ₁, λ₂) et le Δ associé à la matrice d’autocorrélation pour chaque pixel.

    Paramètres :
        Acor (np.ndarray) :
            - Matrice d’autocorrélation locale de taille (ligne, colonne, 3)
            - Acor[:, :, 0] : Gxx (dx²)
            - Acor[:, :, 1] : Gxy (dx*dy)
            - Acor[:, :, 2] : Gyy (dy²)

        ligne (int) : Hauteur de l’image
        colonne (int) : Largeur de l’image

    Retour :
        delta_matrix (np.ndarray) :
            - Matrice Δ = √((Gxx - Gyy)² + 4 * Gxy²)
            - Taille : (ligne, colonne)
        
        Lbd_matrix (np.ndarray) :
            - Matrice des valeurs propres λ₁, λ₂
            - Taille : (ligne, colonne, 2)
            - λ₁ : plus grande valeur propre
            - λ₂ : plus petite valeur propre

    Utilité :
        - Cette décomposition sert à analyser la structure locale (force et orientation)
        - Peut être utilisée pour la segmentation, la détection de crêtes ou la classification directionnelle
    """

    ## ---------[ Initialisation des résultats ]--------- ##
    delta_matrix = np.zeros((ligne, colonne), dtype=np.float32)
    Lbd_matrix = np.zeros((ligne, colonne, 2), dtype=np.float32)

    ## ---------[ Calcul de delta : équivalent du discriminant ]--------- ##
    # Formule : Δ = √[(Gxx - Gyy)² + 4 * Gxy²]
    delta_matrix = np.sqrt((Acor[:, :, 0] - Acor[:, :, 2])**2 + 4 * Acor[:, :, 1]**2)
    delta_matrix = np.nan_to_num(delta_matrix, nan=0.0)

    ## ---------[ Calcul des valeurs propres λ₁ et λ₂ ]--------- ##
    trace = Acor[:, :, 0] + Acor[:, :, 2]
    Lbd_matrix[:, :, 0] = 0.5 * (trace + delta_matrix)   # λ₁ : max eigenvalue
    Lbd_matrix[:, :, 1] = 0.5 * (trace - delta_matrix)   # λ₂ : min eigenvalue

    ## ---------[ Nettoyage des NaN éventuels ]--------- ##
    delta_matrix = np.nan_to_num(delta_matrix, nan=0.0)
    Lbd_matrix = np.nan_to_num(Lbd_matrix, nan=0.0)

    return delta_matrix, Lbd_matrix

def Delta_Lambda_stack(Acor_stack):
    """
    Calcule les valeurs propres λ₁, λ₂ et le delta Δ pour chaque pixel de chaque image d’une pile 3D.

    Paramètre :
        Acor_stack (np.ndarray) :
            - Tableau de forme (H, W, 3, N)
            - H x W : dimensions spatiales des images
            - 3 : composantes de la matrice d’autocorrélation [Gxx, Gyy, Gxy]
            - N : nombre d’images dans la pile

    Retour :
        delta_matrix (np.ndarray) :
            - Matrice (H, W, N)
            - Contient le discriminant Δ par pixel et par image : Δ = |λ₁ - λ₂|

        Lbd_matrix (np.ndarray) :
            - Matrice (H, W, 2, N)
            - Contient les deux valeurs propres (λ₁, λ₂) par pixel et par image

    Détail :
        Pour chaque image i, on applique Lambda_matrice() sur Acor_stack[:, :, :, i],
        ce qui donne :
            - les valeurs propres λ₁ et λ₂ de chaque matrice 2x2 symétrique locale
            - le delta associé : mesure d’anisotropie (écart entre les λ)
        Cela permet de suivre la structure locale sur une pile temporelle ou multi-angle.
    """

    ## ---------[ Récupération des dimensions ]--------- ##
    H, W, nb_couche, nb_images = Acor_stack.shape

    # Initialisation des résultats
    delta_matrix = np.zeros((H, W, nb_images), dtype=np.float32)
    Lbd_matrix = np.zeros((H, W, 2, nb_images), dtype=np.float32)

    ## ---------[ Calcul image par image ]--------- ##
    for i in range(nb_images):
        delta_matrix[:, :, i], Lbd_matrix[:, :, :, i] = Lambda_matrice(Acor_stack[:, :, :, i], H, W)
        print(f"D/L {i+1}/{nb_images}")

    return delta_matrix, Lbd_matrix

## -------[ Calcul de Chatoux ]------- ##

def Chatoux(Lambda, ligne, colonne):
    """
    Calcule une carte de "force locale" dérivée des deux valeurs propres (λ₁, λ₂)
    issues du tenseur de structure (ou matrice d’autocorrélation) pour une image 2D.

    Paramètres :
        Lambda (np.ndarray) :
            - Matrice de forme (ligne, colonne, 2)
            - Contient les deux valeurs propres λ₁ (forte direction) et λ₂ (faible direction) à chaque pixel

        ligne (int) : Hauteur de l’image (nombre de lignes)
        colonne (int) : Largeur de l’image (nombre de colonnes)

    Retour :
        chatoux (np.ndarray) :
            - Matrice 2D de taille (ligne, colonne)
            - Chaque pixel vaut √(λ₁ + λ₂), c’est-à-dire la racine de la trace du tenseur
            - Ce score est proportionnel à la "quantité locale d’information directionnelle"
              (ex. : bords, textures, zones structurées)

    Intuition :
        - √(λ₁ + λ₂) correspond à la norme du gradient moyen local
        - Cette mesure est souvent utilisée pour filtrer ou pondérer des cartes de réponses
          (notamment en segmentation, détection de contours, ou analyse orientationnelle)
    """

    ## ---------[ Initialisation ]--------- ##
    chatoux = np.zeros((ligne, colonne), dtype=np.float32)

    ## ---------[ Calcul vectorisé de la norme locale ]--------- ##
    chatoux[:, :] = np.sqrt(Lambda[:, :, 0] + Lambda[:, :, 1])  # λ₁ + λ₂ = trace ⇒ intensité globale du gradient

    return chatoux

def chatoux_stack(Lambda):
    """
    Calcule, pour chaque image d’une pile 4D contenant des valeurs propres (λ1, λ2),
    la norme √(λ1 + λ2) à chaque pixel. Cette norme mesure l’intensité locale du gradient.

    Paramètre :
        Lambda (numpy.ndarray) : Tableau 4D de dimensions (H, W, 2, N),
                                 contenant les deux valeurs propres [λ1, λ2] pour chaque pixel
                                 de chaque image (N est le nombre d’images dans la pile).

    Retour :
        chatoux_stack (numpy.ndarray) : Tableau 3D de dimensions (H, W, N),
                                        contenant la carte des normes √(λ1 + λ2)
                                        pour chaque image de la pile.
    """
    H, W , nb_couche, nb_images = Lambda.shape

    # Initialisation des matrices
    chatoux_stack = np.zeros((H,W,nb_images),dtype=np.float32)

    # Calcul des normes pour chaque image de la pile
    for i in range(nb_images):
        chatoux_stack[:,:,i] = Chatoux(Lambda[:,:,:,i],H,W)
        print(f"norme {i+1}/{nb_images}")

    return chatoux_stack

## ====================================================[ Etude du gradient ]==================================================== ##

## -------[ Norme ]------- ##

def grad_norme_arg(dx, dy):
    """
    Calcule la norme et l'argument (direction) du vecteur gradient (dx, dy)
    pour chaque pixel.

    Args:
        dx (np.ndarray): Dérivée selon x.
        dy (np.ndarray): Dérivée selon y.

    Returns:
        norme (np.ndarray): Norme du gradient sqrt(dx² + dy²).
        arg (np.ndarray): Angle du gradient (en radians, entre -π et π).
    """
    norme = np.sqrt(dx**2 + dy**2)       # Norme euclidienne du gradient
    arg = np.arctan2(dy, dx)             # Angle du gradient, gère les 4 quadrants
    return norme, arg

## -------[ Angle ]------- ##

def Angle_theta(Lambda, acor, ligne, colonne):
    """
    Calcule un angle caractéristique (Theta_plus) pour chaque pixel, basé sur les
    valeurs propres (Lambda) et le tenseur de structure (acor).

    Args:
        Lambda (np.ndarray): Matrice des valeurs propres (H, W, 2) pour chaque pixel.
        acor (np.ndarray): Matrice du tenseur 2x2 (H, W, 3) sous forme [a, b, g]
                           où chaque pixel a un tenseur symétrique [[a, b], [b, g]].
        ligne (int): Nombre de lignes de l’image.
        colonne (int): Nombre de colonnes de l’image.

    Returns:
        Theta_plus (np.ndarray): Angle local principal, ramené à petite échelle (divisé par 1e6).
    """
    # Initialisation de la matrice résultat
    Theta_plus = np.zeros((ligne, colonne), dtype=np.float32)

    # Parcours de chaque pixel
    for i in range(ligne):
        for j in range(colonne):
            a = acor[i, j, 0]  # Élément (0,0) du tenseur
            b = acor[i, j, 1]  # Élément (0,1) == (1,0) (symétrique)
            g = acor[i, j, 2]  # Élément (1,1)

            # Mesure de l'anisotropie (utilisée pour déterminer le cas)
            tau = (a - g) ** 2 + b ** 2

            # Différence des valeurs propres (utile pour éviter les divisions par 0)
            denom = Lambda[i, j, 0] - Lambda[i, j, 1]

            if denom == 0:
                Theta_plus[i, j] = 0.0  # ou np.nan si tu veux signaler une ambiguïté
                continue

            # Calcul d’un rapport caractéristique
            value = (Lambda[i, j, 0] - a) / denom
            value_clipped = np.clip(value, -1.0, 1.0)  # pour éviter arcsin invalide

            angle = np.arcsin(value_clipped)  # angle associé à la direction principale

            # Ajustement de l’angle selon la structure du tenseur (tau et signe de b)
            if tau > 0:
                # Cas général, orientation bien définie
                Theta_plus[i, j] = (np.sign(b) * angle) / 1e6
            else:
                # Cas où le tenseur est isotrope ou mal conditionné
                Theta_plus[i, j] = (np.sign(b) * angle + np.pi / 2) / 1e6

    return Theta_plus

## ====================================================[ Calcul de confiance des contours ]==================================================== ##

## -------[ Globale  ]------- ##

def conf_contours(img_bin_s1, img_bin_s4, norm, img_theta, param=v_param):
    """
    Calcule une carte de confiance pour des contours détectés à partir d'images binaires
    et d'informations de gradient.

    Arguments :
    - img_bin_s1 : Image binaire des contours à faible sigma (détails fins)
    - img_bin_s4 : Image binaire des contours à fort sigma (contours plus robustes)
    - norm : Norme du gradient (intensité du changement)
    - img_theta_abs : Valeur absolue de l'angle du gradient
    - param : Liste de paramètres : param [0] : poid du contours pour sigma fort (4)
                                          [1] : poid de la norme du gradient 
                                          [2] : poid de l'angle du gradient 
                                          [3] : Coefficient de boost de confiance 
                                          [4] : Coefficient de réduction de confiance 
                                          [5] : Distance min pour considérer un éléments continue ( nombre de pixel )

    Retourne :
    - confidence_map : Carte de confiance finale des contours
    """
    # Étape 1 : Calcul de la carte de confiance initiale
    confidence_map = compute_initial_confidence_v2(
        img_bin_s1,
        img_bin_s4,
        norm,
        img_theta,
        param[0],  # poids contours sigma fort (priorité aux contours plus nets)
        param[1],  # poids de la norme du gradient (intensité du bord)
        param[2]   # poids de l'angle du gradient (orientation du bord)
    )

    # Étape 2 : Ajustement de la confiance locale en fonction des voisins
    # On cherche le maximum local dans une fenêtre de 7x7 autour de chaque pixel
    neighborhood_max = maximum_filter(confidence_map, size=7)
    
    # Création d’un masque indiquant où des voisins sont plus confiants que le pixel courant
    boost_mask = (neighborhood_max > confidence_map)
    
    # Boost ou réduction de la confiance selon la présence de voisins plus forts
    confidence_map = np.where(
        boost_mask,
        confidence_map * param[3],  # boost de confiance si entouré de voisins plus forts
        confidence_map * param[4]   # réduction sinon (isolation = moins de confiance)
    )

    # Étape 3 : Suppression des petits segments peu connectés
    connectivity_mask = filter_by_connectivity_degrade(confidence_map, param[5])  # supprime les éléments trop courts

    # Application finale du masque de connectivité
    confidence_map *= connectivity_mask.astype(np.float32)

    return confidence_map

## -------[ Calcul de la confiance ]------- ##

def compute_initial_confidence(img_bin_s1, img_bin_s4, norm, img_theta_abs, w1, w2, w3):
    """
    Calcule une carte de confiance initiale pour les contours, en combinant
    plusieurs sources pondérées (binaire, gradient, orientation).

    Args:
        img_bin_s1 (np.ndarray): Image binaire (uint8 ou bool) indiquant les pixels considérés comme contours potentiels.
        img_bin_s4 (np.ndarray): Image binaire avec contours plus "forts" (ex. seuil plus élevé).
        norm (np.ndarray): Norme du gradient (image de même taille).
        img_theta_abs (np.ndarray): Valeur absolue de l’angle du gradient, normalisé entre 0 et 1.
        w1 (float): Poids du canal binaire fort (img_bin_s4).
        w2 (float): Poids de la norme du gradient.
        w3 (float): Poids de l’orientation (proximité de 0 = alignement fort avec un axe donné).

    Returns:
        np.ndarray: Carte de confiance initiale (float32), même taille que les images d’entrée.
    """
    # Initialisation de la carte de confiance à 0
    confidence_map = np.zeros_like(img_bin_s1, dtype=np.float32)

    norm_2=lo.normalize_0_1(norm)

    # Récupère les indices des pixels actifs (valeurs à 1) dans img_bin_s1
    indices = np.where(img_bin_s1 == 1)

    # Composantes pondérées du score de confiance :
    c1 = img_bin_s4[indices] * w1           # Présence dans le masque fort (binaire stricte)
    c2 = norm_2[indices] * w2                 # Intensité du gradient local
    c3_full = compute_angle(img_theta_abs, img_bin_s1, w3)
    c3 = c3_full[indices]

    # Calcul final : somme pondérée des composantes sur les pixels actifs
    confidence_map[indices] =  c1 + c2 + c3

    return confidence_map

def compute_initial_confidence_v2(img_bin_s1, img_bin_s4, norm, img_theta_abs, w1, w2, w3):
    """
    Calcule une carte de confiance initiale pour les contours, en combinant
    plusieurs sources pondérées (binaire, gradient, orientation).

    Args:
        img_bin_s1 (np.ndarray): Image binaire (uint8 ou bool) indiquant les pixels considérés comme contours potentiels.
        img_bin_s4 (np.ndarray): Image binaire avec contours plus "forts" (ex. seuil plus élevé).
        norm (np.ndarray): Norme du gradient (image de même taille).
        img_theta_abs (np.ndarray): Valeur absolue de l’angle du gradient, normalisé entre 0 et 1.
        w1 (float): Poids du canal binaire fort (img_bin_s4).
        w2 (float): Poids de la norme du gradient.
        w3 (float): Poids de l’orientation (proximité de 0 = alignement fort avec un axe donné).

    Returns:
        np.ndarray: Carte de confiance initiale (float32), même taille que les images d’entrée.
    """
    # Initialisation de la carte de confiance à 0
    confidence_map = np.zeros_like(img_bin_s1, dtype=np.float32)

    norm_2=lo.normalize_0_1(norm)

    # Masque combiné : pixels binaires OU pixels avec gradient significatif
    mask = (img_bin_s1 == 1) | (norm_2 > 0.2)
    indices = np.where(mask)

    # Composantes pondérées du score de confiance :
    c1 = img_bin_s4[indices] * w1           # Présence dans le masque fort (binaire stricte)
    c2 = norm[indices] * w2                 # Intensité du gradient local
    c3_full = compute_angle_with_perpendicular_boost(img_theta_abs, img_bin_s1, w3)
    c3 = c3_full[indices]

    # Calcul final : somme pondérée des composantes sur les pixels actifs
    confidence_map[indices] =  c1 + c2 + c3

    return confidence_map

## -------[ Impact de l'angle ]------- ##

def angle_difference(a, b):
    """
    Calcule la différence cyclique minimale entre deux angles dans [0, 1].
    Exemple : diff(0.95, 0.05) = 0.10
    """
    diff = np.abs(a - b)
    return np.minimum(diff, 1 - diff)

def compute_angle(img_theta_abs, img_bin_s1, w3, similarity_threshold=0.1, min_similar_count=4):
    """
    Calcule une carte de confiance basée sur la cohérence locale d'orientation.

    Args:
        img_theta_abs (np.ndarray): Image des angles normalisés dans [0, 1].
        img_bin_s1 (np.ndarray): Masque binaire des pixels d'intérêt.
        w3 (float): Poids à appliquer à cette composante.
        similarity_threshold (float): Seuil pour considérer deux angles comme similaires.
        min_similar_count (int): Nombre minimum de voisins similaires pour confiance forte.

    Returns:
        np.ndarray: Carte des scores c3 de confiance (même taille que l'image).
    """
    def coherence_score(patch):
        center_angle = patch[len(patch) // 2]
        diffs = angle_difference(patch, center_angle)
        
        similar_neighbors = np.sum(diffs < similarity_threshold)
        # On retire 1 pour ne pas compter le centre lui-même
        similar_neighbors -= 1
        
        # Score : proportion de voisins similaires (entre 0 et 1)
        return min(similar_neighbors / (len(patch) - 1), 1.0)

    # Appliquer le filtre local
    confidence = generic_filter(
        img_theta_abs.astype(np.float32),
        function=coherence_score,
        size=3,   # voisinage 3x3
        mode='wrap'
    )

    # Masquer les zones hors binaire
    confidence *= img_bin_s1.astype(np.float32)

    # Appliquer le poids
    return confidence * w3

def compute_angle_with_perpendicular_boost(
    img_theta_abs, img_bin_s1, w3, similarity_threshold=0.1, min_similar_count=4, boost_factor=0.5
):
    """
    Calcule une carte de confiance c3 et booste les voisins perpendiculaires à l'orientation locale.

    Args:
        img_theta_abs (np.ndarray): Angles normalisés [0,1].
        img_bin_s1 (np.ndarray): Masque des pixels d'intérêt.
        w3 (float): Poids de la confiance angulaire.
        similarity_threshold (float): Seuil de cohérence locale.
        min_similar_count (int): Voisins similaires requis.
        boost_factor (float): Pourcentage du score transféré aux voisins perpendiculaires.

    Returns:
        np.ndarray: Carte des scores c3 avec boost des voisins perpendiculaires.
    """
    h, w = img_theta_abs.shape
    c3 = np.zeros_like(img_theta_abs, dtype=np.float32)

    def coherence_score(patch):
        center_angle = patch[len(patch) // 2]
        diffs = angle_difference(patch, center_angle)
        similar_neighbors = np.sum(diffs < similarity_threshold) - 1
        return min(similar_neighbors / (len(patch) - 1), 1.0)

    # Score local de confiance
    base_c3 = generic_filter(
        img_theta_abs.astype(np.float32),
        function=coherence_score,
        size=3,
        mode='wrap'
    )

    # Boost des pixels perpendiculaires
    for y in range(h):
        for x in range(w):
            if img_bin_s1[y, x] == 0:
                continue

            angle = img_theta_abs[y, x]  # dans [0, 1]
            # Angle perpendiculaire : +0.25 et -0.25 (dans [0, 1], modulo 1)
            angle_rad = (angle + 0.25) % 1.0 * 2 * np.pi  # converti en radians
            dy = int(round(np.sin(angle_rad)))
            dx = int(round(np.cos(angle_rad)))

            # Coordonnées des deux voisins perpendiculaires (wrap-around)
            y1 = (y + dy) % h
            x1 = (x + dx) % w
            y2 = (y - dy) % h
            x2 = (x - dx) % w

            val = base_c3[y, x] * boost_factor
            c3[y, x] += base_c3[y, x]
            c3[y1, x1] += val
            c3[y2, x2] += val

    # Masquer hors zone
    c3 *= img_bin_s1.astype(np.float32)

    return c3 * w3

## -------[ Corrections ]------- ##

def filter_by_connectivity_degrade(confidence_map, min_size, degrade_factor=0.1):
    """
    Réduit fortement les valeurs de confiance pour les régions trop petites au lieu de les supprimer.

    Arguments :
    - confidence_map : carte de confiance (float32)
    - min_size : taille minimale (en nombre de pixels) pour qu’un composant soit conservé
    - degrade_factor : facteur multiplicatif pour les composants trop petits (ex: 0.1 = 90% de réduction)

    Retour :
    - confidence_map_modifiée : même taille que l’entrée, avec certains pixels décrémentés
    """
    # Création d’un masque binaire des zones de confiance non nulles
    mask_bin = (confidence_map > 0).astype(np.uint8)

    # Étiquetage des composantes connectées
    labeled, _ = label(mask_bin)
    component_sizes = np.bincount(labeled.ravel())

    # Trouver les étiquettes à dégrader
    small_labels = np.where(component_sizes < min_size)[0]

    # Masque des pixels à affaiblir
    degrade_mask = np.isin(labeled, small_labels)

    # Appliquer la dégradation
    confidence_map = confidence_map.copy()
    confidence_map[degrade_mask] *= degrade_factor

    return confidence_map

def close_by_gradient(contour_map, grad_norm, grad_angle, max_dist=15, angle_tol=np.pi/6):
    """
    Ferme les contours en reliant les extrémités selon le gradient, version optimisée.

    contour_map : image binaire des contours
    grad_norm : norme du gradient
    grad_angle : angle du gradient (en radians)
    max_dist : distance max pour relier les extrémités
    angle_tol : tolérance sur la différence d'angle
    """

    # 1. Binarisation + squelette
    contour_bin = (contour_map > 0).astype(np.uint8)
    skeleton = skeletonize(contour_bin).astype(np.uint8)

    # 2. Trouver les extrémités (pixels ayant 1 seul voisin dans le squelette)
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    endpoints_mask = (skeleton == 1) & (neighbors == 2)
    endpoints = np.transpose(np.nonzero(endpoints_mask))  # (N, 2) : (y, x)

    # 3. Pour chaque extrémité, calculer les trajectoires possibles
    closed_contours = contour_bin.copy()

    h, w = contour_map.shape

    for y0, x0 in endpoints:
        theta = grad_angle[y0, x0]
        dx = np.round(np.cos(theta)).astype(int)
        dy = np.round(np.sin(theta)).astype(int)

        # Calcul vectorisé de tous les points candidats le long du rayon directionnel
        xs = x0 + dx * np.arange(1, max_dist + 1)
        ys = y0 + dy * np.arange(1, max_dist + 1)

        # Garder uniquement les points valides dans l'image
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[valid_mask]
        ys = ys[valid_mask]

        # Vérifier présence de contours et tester l'angle
        for x1, y1 in zip(xs, ys):
            if closed_contours[y1, x1]:
                theta2 = grad_angle[y1, x1]
                delta_theta = np.abs(np.mod(theta - theta2 + np.pi, 2 * np.pi) - np.pi)
                if delta_theta < angle_tol:
                    # Utilise skimage.draw.line pour tracer une ligne entre deux pixels
                    rr, cc = skimage_line(y0, x0, y1, x1)
                    closed_contours[rr, cc] = 1
                    break  # Stop après connexion
    return closed_contours

## ====================================================[ Seuillage ]==================================================== ##

## -------[ Norme ]------- ##

def binarise(Chatoux, ligne, colonne,s=0.3):
    """
    Binarisation d'une matrice 2D selon un seuil défini à mi-chemin entre la valeur minimale et maximale.

    Parameters:
        Chatoux (numpy.ndarray): Matrice 2D de taille (ligne, colonne) à binariser.
        ligne (int): Nombre de lignes.
        colonne (int): Nombre de colonnes.

    Returns:
        numpy.ndarray: Matrice binarisée (0 ou 1) de même taille que Chatoux.
    """
    # Initialisation de la matrice de sortie
    Binair = np.zeros((ligne, colonne), dtype=np.float32)

    # Calcul du seuil : (min + max) / 2
    min_val = np.min(Chatoux)
    max_val = np.max(Chatoux)
    seuil = s * (min_val + max_val)

    # Binarisation : 1 si supérieur ou égal au seuil, sinon 0
    Binair[Chatoux >= seuil] = 1.0

    return Binair

def binarise_stack(norme):
    """
    Binarise une pile d’images 3D représentant des cartes de norme (par exemple, 
    des intensités de gradient telles que √(λ1 + λ2)).

    Paramètre :
        norme (numpy.ndarray) : Tableau 3D de dimensions (H, W, N),
                                contenant les normes à binariser pour chaque pixel
                                et chaque image (N est le nombre d’images dans la pile).

    Retour :
        lbd_stack (numpy.ndarray) : Tableau 3D de dimensions (H, W, N),
                                    contenant les cartes binarisées (valeurs 0 ou 1)
                                    correspondant à chaque image.
    """
    H,W,nb_images = norme.shape

    # Initialisation des matrices
    lbd_stack = np.zeros((H,W,nb_images),dtype=np.float32)

    # Calcul des normes pour chaque image de la pile
    for i in range(nb_images):
        lbd_stack[:,:,i] = binarise(norme[:,:,i],H,W)
        print(f"binair {i+1}/{nb_images}")

    return lbd_stack

## -------[ Confiance ]------- ##

def binarize_confidence(conf_map, threshold=0.1):
    """
    Binarise une carte de confiance et comble les trous locaux en pixels isolés.

    Args:
        conf_map (np.ndarray): Carte de confiance (float32).
        threshold (float): Seuil de binarisation (entre 0 et max(conf_map)).

    Returns:
        np.ndarray: Image binaire des contours forts, trous comblés localement.
    """
    # Étape 1 : Normalisation (si la carte n’est pas déjà normalisée)
    norm_conf = conf_map / np.max(conf_map)

    # Étape 2 : Binarisation par seuil
    binary_output = (norm_conf >= threshold).astype(np.uint8)

    # Étape 3 : Détection des pixels à 0 entourés de ≥7 voisins à 1
    # On compte le nombre de voisins à 1 pour chaque pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary_output, -1, kernel)

    # Pixels à 0 avec au moins 7 voisins à 1 (valeurs dans neighbor_count de 7 ou 8)
    # Note : on exclut le centre en testant binary_output == 0
    fill_mask = (binary_output == 0) & (neighbor_count >= 7)

    # Étape 4 : Mise à jour de la carte binaire
    binary_output[fill_mask] = 1

    return binary_output











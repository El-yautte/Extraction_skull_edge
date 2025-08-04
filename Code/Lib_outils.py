'''
Librairie OUTILS 

2025, GILET Eliott , PALEVOPRIM/X-LIM

    Résumé : Cette librairie contient des fonctions utile de 
manipulation et de traitement générale d'image, de 
pie d'image et de données 3D.

## -------[ Structure de la librairie ]------- ##

## [ Manipulation des fichiers ( .png / .tif ) ] ##
#  [ Ouverture ]
#  [ Visualisation ]
#  [ Enregistrement ]

## [ Manipulation des fichiers ( 3D ) ] ##

## [ Traitement d'images ] ##
#  [ Alignement / rotation d'image ]
#  [ traitement mathématique ]

## [ Affichage ] ##
#  [ Affichage simple ]
#  [ Affichage Mathématiques ]
#  [ Affichage spécifique ]

'''

#################################################################################################################################################################################
## ----------------------------------------[ Zone d'import ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################
import os
import pyvista as pv
import math
import numpy as np
import matplotlib.pyplot as plt
import shutil

from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from skimage.io import imread
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.ndimage import label, binary_opening
from sklearn.decomposition import PCA

#################################################################################################################################################################################
## ----------------------------------------[ Zone de déclaration ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################



#################################################################################################################################################################################
## ----------------------------------------[ Zone de fonction ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################


## ====================================================[ Manipulation des fichiers ( .png / .tif ) ]==================================================== ##

## -------[ Ouverture ]------- ##

def open_image(fichier):# Ouvre une image (TIF, PNG, JPG...)
    """
    Ouvre une image (TIF, PNG, JPG...), la convertit en niveaux de gris et renvoie un tableau 2D.

    Args:
        fichier (str): chemin vers le fichier image

    Returns:
        np.ndarray: image en niveaux de gris sous forme de tableau 2D
    """
    try:
        # ----------[ Chargement et conversion ]----------
        image = Image.open(fichier).convert("L")     # Ouvre l’image et la convertit en niveaux de gris ("L" = 8 bits par pixel)
        return np.array(image)                       # Convertit l’image en tableau NumPy 2D (valeurs de 0 à 255)
    
    except Exception as e:
        # ----------[ Gestion des erreurs ]----------
        print(f"Erreur lors de l'ouverture de l'image : {e}")  # Affiche un message d'erreur explicite
        return None                                             # Retourne None en cas d’échec

def open_image_stack(dossier, extension=".tif"):# Ouvre tous les fichiers
    """
    Ouvre tous les fichiers d'une extension donnée dans un dossier, les convertit en niveaux de gris,
    et les empile dans une matrice 3D.

    Args:
        dossier (str): Chemin du dossier contenant les fichiers image
        extension (str): Extension des fichiers (par ex. '.tif', '.png', '.jpg')

    Returns:
        np.ndarray: Matrice 3D de taille (hauteur, largeur, nombre_images)
    """

    ## ---------[ Recherche des fichiers ]--------- ##
    extension = extension.lower()                                              # Met l’extension en minuscule (pour éviter les problèmes de casse)
    fichiers = [f for f in os.listdir(dossier) if f.lower().endswith(extension)]
    fichiers.sort()                                                            # Trie les fichiers pour garantir l’ordre de chargement

    pile_images = []                                                           # Liste pour stocker les images chargées

    ## ---------[ Chargement et conversion des images ]--------- ##
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)                                # Construit le chemin complet du fichier

        try:
            image = Image.open(chemin).convert("L")                            # Ouvre l’image et la convertit en niveaux de gris
            tab_image = np.array(image)                                        # Convertit en tableau NumPy 2D
            pile_images.append(tab_image)                                      # Ajoute à la pile
            print(f"Image chargée ({extension}) : {i+1}")
        except Exception as e:
            print(f"Erreur avec {fichier} : {e}")                              # Affiche un message d’erreur si le fichier ne peut être lu

    ## ---------[ Vérification et empilement final ]--------- ##
    if not pile_images:
        raise ValueError("Aucune image chargée. Vérifiez le dossier et l'extension.")  # Stoppe si aucune image valide n’a été trouvée

    return np.stack(pile_images, axis=2)                                       # Empile les images le long du 3e axe → matrice (H, W, N)

## -------[ Visualisation ]------- ##

def afficher_volume_tif(dossier):
    """
    Charge un dossier contenant des images .tif (slices 2D) et permet de visualiser 
    l’ensemble comme un volume 3D, slice par slice, à l’aide d’un curseur interactif.

    Args:
        dossier (str): Chemin vers le dossier contenant les fichiers .tif

    Returns:
        None: Affiche une interface matplotlib interactive pour explorer les images

    Notes:
        - Les fichiers .tif sont triés par ordre alphabétique.
        - L'affichage utilise un curseur pour naviguer dans les couches du volume.
        - Les images sont chargées avec `plt.imread` (penser à compatibilité si erreur).
    """

    ## ---------[ Chargement des fichiers .tif du dossier ]--------- ##
    fichiers = sorted([f for f in os.listdir(dossier) if f.lower().endswith('.tif')])

    if not fichiers:
        print("❌ Aucun fichier .tif trouvé dans le dossier.")
        return                                                        # Aucun fichier trouvé, arrêt

    ## ---------[ Construction du volume 3D à partir des slices ]--------- ##
    volume = np.array([
        plt.imread(os.path.join(dossier, f)) for f in fichiers        # Lecture de chaque image en niveaux de gris
    ])
    nb_couches = volume.shape[0]                                      # Nombre total de couches (slices)

    ## ---------[ Initialisation de l’affichage matplotlib ]--------- ##
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)                                   # Laisse de la place en bas pour le slider
    img = ax.imshow(volume[0], cmap='gray')                           # Affiche la première coupe
    ax.set_title(f"Couche 1 / {nb_couches}")                          # Titre initial

    ## ---------[ Création du curseur de navigation ]--------- ##
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])                       # Emplacement du slider (bas de la figure)
    slider = Slider(
        ax_slider, 'Couche', 0, nb_couches - 1,
        valinit=0, valfmt='%0.0f')                                    # Slider entre 0 et nb_couches - 1

    ## ---------[ Mise à jour de l’image lorsque le slider change ]--------- ##
    def update(val):
        idx = int(slider.val)                                         # Indice sélectionné
        img.set_data(volume[idx])                                     # Affiche la nouvelle image
        ax.set_title(f"Couche {idx + 1} / {nb_couches}")              # Met à jour le titre
        fig.canvas.draw_idle()                                        # Rafraîchit l'affichage

    slider.on_changed(update)                                         # Relie le slider à la fonction `update`

    ## ---------[ Affichage interactif final ]--------- ##
    plt.show()                                                        # Lance la fenêtre interactive

def visualiser_volume_binaire(dossier_tif):
    """
    Charge des fichiers TIFF 2D, binarise chaque image selon sa moyenne de niveaux de gris,
    puis affiche les coupes sur les 3 axes avec navigation.

    Paramètre :
    - dossier_tif : str, chemin vers le dossier contenant les images TIFF (slices).
    """
    # Chargement trié des fichiers TIFF
    fichiers = sorted([f for f in os.listdir(dossier_tif) if f.lower().endswith('.tif')])
    if not fichiers:
        print("❌ Aucun fichier .tif trouvé.")
        return

    # Lecture et binarisation des slices
    volume = []
    for fichier in fichiers:
        chemin = os.path.join(dossier_tif, fichier)
        image = imread(chemin, as_gray=True)
        moyenne = np.mean(image)
        binaire = (image > moyenne).astype(np.uint8)
        volume.append(binaire)

    volume = np.stack(volume, axis=0)  # (Z, Y, X)

    # Dimensions
    z, y, x = volume.shape

    # Interface interactive matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(bottom=0.25)

    img_axial = axs[0].imshow(volume[z//2, :, :], cmap='gray')
    axs[0].set_title('Axial (Z)')
    img_sagittal = axs[1].imshow(volume[:, :, x//2], cmap='gray')
    axs[1].set_title('Sagittal (X)')
    img_coronal = axs[2].imshow(volume[:, y//2, :], cmap='gray')
    axs[2].set_title('Coronal (Y)')

    # Sliders
    ax_z = plt.axes([0.1, 0.1, 0.25, 0.03])
    ax_x = plt.axes([0.4, 0.1, 0.25, 0.03])
    ax_y = plt.axes([0.7, 0.1, 0.25, 0.03])

    slider_z = Slider(ax_z, 'Z', 0, z-1, valinit=z//2, valstep=1)
    slider_x = Slider(ax_x, 'X', 0, x-1, valinit=x//2, valstep=1)
    slider_y = Slider(ax_y, 'Y', 0, y-1, valinit=y//2, valstep=1)

    def update(val):
        img_axial.set_data(volume[int(slider_z.val), :, :])
        img_sagittal.set_data(volume[:, :, int(slider_x.val)])
        img_coronal.set_data(volume[:, int(slider_y.val), :])
        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    slider_x.on_changed(update)
    slider_y.on_changed(update)

    plt.show()

## -------[ Enregistrement ]------- ##

def archive_file_saved(base_dir="file_saved", archive_root="archived_file"):
    """
    Vérifie si des fichiers sont présents dans les sous-dossiers de 'file_saved'.
    Si oui :
        - Crée 'unused_bin' si nécessaire,
        - Copie le dossier 'file_saved' dans 'unused_bin' avec un suffixe date-heure,
        - Supprime le dossier original 'file_saved'.

    Args:
        base_dir (str): Nom du dossier à archiver (par défaut "file_saved").
        archive_root (str): Dossier où les archives sont placées (par défaut "unused_bin").
    """
    ## ---------[ Vérification de l'existence du dossier ]--------- ##

    if not os.path.exists(base_dir):
        print(f"Le dossier '{base_dir}' n'existe pas.")
        return

    ## ---------[ Vérifie s'il y a des fichiers à archiver ]--------- ##

    has_data = False
    for root, dirs, files in os.walk(base_dir):             # Parcourt tous les sous-dossiers
        if files:                                            # Si des fichiers sont trouvés
            has_data = True
            break

    if not has_data:
        print(f"Aucune donnée à archiver dans '{base_dir}'.")
        return

    ## ---------[ Préparation du nom d'archive avec timestamp ]--------- ##

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")     # Format date-heure → "20250616_134510"
    archive_name = f"{base_dir}_{timestamp}"                 # Exemple : file_saved_20250616_134510
    archive_path = os.path.join(archive_root, archive_name)  # Chemin final de l'archive

    os.makedirs(archive_root, exist_ok=True)                 # Crée le dossier d'archives s’il n’existe pas

    ## ---------[ Copie + Suppression de l'original ]--------- ##

    shutil.copytree(base_dir, archive_path)                  # Copie tout le dossier vers le dossier d’archives
    shutil.rmtree(base_dir)                                  # Supprime le dossier original après copie

    print(f"✅ Dossier archivé sous : {archive_path}")

def ask_save():
    """
    Demande à l'utilisateur s'il souhaite sauvegarder un dossier.
    Si oui, il peut entrer un nom personnalisé ou générer un nom automatiquement.

    Returns:
        str | int:
            - 0 si l'utilisateur ne veut pas sauvegarder
            - nom du dossier (str) sinon
    """
    ## ---------[ Question initiale : enregistrer ou non ? ]--------- ##
    print("##-----------------------------------------##")
    reponse = input("Enregistrer dossier sauvegarder ? (y = oui / n = non) : ").strip().lower()
    print("##-----------------------------------------##")

    if reponse == 'n':
        return 0                                                      # L'utilisateur ne veut pas sauvegarder

    elif reponse == 'y':
        ## ---------[ L'utilisateur veut sauvegarder : demande du nom ]--------- ##
        print("##-----------------------------------------##")
        nom = input("Entrez un nom pour le dossier (laisser vide pour nom automatique) : ").strip()
        print("##-----------------------------------------##")

        if nom:
            return nom                                                # Nom personnalisé saisi
        else:
            date_heure = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Format : 20250616_145600
            return f"Test_0_{date_heure}"                             # Nom par défaut généré automatiquement

    else:
        ## ---------[ Réponse invalide : reposer la question ]--------- ##
        print("Réponse invalide. Veuillez répondre par 'y' ou 'n'.")
        return ask_save()                                             # Relance récursive de la fonction

def save_space(save):
    """
    Sauvegarde le contenu du dossier 'file_saved' sous un nouveau nom, dans un répertoire de destination.
    Puis nettoie les sous-dossiers 'data_generate' et 'laste_show' dans 'file_saved'.

    Args:
        save (str or int): Nom du dossier de sauvegarde. Si 0, aucune opération n’est effectuée.

    Returns:
        None
    """
    ## ---------[ Vérification de la demande de sauvegarde ]--------- ##
    if save == 0:
        return  # Aucune sauvegarde demandée

    ## ---------[ Demande du chemin de destination ]--------- ##
    print("##-----------------------------------------##")
    chemin = input("Entrez le chemin de destination pour la sauvegarde (par défaut : 'archived_file') : ").strip()
    print("##-----------------------------------------##")

    if chemin == "":
        chemin = "archived_file"  # Valeur par défaut

    ## ---------[ Création du dossier s’il n’existe pas ]--------- ##
    if not os.path.exists(chemin):
        try:
            os.makedirs(chemin)
            print(f"Dossier '{chemin}' créé.")
        except Exception as e:
            print(f"Impossible de créer le dossier '{chemin}' : {e}")
            return
    elif not os.path.isdir(chemin):
        print("Chemin invalide. Opération annulée.")
        return

    ## ---------[ Préparation des chemins source et destination ]--------- ##
    source = "file_saved"                                      # Dossier à sauvegarder
    destination = os.path.join(chemin, str(save))              # Dossier cible avec nom fourni

    ## ---------[ Copie du dossier source ]--------- ##
    try:
        shutil.copytree(source, destination)
        print(f"Dossier copié vers : {destination}")
    except FileExistsError:
        print(f"Le dossier '{destination}' existe déjà. Opération annulée.")
        return
    except Exception as e:
        print(f"Erreur lors de la copie : {e}")
        return

    ## ---------[ Nettoyage des sous-dossiers ]--------- ##
    data_generate_path = os.path.join(source, "data_generate")
    laste_show_path = os.path.join(source, "laste_show")

    for folder in [data_generate_path, laste_show_path]:
        if os.path.isdir(folder):                              # Vérifie si le dossier existe
            for file in os.listdir(folder):                    # Parcourt tous les fichiers
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)                   # Supprime les fichiers ou liens
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)               # Supprime les sous-dossiers
                except Exception as e:
                    print(f"Erreur lors de la suppression dans {folder} : {e}")

    print("Nettoyage terminé dans 'file_saved/data_generate' et 'file_saved/laste_show'")

def save_image_stack(image_stack,type=".png", folder_name="img_stack", save_path="file_saved/data_generate"):
    """
    Enregistre une pile d'images binaires (0 et 1) en PNG dans un dossier spécifié.

    Args:
        image_stack (np.ndarray): Tableau numpy de forme (H, W, N) avec des valeurs 0 ou 1,
                                  où N est le nombre d'images.
        folder_name (str): Nom du dossier dans lequel enregistrer les images. Si 0, la fonction ne fait rien.
        save_path (str): Chemin où le dossier sera enregistré.
    """
    if folder_name == 0:
        return

    # Créer le chemin complet
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    num_images = image_stack.shape[2]
    for i in range(num_images):
        img = image_stack[:, :, i]
        img_uint8 = (img * 255).astype(np.uint8)
        image = Image.fromarray(img_uint8)
        image.save(os.path.join(full_path, f"image_{i:04d}{type}"))

## ====================================================[ Manipulation des fichiers ( 3D ) ]==================================================== ##

def visualiser_fichier_3D(chemin):  # Visualisation de fichiers 3D (.vtk, .vtp, .stl, .ply, .obj)
    """
    Affiche un ou plusieurs fichiers 3D avec PyVista, chacun dans sa propre vue,
    sans superposition ni translation. L'affichage se fait dans une fenêtre unique
    divisée en plusieurs sous-fenêtres (subplots).

    Paramètre :
        chemin (str) : Chemin vers un fichier 3D ou vers un dossier contenant des fichiers 3D.

    Retour :
        None (ouvre une fenêtre interactive avec les objets 3D).
    """

    ## ---------[ Liste des extensions 3D supportées ]--------- ##
    extensions_3d = (".vtk", ".vtp", ".stl", ".ply", ".obj")
    meshes = []  # Liste des objets 3D chargés

    ## ---------[ Cas : chemin = fichier unique ]--------- ##
    if os.path.isfile(chemin) and chemin.lower().endswith(extensions_3d):
        try:
            mesh = pv.read(chemin)
            meshes.append((mesh, os.path.basename(chemin)))  # (objet 3D, nom du fichier)
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du fichier : {e}")
            return

    ## ---------[ Cas : chemin = dossier contenant plusieurs fichiers 3D ]--------- ##
    elif os.path.isdir(chemin):
        fichiers = [f for f in os.listdir(chemin) if f.lower().endswith(extensions_3d)]
        if not fichiers:
            print("⚠️ Aucun fichier 3D trouvé dans ce dossier.")
            return
        for fichier in fichiers:
            chemin_fichier = os.path.join(chemin, fichier)
            try:
                mesh = pv.read(chemin_fichier)
                meshes.append((mesh, fichier))
            except Exception as e:
                print(f"❌ Erreur avec {fichier} : {e}")

    ## ---------[ Cas : chemin invalide ou format non supporté ]--------- ##
    else:
        print("❌ Chemin invalide ou format de fichier non supporté.")
        return

    ## ---------[ Préparation de la grille d'affichage ]--------- ##
    n_meshes = len(meshes)
    cols = math.ceil(math.sqrt(n_meshes))  # Nombre de colonnes
    rows = math.ceil(n_meshes / cols)      # Nombre de lignes

    # Création de la fenêtre divisée en (rows, cols) sous-fenêtres
    plotter = pv.Plotter(shape=(rows, cols))

    ## ---------[ Placement des objets dans chaque sous-fenêtre ]--------- ##
    for idx, (mesh, nom) in enumerate(meshes):
        row = idx // cols
        col = idx % cols
        plotter.subplot(row, col)                  # Active la sous-fenêtre
        plotter.add_text(nom, position='upper_edge', font_size=10)
        plotter.add_mesh(mesh)                     # Ajoute l'objet 3D
        plotter.add_axes()                         # Ajoute des axes

    ## ---------[ Affichage final ]--------- ##
    plotter.show()  # Ouvre la fenêtre interactive

def aligner_fichiers_3D(dossier):
    """
    Aligne tous les fichiers 3D présents dans un dossier par rapport au premier fichier,
    en utilisant l'algorithme ICP (Iterative Closest Point) sur des nuages de points
    échantillonnés à partir des maillages.

    Paramètre :
        dossier (str) : Chemin vers un dossier contenant des fichiers 3D
                        (formats supportés : .ply, .stl, .obj, .vtp)

    Retour :
        None (affiche chaque alignement en superposition avec la référence).
    """

    ## ---------[ Étape 1 : Sélection des fichiers 3D valides ]--------- ##
    extensions_3d = (".ply", ".stl", ".obj", ".vtp")
    fichiers = sorted([f for f in os.listdir(dossier) if f.lower().endswith(extensions_3d)])

    if not fichiers:
        print("❌ Aucun fichier 3D trouvé dans ce dossier.")
        return

    ## ---------[ Étape 2 : Chargement du fichier de référence ]--------- ##
    fichier_reference = os.path.join(dossier, fichiers[0])
    maillage_reference = o3d.io.read_triangle_mesh(fichier_reference)
    pcd_reference = maillage_reference.sample_points_uniformly(number_of_points=10000)
    pcd_reference.estimate_normals()
    print(f"✅ Premier fichier (référence) : {fichier_reference}")

    ## ---------[ Étape 3 : Alignement de chaque fichier cible ]--------- ##
    for fichier in fichiers[1:]:
        fichier_cible = os.path.join(dossier, fichier)

        # Chargement du maillage cible et conversion en nuage de points
        maillage_cible = o3d.io.read_triangle_mesh(fichier_cible)
        pcd_cible = maillage_cible.sample_points_uniformly(number_of_points=10000)
        pcd_cible.estimate_normals()

        # Paramètres pour ICP
        threshold = 5.0  # Distance max pour mise en correspondance
        init_transfo = np.eye(4)  # Matrice de transformation initiale (identité)

        # Application de l'algorithme ICP
        transformation_icp = o3d.pipelines.registration.registration_icp(
            pcd_cible, pcd_reference, threshold, init_transfo,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Application de la transformation au maillage 3D
        maillage_cible.transform(transformation_icp.transformation)
        print(f"✅ Aligné : {fichier_cible}")

        # Affichage des maillages : référence + aligné
        o3d.visualization.draw_geometries([maillage_reference, maillage_cible])

    ## ---------[ Fin du processus ]--------- ##
    print("✅ Alignement terminé pour tous les fichiers.")

## -------[ convertion .tif/.png -> 3D ]------- ##

def save_binary_stack_as_mesh_obj(image_stack, filename="model.obj", save_path="file_saved/3d_models"):
    """
    Transforme une pile binaire (0/1) en une surface 3D au format .obj en utilisant l'algorithme
    de Marching Cubes, et l'enregistre dans un dossier donné.

    Paramètres :
        image_stack (np.ndarray) : Tableau numpy 3D de forme (H, W, D) avec des valeurs binaires (0 ou 1).
        filename (str) : Nom du fichier de sortie (.obj).
        save_path (str) : Dossier dans lequel enregistrer le modèle généré.

    Retour :
        None (sauvegarde le fichier et affiche un message).
    """

    ## ---------[ Création du dossier de sauvegarde si nécessaire ]--------- ##
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## ---------[ Conversion de la pile binaire en volume 3D compatible PyVista ]--------- ##
    volume = pv.wrap(image_stack.astype(np.uint8))

    ## ---------[ Extraction de surface 3D par Marching Cubes ]--------- ##
    try:
        mesh = volume.contour(isosurfaces=[0.5])  # Iso-surface entre 0 et 1
    except Exception as e:
        print(f"❌ Erreur lors de la génération du maillage : {e}")
        return

    ## ---------[ Exportation du maillage en format .obj ]--------- ##
    export_path = os.path.join(save_path, filename)
    mesh.save(export_path)
    print(f"✅ Modèle 3D sauvegardé : {export_path}")


## ====================================================[ Traitement d'images ]==================================================== ##

## -------[ Alignement / rotation d'image ]------- ##

def align_skull(image, orientation='vertical', retourn=0):
    """
    Aligne une image binaire 2D selon son axe principal via PCA, avec extension automatique
    (padding) pour éviter les coupures dues à la rotation.

    Paramètres :
        image (np.ndarray)     : Image binaire (valeurs 0 ou 1), de forme (H, W)
        orientation (str)      : 'vertical' (par défaut) ou 'horizontal' — indique l’axe cible après rotation
        retourn (int)          : 
                                 - 0 → retourne image alignée + centre de rotation
                                 - 1 → retourne angle seul
                                 - 2 → retourne image alignée + centre + angle

    Retour :
        - np.ndarray (si retourn=0 ou 2) : L’image alignée (avec padding)
        - tuple (y, x)                   : Centre de rotation (pixels flottants)
        - float                          : Angle de rotation appliqué (en radians)
    """

    ## ---------[ Étape 1 : Extraction des coordonnées des pixels blancs ]--------- ##
    h, w = image.shape
    coords = np.column_stack(np.nonzero(image))  # [(y1, x1), (y2, x2), ...]

    if len(coords) == 0:
        print("⚠️ Image vide.")
        if retourn == 0:
            return image, (h / 2, w / 2)
        elif retourn == 1:
            return np.pi / 2 if orientation == 'vertical' else 0
        elif retourn == 2:
            return image, (h / 2, w / 2), (np.pi / 2 if orientation == 'vertical' else 0)

    ## ---------[ Étape 2 : Calcul de l'axe principal via PCA ]--------- ##
    pca = PCA(n_components=2)
    pca.fit(coords)
    axis = pca.components_[0]               # Premier axe principal
    angle = np.arctan2(axis[1], axis[0])    # Angle entre l'axe et l'horizontale

    # Calcul de l'angle de rotation nécessaire
    if orientation == 'vertical':
        rotation_angle = np.pi / 2 - angle
    elif orientation == 'horizontal':
        rotation_angle = -angle
    else:
        raise ValueError("❌ L’argument 'orientation' doit être 'vertical' ou 'horizontal'.")

    ## ---------[ Étape 3 : Padding pour éviter les pertes de pixels après rotation ]--------- ##
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))  # Longueur diagonale : taille minimale pour contenir l'image pivotée
    pad_h = (diag_len - h) // 2
    pad_w = (diag_len - w) // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    H, W = padded_image.shape
    center = np.array([H / 2, W / 2])  # Centre de l’image agrandie

    ## ---------[ Étape 4 : Rotation des pixels autour du centre ]--------- ##
    coords = np.column_stack(np.nonzero(padded_image))  # Coordonnées des pixels blancs
    shifted = coords - center                           # Translation pour centrer l’origine
    rot_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ])
    rotated = shifted @ rot_matrix.T                    # Rotation des coordonnées
    new_coords = np.round(rotated + center).astype(int)  # Translation inverse + arrondi

    ## ---------[ Étape 5 : Création de l’image finale alignée ]--------- ##
    new_image = np.zeros_like(padded_image)
    valid = (
        (0 <= new_coords[:, 0]) & (new_coords[:, 0] < H) &
        (0 <= new_coords[:, 1]) & (new_coords[:, 1] < W)
    )
    new_coords = new_coords[valid]                      # Filtrage des coordonnées valides
    new_image[new_coords[:, 0], new_coords[:, 1]] = 1   # Remplissage (vectorisé)

    center_coords = tuple(center)

    ## ---------[ Étape 6 : Retour des résultats en fonction de 'retourn' ]--------- ##
    if retourn == 0:
        return new_image, center_coords
    elif retourn == 1:
        return rotation_angle
    elif retourn == 2:
        return new_image, center_coords, rotation_angle

def stack_alignement_by_lenth(image_stack, target_orientation="horizontal", retourn=0):
    """
    Aligne une pile d'images binaires 3D selon leur plus grand axe (déduit de l’enveloppe convexe),
    en ajoutant un padding pour éviter la perte de données pendant la rotation.

    Args:
        image_stack (np.ndarray): pile d’images (H, W, N), binaires (0 ou 1)
        target_orientation (str): 'horizontal' ou 'vertical' (orientation cible)
        retourn (int): 0 → retourne la pile et le centre, 1 → retourne l’angle seul, 2 → tout retourne

    Returns:
        np.ndarray: pile alignée
        tuple: coordonnées (x, y) du centre
        float: angle de rotation (si demandé)
    """
    ## ----------------[ Préparation des images ]----------------## 

    h, w, n = image_stack.shape                                         # Récupère les dimensions de la pile : hauteur, largeur, nombre d’images
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))                       # Calcule la longueur maximale que peut prendre une image après rotation (diagonale)
    pad_h = (diag_len - h) // 2
    pad_w = (diag_len - w) // 2                                         # Détermine combien de pixels ajouter autour de l’image pour que rien ne soit coupé
    padded_stack = np.pad(
        image_stack,
        ((pad_h, pad_h),
        (pad_w, pad_w),
        (0, 0)),
        mode='constant')                                                # Ajoute du padding (zéros) autour de chaque image de la pile
    H, W = padded_stack.shape[:2]                                       # Dimensions après padding
    max_length = 0                                                      # Longueur maximale rencontrée
    ref_idx = 0                                                         # Index de l’image de référence (celle avec l’axe le plus long)

    ## ----------------[ recherche de l'image de référence ]----------------## 

    for i in range(n):
        coords = np.column_stack(np.nonzero(padded_stack[:, :, i]))                     # Coordonnées (y, x) des pixels actifs
        if len(coords) < 3:
            continue                                                                    # Trop peu de points pour calculer une enveloppe convexe
        try:
            hull = ConvexHull(coords)
            points = coords[hull.vertices]                                              # Calcule l’enveloppe convexe (le contour minimal qui englobe les points)
            dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)    # Calcule toutes les distances entre les points de cette enveloppe
            length = np.max(dists)                                                      # Récupère la plus grande de ces distances
        except:
            continue                                                                    # En cas d’erreur (rare mais possible), passe à l’image suivante
        if length > max_length:                                                         ## Met à jour l’image de référence si la distance trouvée est plus grande
            max_length = length
            ref_idx = i

    ## ----------------[ Définition de la rotation a appliqué ]----------------## 

    ref_image = padded_stack[:, :, ref_idx]
    coords = np.column_stack(np.nonzero(ref_image))                     ## Utilise l’image de référence trouvée pour déterminer l’orientation
    if coords.shape[0] == 0:
        raise ValueError("Image de référence vide.")

    pca = PCA(n_components=2)
    pca.fit(coords)                                                     # Applique une analyse en composantes principales (PCA) pour obtenir l’axe principal
    axis = pca.components_[0]                                           # Premier axe principal
    angle = np.arctan2(axis[1], axis[0])                                # Calcule l’angle de cet axe par rapport à l’horizontale
    if target_orientation == "horizontal":                              ## Détermine l’angle de rotation nécessaire selon l’orientation désirée
        rotation_angle = -angle
    elif target_orientation == "vertical":
        rotation_angle = np.pi / 2 - angle
    else:
        raise ValueError("target_orientation doit être 'horizontal' ou 'vertical'.")
    
    ## ----------------[ Préparation de la rotation ]----------------## 

    rot_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]])             # Matrice de rotation 2D (sens trigonométrique)
    center = np.array([H / 2, W / 2])                                   # Centre de rotation : centre de l’image padded
    aligned_stack = np.zeros_like(padded_stack)                         # Création d'une pile alignée, vide pour l’instant

    ## ----------------[ Rotation des images ]----------------## 

    for i in range(n):                                                  ## Applique la rotation à chaque image de la pile
        img = padded_stack[:, :, i]
        pts = np.column_stack(np.nonzero(img))                          # Coordonnées des pixels actifs
        if pts.shape[0] == 0:
            continue
        shifted = pts - center                                          # Translate les points pour centrer autour de l’origine
        rotated = shifted @ rot_matrix.T                                # Applique la rotation
        new_pts = np.round(rotated + center).astype(int)                # Repositionne les points dans le référentiel original
        aligned = np.zeros((H, W), dtype=np.uint8)                      # Crée une nouvelle image avec les points tournés
        valid = (
            (0 <= new_pts[:, 0]) & (new_pts[:, 0] < H) &
            (0 <= new_pts[:, 1]) & (new_pts[:, 1] < W)
            )
        new_pts = new_pts[valid]                                        # Supprime les points hors des limites de l’image                                
        aligned[new_pts[:, 0], new_pts[:, 1]] = 1                       # Active les nouveaux pixels dans l’image finale
        aligned_stack[:, :, i] = aligned                                # Ajoute cette image à la pile finale
        print(f"Alignement image {i+1}/{n}")

    ## ----------------[ Retour des données ]----------------## 

    # Recalcule les coordonnées du centre du crâne dans l’image de référence alignée
    ref_coords = np.column_stack(np.nonzero(aligned_stack[:, :, ref_idx]))
    center_coords = tuple(np.mean(ref_coords, axis=0)) if len(ref_coords) > 0 else (H / 2, W / 2)

    # Retourne les données selon la valeur de l'argument `retourn`
    if retourn == 0:
        return aligned_stack, (center_coords[1], center_coords[0])  # (x, y)
    if retourn == 1:
        return rotation_angle
    if retourn == 2:
        return aligned_stack, (center_coords[1], center_coords[0]), rotation_angle

## -------[ traitement mathématique ]------- ##

def normalize_0_1(arr):
    """
    Normalise une matrice 2D pour que ses valeurs soient comprises entre 0 et 1.
    
    Args:
        arr (np.ndarray): Matrice à normaliser (2D ou +).

    Returns:
        np.ndarray: Matrice normalisée dans [0, 1], même forme que `arr`.
    """
    arr = arr.astype(np.float32)  # Conversion sécurisée

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr)  # évite division par zéro si matrice constante

def smart_threshold_clean(image, threshold_ratio=0.2, min_hole_size=2000):
    """
    Supprime les grandes zones sombres du fond sans supprimer les détails internes.

    Args:
        image (np.ndarray): Image en niveaux de gris.
        threshold_ratio (float): Seuil relatif au max (ex: 0.2 pour 20%).
        min_hole_size (int): Taille minimale d'une zone sombre à supprimer (en pixels).

    Returns:
        np.ndarray: Image nettoyée.
    """
    image = image.copy()
    
    # 1. Seuil bas pour trouver les zones sombres
    threshold = threshold_ratio * np.max(image)
    low_mask = (image < threshold).astype(np.uint8)

    # 2. Label des composantes sombres
    labeled, _ = label(low_mask)
    sizes = np.bincount(labeled.ravel())

    # 3. Identifier les *grandes zones sombres* → qu'on veut supprimer
    remove_labels = np.where(sizes >= min_hole_size)[0]
    remove_mask = np.isin(labeled, remove_labels)

    # 4. Supprimer ces grandes zones du fond
    image[remove_mask] = 0

    return image

def threshold_relative_20(mat):
    """
    Met à zéro les pixels dont la valeur est inférieure à 20 % de la valeur maximale de la matrice.

    Args:
        mat (np.ndarray): Matrice d'entrée représentant une image (2D ou 3D).

    Returns:
        np.ndarray: Matrice seuillée.
    """
    max_val = np.max(mat)
    threshold = 0.2 * max_val
    result = mat.copy()
    result[result < threshold] = 0
    return result

## ====================================================[ Affichage ]==================================================== ##

## -------[ Affichage simple ]------- ##

def affiche_chatoux(chatoux, titre="Matrice de Chatoux", cmap="viridis", save=0):
    """
    Affiche une matrice sous forme de heatmap (carte de chaleur).
    Si 'save' est une chaîne de caractères, les données et la figure sont automatiquement enregistrées.

    Args:
        chatoux (np.ndarray): Matrice 2D à afficher.
        titre (str): Titre de la heatmap.
        cmap (str): Colormap à utiliser pour l'affichage (ex : 'viridis', 'hot', 'gray', etc.).
        save (str | int): Si chaîne, active la sauvegarde des données + image. Si 0, aucun enregistrement.

    Returns:
        None
    """

    ## ---------[ Préparation à la sauvegarde si demandée ]--------- ##
    if isinstance(save, str):
        data_path = os.path.join("file_saved", "data_generate")  # Dossier pour les données
        img_path = os.path.join("file_saved", "laste_show")      # Dossier pour les images
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

        # Nettoyage du nom pour un usage fichier
        safe_name = f"{save}_{titre}".replace(" ", "_").replace(":", "-")

        # Sauvegarde des valeurs de la matrice dans un fichier texte
        np.savetxt(os.path.join(data_path, f"{safe_name}.txt"), chatoux, fmt="%.6f")

    ## ---------[ Affichage de la heatmap avec matplotlib ]--------- ##
    plt.figure(figsize=(8, 6))                          # Définir la taille de la figure
    im = plt.imshow(chatoux, cmap=cmap)                 # Affichage de la matrice
    plt.colorbar(im)                                    # Ajout de la barre de couleur

    # Gestion du titre (avec ou sans nom de sauvegarde)
    if isinstance(save, str):
        full_title = f"Name : {save}\n{titre}"          # Titre sur deux lignes si sauvegarde
    else:
        full_title = titre

    plt.title(full_title, fontsize=14)
    plt.tight_layout()                                  # Ajuste les marges automatiquement

    ## ---------[ Sauvegarde de la figure si demandée ]--------- ##
    if isinstance(save, str):
        plt.savefig(os.path.join(img_path, f"{safe_name}.png"), dpi=300)  # Image en haute qualité

def afficher_pile(pile_images, cmap='gray'):
    """
    Affiche une pile d'images 2D à l'aide d'un curseur interactif.

    Args:
        pile_images (np.ndarray): Tableau 3D de forme (hauteur, largeur, profondeur),
                                  représentant une pile d'images en niveaux de gris.
        cmap (str): Colormap utilisée pour l'affichage (par défaut 'gray').

    Returns:
        None (ouvre une fenêtre interactive avec matplotlib)
    """

    ## ---------[ Initialisation de la figure matplotlib ]--------- ##
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)  # Réserve de l'espace en bas pour le curseur

    # Affichage de la première image de la pile
    img_display = ax.imshow(pile_images[:, :, 0], cmap=cmap)
    ax.set_title('Image 0')
    ax.axis('off')  # Masque les axes pour un affichage plus propre

    ## ---------[ Ajout d’un slider pour naviguer dans la pile ]--------- ##
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position du slider
    slider = Slider(ax_slider, 'Image', 0, pile_images.shape[2] - 1,
                    valinit=0, valstep=1)

    # Fonction appelée quand l'utilisateur bouge le curseur
    def update(val):
        idx = int(slider.val)  # Index actuel
        img_display.set_data(pile_images[:, :, idx])  # Met à jour l'image affichée
        ax.set_title(f'Image {idx}')  # Met à jour le titre
        fig.canvas.draw_idle()  # Rafraîchit la figure

    slider.on_changed(update)  # Associe la fonction au slider

    ## ---------[ Affichage final ]--------- ##
    plt.show()

## -------[ Affichage Mathématiques ]------- ##

def show_histo ( img_r) :
    valeurs_non_nulles = img_r[img_r > 0]
    plt.figure()
    plt.hist(valeurs_non_nulles, bins=100, color='purple')
    plt.title("Distribution des rayons (pixels)")
    plt.xlabel("Rayon")
    plt.ylabel("Fréquence")
    plt.grid(True)

## -------[ Affichage spécifique ]------- ##

def afficher_image(image, mode="pic", cmap="gray", save=0):
    """
    Affiche une image 2D en niveaux de gris, soit sous forme de matrice brute,
    soit sous forme visuelle, avec option de sauvegarde.

    Paramètres :
        image (numpy.ndarray) : Image 2D (hauteur, largeur) en niveaux de gris.
        mode (str) : Mode d'affichage :
                     - "raw" : Affiche les dimensions et valeurs de la matrice.
                     - "pic" : Affiche l'image avec matplotlib.
        cmap (str) : Colormap utilisée en mode "pic".
        save (str|int) : Si chaîne, active la sauvegarde des données et de l'image.
    """
    # Chemins et nom de fichier sécurisés
    if isinstance(save, str):
        data_path = os.path.join("file_saved", "data_generate")
        img_path = os.path.join("file_saved", "laste_show")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)
        safe_name = f"{save}_image".replace(" ", "_").replace(":", "-")

        # Sauvegarde des données
        np.savetxt(os.path.join(data_path, f"{safe_name}.txt"), image, fmt="%.6f")

    if mode == "raw":
        print(f"Dimensions de l'image : {image.shape}")
        print("Matrice des niveaux de gris (extrait) :")
        print(image)

    elif mode == "pic":
        plt.figure(figsize=(6, 5))
        plt.imshow(image, cmap=cmap)
        title = f"Name : {save}\nImage en niveaux de gris" if isinstance(save, str) else "Image en niveaux de gris"
        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.tight_layout()

        # Sauvegarde de l'image
        if isinstance(save, str):
            plt.savefig(os.path.join(img_path, f"{safe_name}.png"), dpi=300)
    else:
        raise ValueError("Mode inconnu. Utilisez 'raw' ou 'pic'.")

def affiche_filtres(horizontal_filter, vertical_filter, titre_h="Filtre Horizontal", titre_v="Filtre Vertical", cmap="viridis", save=0):
    """
    Affiche les filtres horizontal et vertical avec option de sauvegarde.

    Parameters:
        horizontal_filter (numpy.ndarray): Filtre horizontal.
        vertical_filter (numpy.ndarray): Filtre vertical.
        titre_h (str): Titre pour le filtre horizontal.
        titre_v (str): Titre pour le filtre vertical.
        cmap (str): Colormap utilisée pour afficher les filtres.
        save (str|int): Si chaîne, active la sauvegarde des données et de la figure.
    """
    if isinstance(save, str):
        # Préparation des répertoires
        data_path = os.path.join("file_saved", "data_generate")
        img_path = os.path.join("file_saved", "laste_show")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

        # Nettoyage du nom pour le rendre sûr
        safe_name = save.replace(" ", "_").replace(":", "-")

        # Sauvegarde des matrices
        np.savetxt(os.path.join(data_path, f"{safe_name}_filtre_horizontal.txt"), np.real(horizontal_filter), fmt="%.6f")
        np.savetxt(os.path.join(data_path, f"{safe_name}_filtre_vertical.txt"), np.real(vertical_filter), fmt="%.6f")

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im_h = axes[0].imshow(np.real(horizontal_filter), cmap=cmap, interpolation="nearest")
    axes[0].set_title(f"Name : {save}\n{titre_h}" if isinstance(save, str) else titre_h, fontsize=14)
    axes[0].axis("off")
    plt.colorbar(im_h, ax=axes[0], fraction=0.046, pad=0.04)

    im_v = axes[1].imshow(np.real(vertical_filter), cmap=cmap, interpolation="nearest")
    axes[1].set_title(f"Name : {save}\n{titre_v}" if isinstance(save, str) else titre_v, fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if isinstance(save, str):
        # Sauvegarde de l’image combinée
        plt.savefig(os.path.join(img_path, f"{safe_name}_filtres.png"), dpi=300)

def affiche_acor(Acor, titre="Matrice Acov - Canaux séparés", cmap="viridis", save=0):
    """
    Affiche séparément les trois canaux de la matrice Acor.
    Si 'save' est une chaîne, les données et la figure sont enregistrées.

    Parameters:
        Acor (numpy.ndarray): Matrice Acor à afficher, de taille (NbLg, NbCol, 3).
        titre (str): Titre principal des figures.
        cmap (str): Colormap utilisée pour afficher les canaux.
        save (str|int): Si chaîne, active la sauvegarde des données et de la figure.
    """
    if Acor.shape[-1] != 3:
        raise ValueError("La matrice Acor doit avoir exactement 3 canaux (dernier axe de taille 3).")

    if isinstance(save, str):
        # Crée les répertoires si nécessaire
        data_path = os.path.join("file_saved", "data_generate")
        img_path = os.path.join("file_saved", "laste_show")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

        # Nettoyage du nom
        safe_name = save.replace(" ", "_").replace(":", "-")

        # Sauvegarde des trois canaux
        np.savetxt(os.path.join(data_path, f"{safe_name}_Acor_xx.txt"), Acor[:, :, 0], fmt="%.6f")
        np.savetxt(os.path.join(data_path, f"{safe_name}_Acor_xy.txt"), Acor[:, :, 1], fmt="%.6f")
        np.savetxt(os.path.join(data_path, f"{safe_name}_Acor_yy.txt"), Acor[:, :, 2], fmt="%.6f")

    # Affichage des canaux
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    canaux = ["Canal α (XX)", "Canal β (XY)", "Canal γ (YY)"]

    for i in range(3):
        ax = axes[i]
        im = ax.imshow(Acor[:, :, i], cmap=cmap)
        subtitle = f"Name : {save}\n{titre}" if isinstance(save, str) else titre
        ax.set_title(f"{subtitle} - {canaux[i]}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if isinstance(save, str):
        plt.savefig(os.path.join(img_path, f"{safe_name}_acor_canaux.png"), dpi=300)

def affiche_delta_lambda(delta_matrix, lambda_matrix, titre="Matrices Delta et Lambda", cmap_delta="viridis", cmap_lambda="plasma", save=0):
    """
    Affiche la matrice Delta et les deux canaux de la matrice Lambda avec des colormaps différentes.
    Si 'save' est une chaîne, les données et la figure sont enregistrées.

    Parameters:
        delta_matrix (np.ndarray): Matrice Delta (2D).
        lambda_matrix (np.ndarray): Matrice Lambda (3D), taille (NbLg, NbCol, 2).
        titre (str): Titre principal des figures.
        cmap_delta (str): Colormap pour la matrice Delta.
        cmap_lambda (str): Colormap pour les matrices Lambda.
        save (str|int): Si chaîne, active la sauvegarde des données + figure.
    """
    if lambda_matrix.shape[-1] != 2:
        raise ValueError("La matrice Lambda doit avoir exactement 2 canaux (dernier axe de taille 2).")

    if isinstance(save, str):
        # Création des répertoires de sauvegarde
        data_path = os.path.join("file_saved", "data_generate")
        img_path = os.path.join("file_saved", "laste_show")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

        # Nom de fichier sécurisé
        safe_name = save.replace(" ", "_").replace(":", "-")

        # Sauvegarde des matrices
        np.savetxt(os.path.join(data_path, f"{safe_name}_Delta.txt"), delta_matrix, fmt="%.6f")
        np.savetxt(os.path.join(data_path, f"{safe_name}_Lambda_plus.txt"), lambda_matrix[:, :, 0], fmt="%.6f")
        np.savetxt(os.path.join(data_path, f"{safe_name}_Lambda_minus.txt"), lambda_matrix[:, :, 1], fmt="%.6f")

    # Préparation des affichages
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    matrices = [delta_matrix, lambda_matrix[:, :, 0], lambda_matrix[:, :, 1]]
    noms = ["Déterminant : Δ", "Valeur Propre : λ+", "Valeur Propre : λ-"]
    colormaps = [cmap_delta, cmap_lambda, cmap_lambda]

    for i, (mat, nom, cmap) in enumerate(zip(matrices, noms, colormaps)):
        ax = axes[i]
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(nom, fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Titre global sur 2 lignes si 'save' est une chaîne
    if isinstance(save, str):
        full_title = f"Name : {save}\n{titre}"
    else:
        full_title = titre

    plt.suptitle(full_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if isinstance(save, str):
        plt.savefig(os.path.join(img_path, f"{safe_name}_delta_lambda.png"), dpi=300)

def affiche_rayon(rayon, titre="Rayon de courbure", cmap="magma", save=0):
    """
    Affiche la matrice du rayon de courbure sous forme de heatmap.
    Les valeurs nulles (0) sont affichées en blanc.
    Si 'save' est une chaîne, les données et la figure sont enregistrées.

    Args:
        rayon (np.ndarray): Matrice à afficher.
        titre (str): Titre de la heatmap.
        cmap (str): Colormap à utiliser.
        save (str|int): Si chaîne, active la sauvegarde des données + image.
    """
    # Masquer les valeurs nulles
    masked = np.ma.masked_where(rayon == 0, rayon)

    # Adapter la colormap avec 0 en blanc
    base_cmap = plt.get_cmap(cmap)
    new_cmap = base_cmap(np.linspace(0, 1, 256))
    new_cmap[0] = [1, 1, 1, 1]
    final_cmap = ListedColormap(new_cmap)

    # Crée les répertoires si nécessaire
    if isinstance(save, str):
        data_path = os.path.join("file_saved", "data_generate")
        img_path = os.path.join("file_saved", "laste_show")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)

        # Nom de fichier sécurisé
        safe_name = f"{save}_{titre}".replace(" ", "_").replace(":", "-")
        np.savetxt(os.path.join(data_path, f"{safe_name}.txt"), rayon, fmt="%.6f")

    # Affichage
    plt.figure(figsize=(8, 6))
    im = plt.imshow(masked, cmap=final_cmap)
    plt.colorbar(im)

    if isinstance(save, str):
        full_title = f"Name : {save}\n{titre}"
    else:
        full_title = titre

    plt.title(full_title, fontsize=14)
    plt.tight_layout()

    # Sauvegarde de la figure si nécessaire
    if isinstance(save, str):
        plt.savefig(os.path.join(img_path, f"{safe_name}.png"), dpi=300)

def affiche_segment(af,data,save):
    i=0
    if af[0] == 1 :
        afficher_image(data[i],"raw","gray",save)
        i = i+1
    if af[1] == 1:
        afficher_image(data[i],"pic","gray",save)
        i = i+1
    if af[2] == 1:
        affiche_filtres(data[i],data[i+1],'Filtre H','Filtre V','hot',save)
        i = i+2
    if af[3] == 1:
        affiche_filtres(data[i],data[i+1],'Dx','Dy','inferno',save)
        i = i+2
    if af[4] == 1:
        affiche_acor(data[i],'Matrice Acor','inferno',save)
        i = i+1
    if af[5] == 1:
        affiche_delta_lambda(data[i],data[i+1] ,'Matrice delta et lambda','plasma','magma',save)
        i = i+2
    if af[6] == 1:
        affiche_chatoux(data[i],"Norme ( Chatoux ) ","coolwarm",save)
        i = i+1
    if af[7] == 1:
        affiche_chatoux(data[i], "Contours ", "Purples",save)
        i = i+1

def affiche_segment_evoleved(af,data,save):
    i=0
    if af[0] == 1 :
        affiche_chatoux(data[i],"normes du gradient ","viridis",save)
        i = i+1
    if af[1] == 1:
        affiche_chatoux(data[i],"argument du gradient ", "viridis",save)
        i = i+1
    if af[2] == 1:
        affiche_chatoux(data[i],"argument du gradient (Absolue) ","magma",save)
        i = i+1
    if af[3] == 1:
        affiche_rayon(data[i], "Contours ", "cool",save)
        i = i+1
    if af[4] == 1:
        affiche_rayon(data[i], "Contours confiances ", "cool",save)
        i = i+1

def afficher_derive_stack(pile1, pile2, titre1='Pile 1', titre2='Pile 2'):
    """
    Affiche deux piles d'images 2D synchronisées avec un curseur pour naviguer entre les couches.

    Paramètres :
        pile1 (numpy.ndarray) : Première pile d'images (H, W, N)
        pile2 (numpy.ndarray) : Deuxième pile d'images (H, W, N)
        titre1 (str) : Titre de la première pile
        titre2 (str) : Titre de la deuxième pile
    """
    assert pile1.shape == pile2.shape, "Les deux piles doivent avoir les mêmes dimensions."

    nb_images = pile1.shape[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.15)

    img1 = ax1.imshow(pile1[:, :, 0], cmap='gray')
    ax1.set_title(f'{titre1} - Image 0')
    ax1.axis('off')

    img2 = ax2.imshow(pile2[:, :, 0], cmap='gray')
    ax2.set_title(f'{titre2} - Image 0')
    ax2.axis('off')

    # Curseur
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Index', 0, nb_images - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        img1.set_data(pile1[:, :, idx])
        ax1.set_title(f'{titre1} - Image {idx}')
        img2.set_data(pile2[:, :, idx])
        ax2.set_title(f'{titre2} - Image {idx}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def plot_radial(angles, min_distances, max_distances, mode='polar', number='both',label1='Distance min',label2='Distance max'):
    """
    Affiche le profil radial en mode polaire ou linéaire.

    Args:
        angles (np.ndarray): angles en degrés
        min_distances (np.ndarray): distances minimales (1er pixel actif)
        max_distances (np.ndarray): distances maximales (dernier pixel actif)
        mode (str): 'polar' ou 'linear'
        number (str): 'min', 'max', ou 'both'
    """
    if number not in ['min', 'max', 'both']:
        raise ValueError("number doit être 'min', 'max' ou 'both'")

    if mode == 'polar':
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, projection='polar')
        if number in ['min', 'both']:
            ax.plot(np.deg2rad(angles), min_distances, label=label1)
        if number in ['max', 'both']:
            ax.plot(np.deg2rad(angles), max_distances, label=label2)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_title("Profil radial (coordonnées polaires)")
        ax.legend()
    elif mode == 'linear':
        plt.figure(figsize=(8, 6))
        if number in ['min', 'both']:
            plt.plot(angles, min_distances, label=label1)
        if number in ['max', 'both']:
            plt.plot(angles, max_distances, label=label2)
        plt.xlabel("Angle (°)")
        plt.ylabel("Distance au contour (px)")
        plt.title("Profil radial (affichage linéaire)")
        plt.grid(True)
        plt.legend()
    else:
        raise ValueError("mode doit être 'polar' ou 'linear'")

def plot_radial_stack(angles, min_distances, max_distances, which='both'):
    """
    Affiche un graphe 3D des distances min et/ou max selon les angles et les coupes.

    Args:
        angles (np.ndarray): angles en degrés
        min_distances (np.ndarray): (N, num_angles) premier pixel à 1
        max_distances (np.ndarray): (N, num_angles) dernier pixel à 1
        which (str): 'min', 'max', ou 'both' pour choisir l'affichage
    """
    n_slices, num_angles = min_distances.shape
    A, S = np.meshgrid(angles, np.arange(n_slices))  # (N, num_angles)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    if which in ('min', 'both'):
        ax.plot_surface(A, S, min_distances, cmap='Blues', alpha=0.7, edgecolor='none', label='min')
    if which in ('max', 'both'):
        ax.plot_surface(A, S, max_distances, cmap='Oranges', alpha=0.7, edgecolor='none', label='max')

    ax.set_xlabel('Angle (°)')
    ax.set_ylabel('Index de coupe')
    ax.set_zlabel('Distance au contour (px)')
    ax.set_title('Profil radial : distances min et max')
    plt.tight_layout()

#################################################################################################################################################################################
## ----------------------------------------[ Zone d'import ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################
import os
import pyvista as pv
import math
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio 
import math
import Lib_outils as lo

from scipy.ndimage import affine_transform,rotate 
from sklearn.decomposition import PCA
from scipy.optimize import leastsq
from skimage.draw import disk
from skimage.draw import circle_perimeter
from matplotlib.widgets import Slider
from skimage.io import imread
from mpl_toolkits.mplot3d import Axes3D  
from scipy.ndimage import gaussian_filter1d

from numpy.polynomial.polynomial import Polynomial





#################################################################################################################################################################################
## ----------------------------------------[ Zone de déclaration ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################




#################################################################################################################################################################################
## ----------------------------------------[ Zone de fonction ]---------------------------------------------------------------------------------------------------------------------
#################################################################################################################################################################################

## Générale 


## profil radiale 

def radial_profile(image, center, resolution=1):
    """
    Calcule les distances radiales minimale et maximale entre le centre
    et les pixels actifs (valant 1) pour chaque angle entre 0 et 360°.

    Args:
        image (np.ndarray): image binaire 2D
        center (tuple): coordonnées (y, x) du centre
        resolution (int): pas angulaire en degrés

    Returns:
        angles (np.ndarray): angles en degrés
        min_distances (np.ndarray): distance au premier pixel rencontré
        max_distances (np.ndarray): distance au dernier pixel rencontré
    """
    h, w = image.shape
    y0, x0 = center
    angles = np.arange(0, 360, resolution)
    min_distances = np.full_like(angles, np.nan, dtype=float)
    max_distances = np.full_like(angles, np.nan, dtype=float)

    max_radius = int(np.hypot(h, w))

    for i, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        dx = np.cos(theta)
        dy = np.sin(theta)

        first_found = False
        last_distance = np.nan

        for r in range(max_radius):
            x = int(round(x0 + r * dx))
            y = int(round(y0 + r * dy))

            if 0 <= x < w and 0 <= y < h:
                if image[y, x] == 1:
                    if not first_found:
                        min_distances[i] = r
                        first_found = True
                    last_distance = r

        if first_found:
            max_distances[i] = last_distance

    return angles, min_distances, max_distances

def radial_profile_stack(aligned_stack, center_coords, resolution=1):
    """
    Calcule les distances min (premier pixel à 1) et max (dernier pixel à 1)
    pour chaque angle et chaque coupe dans la pile.

    Args:
        aligned_stack (np.ndarray): pile réalignée (H, W, N)
        center_coords (tuple): (x, y) du centre
        resolution (int): pas angulaire en degrés

    Returns:
        angles (np.ndarray): angles en degrés (0 à 360)
        min_distances (np.ndarray): (N, num_angles) distances au premier pixel à 1
        max_distances (np.ndarray): (N, num_angles) distances au dernier pixel à 1
    """
    h, w, n = aligned_stack.shape
    x0, y0 = center_coords
    angles = np.arange(0, 360, resolution)
    num_angles = len(angles)

    min_distances = np.full((n, num_angles), np.nan)
    max_distances = np.full((n, num_angles), np.nan)

    max_radius = int(np.hypot(h, w))

    for idx in range(n):
        image = aligned_stack[:, :, idx]
        for i, angle in enumerate(angles):
            theta = np.deg2rad(angle)
            dx = np.cos(theta)
            dy = np.sin(theta)

            found = False
            for r in range(max_radius):
                x = int(round(x0 + r * dx))
                y = int(round(y0 + r * dy))
                if 0 <= x < w and 0 <= y < h:
                    if image[y, x] == 1:
                        if not found:
                            min_distances[idx, i] = r
                            found = True
                        max_distances[idx, i] = r

    return angles, min_distances, max_distances


## lissage 

def smooth_rad(image, center, distances, angles, sigma=2):
    """
    Lisse un profil radial et met à jour l'image en traçant le contour lissé.

    Args:
        image (np.ndarray): image binaire 2D
        center (tuple): coordonnées (y, x) du centre
        distances (np.ndarray): distances radiales (ex: min_distances ou max_distances)
        angles (np.ndarray): angles correspondants en degrés
        sigma (float): écart-type du filtre gaussien pour le lissage

    Returns:
        updated_image (np.ndarray): nouvelle image avec contour lissé
        smoothed_distances (np.ndarray): profil radial lissé
    """
    h, w = image.shape
    y0, x0 = center

    # Lissage du profil radial
    smoothed = gaussian_filter1d(distances, sigma=sigma)

    # Création d’une nouvelle image
    updated_image = np.copy(image)

    for r, angle in zip(smoothed, angles):
        if np.isnan(r):
            continue  # Ignore cette direction

        theta = np.deg2rad(angle)
        x = int(round(x0 + r * np.cos(theta)))
        y = int(round(y0 + r * np.sin(theta)))

        if 0 <= x < w and 0 <= y < h:
            updated_image[y, x] = 1


    return updated_image, smoothed

def approximate_radial_profile_poly(image, center, distances, angles, degree=20):
    """
    Approximé un profil radial par une courbe polynomiale en fonction de l'angle.

    Args:
        image (np.ndarray): image binaire 2D
        center (tuple): coordonnées (y, x) du centre (format (y, x))
        distances (np.ndarray): distances radiales à approximer
        angles (np.ndarray): angles en degrés
        degree (int): degré du polynôme pour l'ajustement

    Returns:
        updated_image (np.ndarray): image avec le contour reconstruit
        smoothed_distances (np.ndarray): distances reconstruites par le polynôme
    """
    h, w = image.shape
    y0, x0 = center

    # Filtrage des NaN
    valid = ~np.isnan(distances)
    angles_valid = angles[valid]
    distances_valid = distances[valid]

    # Conversion des angles en radians pour l'ajustement
    angles_rad = np.deg2rad(angles_valid)

    # Ajustement polynômial sur les angles en radians
    coeffs = np.polyfit(angles_rad, distances_valid, deg=degree)
    poly = np.poly1d(coeffs)

    # Reconstruction du profil complet
    angles_full_rad = np.deg2rad(angles)
    smoothed = poly(angles_full_rad)

    # Création d'une nouvelle image avec le contour reconstruit
    updated_image = np.copy(image)

    for r, theta in zip(smoothed, angles_full_rad):
        if np.isnan(r):
            continue
        x = int(round(x0 + r * np.cos(theta)))
        y = int(round(y0 + r * np.sin(theta)))
        if 0 <= x < w and 0 <= y < h:
            updated_image[y, x] = 1

    return updated_image, smoothed

# Rayon moyen du voisinnage 
def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    try:
        center, _ = leastsq(f, center_estimate, maxfev=1000)
        Ri = calc_R(*center)
        radius = Ri.mean()

        if np.isnan(radius) or radius <= 0 or radius > 1e3:
            return center, 0
        return center, radius
    except Exception:
        return (0, 0), 0

def approx_radius_map(img_bin, r_voisin = 10 ):
    h, w = img_bin.shape
    rayon_mat = np.zeros_like(img_bin, dtype=np.float32)
    total_points = 0
    tries = 0

    ys, xs = np.where(img_bin == 1)
    for i, j in zip(ys, xs):
            rr, cc = disk((i, j), r_voisin, shape=img_bin.shape)
            voisins = img_bin[rr, cc]
            pts_1 = np.array([[x, y] for x, y, val in zip(rr, cc, voisins) if val == 1])
            tries += 1
            total_points += len(pts_1)
            if len(pts_1) >= 3:
                x = pts_1[:, 1]
                y = pts_1[:, 0]
                try:
                    _, rayon = fit_circle(x, y)
                    rayon_mat[i, j] = rayon
                except Exception:
                        pass  # Erreur d'ajustement
                if np.isnan(rayon) or rayon > 1e3 or rayon < 0:
                    rayon = 0  # ou continue
                    rayon_mat[i, j] = rayon
    print("Nombre de pixels à 1 testés :", tries)
    print("Nombre total de points à 1 dans les voisinages :", total_points)
    return rayon_mat

# rayon du voisinnage avec condition d'appartenance.

def approx_radius_map_V2(img_bin, r_voisin=10):
    h, w = img_bin.shape
    rayon_mat = np.zeros_like(img_bin, dtype=np.float32)
    total_points = 0
    tries = 0

    ys, xs = np.where(img_bin == 1)
    for i, j in zip(ys, xs):
        rr, cc = disk((i, j), r_voisin, shape=img_bin.shape)
        voisins = img_bin[rr, cc]
        pts_1 = np.array([[x, y] for x, y, val in zip(rr, cc, voisins) if val == 1])
        tries += 1
        total_points += len(pts_1)

        if len(pts_1) >= 3:
            x = pts_1[:, 1]
            y = pts_1[:, 0]
            _, rayon = fit_circle(x, y)

            if not np.isnan(rayon) and 0 < rayon <= 1e3:
                # Vérification : tolérance de 50% sur le cercle
                yc, xc = int(round(np.mean(y))), int(round(np.mean(x)))
                rr_circ, cc_circ = circle_perimeter(yc, xc, int(round(rayon)))

                # Garder uniquement les coordonnées valides dans l'image
                valid = (rr_circ >= 0) & (rr_circ < h) & (cc_circ >= 0) & (cc_circ < w)
                rr_circ = rr_circ[valid]
                cc_circ = cc_circ[valid]

                nb_total = len(rr_circ)
                nb_1 = np.sum(img_bin[rr_circ, cc_circ] == 1)

                if nb_total > 0 and nb_1 / nb_total >= 0.5:
                    rayon_mat[i, j] = rayon
                else:
                    rayon_mat[i, j] = 0  # moins de 50 % des points sur des 1
            else:
                rayon_mat[i, j] = 0  # rayon invalide
    print("Nombre de pixels à 1 testés :", tries)
    print("Nombre total de points à 1 dans les voisinages :", total_points)
    return rayon_mat

# rayon avec condition d'appartenance ET optimisation position

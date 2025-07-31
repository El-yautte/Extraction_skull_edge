# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 23:40:19 2022

    File   : gradients_2D_image.py
    Date   : 07/01/2023
    Status : Ok 


    Library file for gradient image processing and validation (main part)
    
    Included functions :
        gaussienne(): generate a 1D gaussian function
        derivee_gaussienne() : generate a 1D derivative version of a gaussian function
        NR_gradient_image()  : process the gradient image of an intensity image
        
    Validation part :
        Create the 2D mask filter, apply it on a png file (defined from a set),
        PLot the parial derivative part (horizontal and vertical), the norm and
        argument, the histograms of norm and arguments, the gradients norm and 
        argument for pixel having a gradient norm greater that a threshold.

@author: nrichard
"""
###########################################################################
###    Zone 1 : zone d'import
###########################################################################

import numpy   as np
import imageio.v2 as io
import matplotlib.pyplot as plt


###########################################################################
###    Zone 2 : Constantes globales
###########################################################################

Repertory = "../../images/part_2_a/"
FileName = ["food_tea_03_tension_rgb", "stone_01_rgb", "textile_14_cyan_rgb",
            "vegetation_leaf_03_amber_rgb"]

Selected_file = 0
sigma = 5.0     # scale parameter = filter_size
gradient_threshold = 4.0  #4.0 for tea, 2.0 for stone, 0.8 for textile, 4.0 for leaf)

###########################################################################
###    Zone 3 : Functions définition
###########################################################################
def gaussienne(time, mu, sigma):
    gauss =  np.exp( - (time-mu)*(time-mu)/(2*sigma*sigma) )  \
                / (sigma * np.sqrt(2* np.pi))               
    return gauss
##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------            
def derivee_gaussienne(time, mu, sigma):
    der_gauss = - (time - mu)  \
                *  np.exp( - (time-mu)*(time-mu)/(2*sigma*sigma) )  \
                / (sigma**3 * np.sqrt(2* np.pi))
    return der_gauss

##-----------------------------------------------------------------------------
def NR_gradient_image(img, sigma):
    '''
    return a 3D numpy array embedding as layer 1 the gradient related to the 
    vertical axis and as layer 2 the gradient related to the horizontal axis

    Parameters
    ----------
    img : 2d numpy array
        the intensity image.
    sigma : numpy float
        scale of the gradient processing.

    Returns
    -------
    gradient_img : 3D numpy array of floats.
        (nb_lg, nb_col, 2) image
    '''
    gradient = np.zeros( (img.shape[0], img.shape[1], 2) )
    ## Step 1.0 : create the filter mask in the spatial domain
    time = np.arange(0.0, 6*sigma+1)
    part1 = gaussienne(time, time.max()/2, sigma)  
    part2 = derivee_gaussienne(time, time.max()/2, sigma)   
    
    part1 = part1.reshape(1, part1.size)
    part2 = part2.reshape(1, part2.size)
    filtre_v = part1 * part2.T
        
    ## Step 2.0 : process the filtering in the Fourier domain
    ## 0 : related to the vertical axis, 1 : related to the horizontal axis)
    gradient[ : , :, 0] = np.real(np.fft.ifft2(np.fft.fft2(img)*np.fft.fft2(filtre_v  , s=img.shape)))
    gradient[ : , :, 1] = np.real(np.fft.ifft2(np.fft.fft2(img)*np.fft.fft2(filtre_v.T, s=img.shape)))

    return gradient
##-----------------------------------------------------------------------------


###########################################################################
###########################################################################
###                                                                     ###
###    Zone 4 : Main Part fo the application
###                                                                     ###
###########################################################################
###########################################################################
if __name__ == "__main__":
    plt.close('all')

    ###########################################################################
    ###    Step 1 : Multi-file processing preparation and Constants initialization
    ###########################################################################
    
    ##  Read the firsy image to keep the image parameters    
    img = io.imread(Repertory + FileName[Selected_file ] + '.png')
    nb_lg, nb_col, nb_canaux = img.shape
    
    nb_img = len(FileName)
 
    
    image = img[:,:,1]  ## select the green channel for the test
    plt.figure()
    plt.imshow(image, cmap='gist_gray') ## affichage du canal vert
    plt.title("1.0 : Image : " + FileName[Selected_file])
    plt.colorbar()
    plt.show()
    
    ###########################################################################
    ###    Step 2 : Show the filter constituting parts
    ###########################################################################
    ## taille du filtre : (6*int(sigma)+1, 6*int(sigma)+1) 
    
    time = np.arange(0.0, 6*sigma+1)
    part1 = gaussienne(time, time.max()/2, sigma)  
    part2 = derivee_gaussienne(time, time.max()/2, sigma)
    
    plt.figure()
    plt.plot(time, part1, label="gauss")
    plt.plot(time, part2, label="der gauss")
    plt.legend()
    plt.title("2.0 : Gaussian and derivative Gaussian filter")
    plt.show()
    
    
    part1 = part1.reshape(1, part1.size)
    part2 = part2.reshape(1, part2.size)
    filtre_v = part1 *part2.T
    
    fig, tab_ax = plt.subplots(1,2)
    plt.axes(tab_ax[0])
    plt.imshow(filtre_v)
    plt.colorbar()
    plt.title("Filter pattern for vertical derivation")
    
    plt.axes(tab_ax[1])
    plt.imshow(filtre_v.T)
    plt.colorbar()
    plt.title("Filter pattern for horizontal derivation")
    plt.suptitle( "2.1 : derivative filters (sigma = " + str(sigma) + ")")
    plt.show()
    
    ###########################################################################
    ###    Step 3 : Filtering, Norm and argument
    ###########################################################################
    
    gradient_img = NR_gradient_image(image, sigma)
    
    fig, tab_ax = plt.subplots(1,2)
    plt.axes(tab_ax[0])
    plt.imshow(gradient_img[:,:,0], cmap='PiYG')
    plt.colorbar()
    plt.title("Derivation according to the vertical axis")

    plt.axes(tab_ax[1])
    plt.imshow(gradient_img[:,:,1], cmap='PiYG')
    plt.colorbar()
    plt.title("Derivation according to the horizontal axis")
    plt.suptitle("3.0 : directionnal derivative parts of the image "+ FileName[Selected_file ] )
    plt.show()
    
    gradient_norm  = np.sqrt(np.sum( gradient_img*gradient_img, axis=2))
    gradient_angle = np.arctan( gradient_img[ :, :, 1] / gradient_img[: , :, 0])
    
    fig, tab_ax = plt.subplots(1,2)
    plt.axes(tab_ax[0])
    plt.imshow(gradient_norm, cmap='gist_gray') ## norm = intensity so grey map
    plt.colorbar()
    plt.title("Gradient norm")

    plt.axes(tab_ax[1])
    plt.imshow(gradient_angle, cmap='hsv') ## angle = colormap cyclic
    plt.colorbar()
    plt.title("Gradient angle")
    plt.suptitle("Gradient norm and angle (sigma = " + str(sigma) + ") of "+ FileName[Selected_file ])
    plt.show()

    ###########################################################################
    ###    Step 3 : Where are the edges
    ###########################################################################
    ## 3.1 : statistical analysis of the gradient norm
    fig, tab_ax = plt.subplots(1,2)
    plt.axes(tab_ax[0])
    plt.hist(gradient_norm.reshape( (nb_lg*nb_col)), bins='sqrt', range= (0,12.0))
    plt.title("Histogram of the gradient norms")
    
    plt.axes(tab_ax[1])
    plt.hist(gradient_angle.reshape( (nb_lg*nb_col)), bins='sqrt', range= (-np.pi/2.0, np.pi/2.0))
    plt.title("Histogram of the gradient angle")
    
    plt.suptitle("Statistical distribution of the gradient norms and angles of "+ FileName[Selected_file ])
    plt.show()

    ## 3.2 : keep only the contour part (texture pattern information)
    contour     = np.where( gradient_norm > gradient_threshold , 10.0, 0.0)
    orientation = np.where( contour > 0, gradient_angle, np.NAN)
    
    ## 3.3 : plot the contour informations
    fig, tab_ax = plt.subplots(1,2)
    plt.axes(tab_ax[0])
    plt.imshow(contour, cmap='gist_gray') ## norm = intensity so grey map
    plt.colorbar()
    plt.title("Extracted border information")
    
    plt.axes(tab_ax[1])
    plt.imshow(orientation, cmap='hsv') ## cyclic map : angle
    plt.colorbar()
    plt.title("Extracted angle of the contours")
    plt.suptitle("Contours with magnitude > " + str(gradient_threshold) + "of " + FileName[Selected_file ])
    plt.show()
    
    plt.figure()
    plt.hist(orientation.reshape( (nb_lg*nb_col)), bins='sqrt', range= (-np.pi/2.0, np.pi/2.0))
    plt.title("Angle distribution on the contours with magnitude > " + str(gradient_threshold) + "of " + FileName[Selected_file ])
    plt.show()
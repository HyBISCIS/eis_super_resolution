from pickle import FALSE, TRUE
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib import image
from skimage.transform import resize,rescale, rotate
from skimage.filters import gaussian, unsharp_mask
from skimage.morphology import dilation,erosion,selem
import scipy.signal as sig
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1
import time

import deconv_func as func

'''
This is the script that actually runs through and performs the super-resolution
composite images at different distances for the figure that shows different 
distances of composite image composition.
'''


# ===================   Settings     ===================

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16         # FIXME: DO NOT CHANGE. WINDOW IS HARD CODED RIGHT NOW
FREQ = 3125                 #6250  # in kHz
#FREQ = 6250
SAVE_IMAGES = True
LOW_SALT = True
INTERPOLATE_ORDER = 2

# Cropping Parameters (In order of Importance)
CROP = True
RAD = 8                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO

#single_lobe = (2944,3287)       # Isolated Single Lobe (I think (1,1) will be better, but right now, (0,1))
#single_lobe = (2925,3278)   # This is the first isolated lobe, but with (1,1) instead so slightly different
#single_lobe = (6258, 2652)      # (1,0) offset center
single_lobe = (6824, 1317)      # TEST  offset (1,0)
# single_lobe = (6897, 1949)
# single_lobe = (6837, 1324)
# single_lobe = (6459, 2352)


# Three okay ones....
two_lobe = (4413, 1830)         # First Good One for Isolated Double Lobe (-2,0)
# two_lobe = (1359, 791)      # TEST (OKAY)   (-2,-1)
#two_lobe = (4726, 1630)    # TEST (-3, -1) center offset

NUM_LOBES = 2

if NUM_LOBES == 2:
    INTEREST_POINT = two_lobe
else:
    INTEREST_POINT = single_lobe


# Row and Column Offset Names
if (CHIP_NAME == "LILLIPUT"):
    im_size = (100,100)
    row = "R"
    col = "C"
    f_name = "sw_freq"
    FREQ = FREQ
    im = 'image'

    logdir = r"../data/bead_20um"
    logfile = r"phase2_sweep_11x11_block.h5"
    microscope_img = 'Microscope_bead_20u_0639pm.bmp'
    CENTER = (5,5)
    # Note: Freq in kHz
else:
    im_size = (512,256)
    row = "row_offset"
    col = "col_offset"
    f_name = "f_sw"
    FREQ = FREQ * 1000
    im = 'image_2d_ph2'
    # Note: Freq in Hz

    logdir = r"../data/super_res"
    logfile = r"ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_set_1.h5"

    microscope_img = None
    CENTER = (0,0)
    
    if (LOW_SALT):
        logfile = r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_1.h5"
        #logfile = r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_VCM_500_VSTBY_300_set_2.h5"

# ======================================================





# ------------------------ Beginning of Script -----------------------------

# Initialize Composite Images
compositeimage3 = None
compositeimage5 = None
compositeimage7 = None
compositeimage9 = None
compositeimage11 = None

count = 0

# Load data and sort keys
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

# Creating Window for Deconvolution Superresolution Technique
if CROP:
    window_scale = 0.5
    window_length = int(CROPPED_LENGTH * window_scale)
    pad_length = int(CROPPED_LENGTH * ((1 - window_scale) / 2))
    window1d = np.abs(np.hanning(window_length))
    window2d = np.sqrt(np.outer(window1d, window1d))
    window2d = np.pad(window2d, pad_length)
else:
    window2d = func.create_window(CHIP_NAME, BLOCK_SIZE[0], UPSAMPLE_RATIO)

print("Cropped Size:", CROPPED_LENGTH)
print("Window Size:", window2d.shape)

# Obtain Reference Images
if NUM_LOBES == 2:
    CENTER = (CENTER[0]-2, CENTER[1])      # FIRST GOOD ONE
    #CENTER = (CENTER[0], CENTER[1]+1)
else:
    CENTER = (CENTER[0]+1, CENTER[1])


k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0] and int(mydata[x].attrs[col])==CENTER[1]][0]
reference_image,ref_shifted = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, LOW_SALT, INTERPOLATE_ORDER)

if CROP:
    reference_image = func.get_area_around(reference_image, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

np.save("../log/composite_image_1.npy", reference_image)

outputimage = reference_image - gaussian(reference_image,sigma=10*UPSAMPLE_RATIO)
output_f = np.fft.rfft2(outputimage)

# Iterate through image offsets
for i in sortedkeys:
    start = time.time() 
    # Obtain Row and Col Offsets of Particular Image
    myrow = int(mydata[i].attrs[row])
    mycol = int(mydata[i].attrs[col])

    dist = np.sqrt(myrow**2 + mycol**2)

    if (CHIP_NAME == "MINERVA"):
        # Dirty Fix for Minerva indexing from -5 to 5, rather than the 0 to 11 we expected previously.
        myrow += 5
        mycol += 5

    if (myrow == 5 and mycol == 5):
        # Skip middle as it is not useful to us and just adds noise
        continue
    
    myimage,myimage_shift = func.getimage(mydata,i,UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME,LOW_SALT,INTERPOLATE_ORDER,shiftrow=myrow,shiftcol=mycol)

    if CROP:
        myimage = func.get_area_around(myimage, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

    # Perform linear filter
    myimage_filtered,kernel_smoothed = func.linear_filter(myimage, output_f, UPSAMPLE_RATIO, window2d)

    # If initial add up
    if compositeimage3 is None:
        compositeimage3 = np.zeros_like(myimage)

    if compositeimage5 is None:
        compositeimage5 = np.zeros_like(myimage)

    if compositeimage7 is None:
        compositeimage7 = np.zeros_like(myimage)

    if compositeimage9 is None:
        compositeimage9 = np.zeros_like(myimage)

    if compositeimage11 is None:
        compositeimage11 = np.zeros_like(myimage)
    
    # Add based off of distances

    if (dist <= np.sqrt(myrow**1 + mycol**1)):
        compositeimage3 += myimage_filtered 

    if (dist <= np.sqrt(myrow**2 + mycol**2)):
        compositeimage5 += myimage_filtered 

    if (dist <= np.sqrt(myrow**3 + mycol**3)):
        compositeimage7 += myimage_filtered 

    if (dist <= np.sqrt(myrow**4 + mycol**4)):
        compositeimage9 += myimage_filtered 

    compositeimage11 += myimage_filtered

    count += 1
    print("Count: ", count)

    end = time.time()
    print("Linear Filter Elapsed Time (sec):", end-start)


# ====================================================

# Save Images
if SAVE_IMAGES:
    np.save("../log/composite_image_3.npy", compositeimage3)

    np.save("../log/composite_image_5.npy", compositeimage5)

    np.save("../log/composite_image_7.npy", compositeimage7)

    np.save("../log/composite_image_9.npy", compositeimage9)

    np.save("../log/composite_image_11.npy", compositeimage11)

# ====================================================
# EIS Super-Resolution Reconstruction Algorithms

This repo contains code used for performing super-resolution on 
mutual capacitance images gathered from a regular pixel array of
impedance sensors.These have been noted below and each file is 
explained in further detail.

## Package Requirements
In order to use this repo, there are certain packages required in order
to make everything work correctly. They have been listed below:
- pickle
- h5py
- matplotlib
- numpy
- cv2
- skimage
- scipy

## Necessary Data
- Before use of this repo, it is necessary to create a data folder that 
  containins the h5 files that someone may want to run.

## Files
### Main Files for Linear Deconvolution
- deconv_superres.py is the main work horse script that all other super
  resolution scripts is based off of. It is for linear deconvolution
- shift_sum_superres.py is the shift-sum equivalent to the linear
  deconvolution method mentioned above.
- deconv_func.py includes all functions used for linear deconvolution and
  shift sum scripts.
- super_res_at_distances.py was the script used to generate the different
  distance composite images.

### Figure Scripts
- line_plots.py is the script used to make the line profiles through
  algae.
- make_distance_plots.py was the script used to make the different distance
  composite images. It was not used for the paper submission.
- make_microscope_cosmarium_pediastrum_plot.py was used to make the figure
  comparing the microscope, reference, and the impedance image.
- make_three_cosmarium_plots.py is the script used to make the three raw images and three super-resolution cosmarium figure.
- make_three_pediastrum_plots.py is the script used to make the three raw images and three super-resolution pediastrum figure.
- show_algae.py is a script used to show off the array of 120 impedance
images used in that big 11x11 array. This is for the figure for pediastrum and
for cosmarium
- mse_graphs.py is a script used to generate MSE plots from the images gathered in
  super_res_at_distances.py. It requires running super_res_at_distance.py to first
  gather the .npy files of the reconstructions.

### Helpful Scripts
- find_algae.py is a script used to find algae by comparing the optical
  microscope image against the full raw impedance image to look for
  certain algae to do super-resolution on.
- high_pass_filter.py is a script used to determine which parameters were
  best for the high pass filtering after linear-deconvolution method
- spatial_resolution.py is a script that was NOT FINISHED for our semiconductor sensor, but hopes to demonstrate the spatial-resolution superiority of the linear-deconvolution method over the shift sum method.
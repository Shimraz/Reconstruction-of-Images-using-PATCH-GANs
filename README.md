In this report, we propose deployment of the Generative Patch Prior (GPP) or Patch
GANs for carrying out the tasks of image deblurring, missing pixels recovery and
super-resolution. Patch GAN used is with pre-trained weights and the patches
being trained for wide variety of image types, appeal to be of good promise for
reconstructing images not limited to a specific domain. The pre-trained patch generator
generates patches that are merged together to obtain the final reconstructed
image. We implement a scheme of applying the measurement operation to the
output of generator for every iteration. The measurement operator here consists
of motion blur for the problem of deblurring, downsampling for super-resolution
and Boolean mask with specified percent of Boolean values for missing pixels
problem. This strategy is applied for the same image with a low resolution and a
high resolution, and their PSNR values are compared to determine the quality of
image obtained. The results of the experiments are not up to the mark and requires
further improvements.
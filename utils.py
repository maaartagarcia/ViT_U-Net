import tensorflow as tf
import numpy as np
import h5py, os, pdb

def conv_mask_gt(z):
   # Given a mask, returns two class tensor for every pixel
    z = tf.convert_to_tensor(z)

    background = tf.cast(( z == 0), dtype = 'float32')
    manipulated = tf.cast(( z == 1), dtype = 'float32')


    return [background, manipulated] 

def transform_masks(mask):
    result = np.zeros((*mask.shape, 2))
    result[:, :, 0] = 1 - mask
    result[:, :, 1] = mask
    return result

def transform_predictions(pred, threshold):
    m_pixels = pred[:, :, 1]
    return np.where(m_pixels >= threshold, 1, 0) 

def validate_input_images(hdf5_file, images_label, masks_label):
    if not os.path.exists(hdf5_file):
        print("Fail. ", hdf5_file, " doesn't exist.")
        return False
    
    f = h5py.File(hdf5_file, 'r')

    if images_label not in f or masks_label not in f:
        print("Fail. ", hdf5_file, " doesn't contain ", images_label, " and ", masks_label, " datasets.")
        return False

    images = f[images_label]
    masks = f[masks_label]

    # Validate if num of images and masks are the same
    if images.shape[0] != masks.shape[0]:
        print("Fail. Amount of images and masks don't match.")    
        return False

    # Validate that groundtruth only contain 0 and 1's
    unique_values = np.unique(masks)
    one_zero_array = np.array( [0,1] ).astype(np.float64)
    contains_only_one_zero = np.array_equal(unique_values, one_zero_array)

    if not contains_only_one_zero:
        print("Fail. Grountruth contain values different to 0's and 1's.")
        return False

    return True



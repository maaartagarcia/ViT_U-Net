import tensorflow as tf
import numpy as np

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
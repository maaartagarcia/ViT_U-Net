import keras
from keras import layers

import matplotlib.pyplot as plt

# Added
import pdb
import cv2
import tensorflow as tf
import numpy as np
import math

# Added
import h5py
import os
from utils import transform_masks, transform_predictions

def resUnit(input_layer, i, nbF):
    # Input Layer, number of layer, number of filters to be applied
    x = layers.BatchNormalization()(input_layer)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = nbF, kernel_size = kernel, activation = None, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = nbF, kernel_size = kernel, activation = None, padding = 'same')(x)

    return layers.add([input_layer, x])

nbFilter = 32 # Filter size  
kernel = (3, 3)
pool_kernel = (2, 2) # Max Pooling Kernel Size
# Down sample Kernel to convert Encoder and ViT's output to (16, 16, 32) before Decoder
# Down sample Kernel to convert Decoder to two class maps
down_kernel = (1,1) 
batch_size = 128
outSize = 16
upsampling_factor = (4, 4)
num_classes = 2
epochs = 500
num_imgs = None # Assigned when reading the data file 
input_shape = None
pt = 0.95 # Proportion of images used for training ( 1 - pt will be validation images)  
# Input shape: width, height, channels
w = 256
h = 256
c = 3

# DATA PREPARATION
# ------------------------------------------------------------------------------------------------------------------------------

# from keras.datasets import mnist
# from keras.datasets import cifar10

# (x_train, _), (x_test, _) = mnist.load_data()
# (x_train, _), (x_test, _) = cifar10.load_data()

# imgs_file = '/scratch.local2/al404273/m08/data/vit_data.hdf5'
# imgs_file = './hdf5/training_01.hdf5'
imgs_file = './hdf5/training_01.hdf5'
images_label = "train_img"
masks_label = "train_labels"

if not os.path.exists(imgs_file):
    print("Fail. ", imgs_file, " doesn't exist.")
    exit()

f = h5py.File(imgs_file, 'r')

if images_label not in f or masks_label not in f:
    print("Fail. ", imgs_file, " doesn't contain ", images_label, " and ", masks_label, " datasets.")
    exit()

X = f[images_label]
Y = f[masks_label]  

num_imgs = X.shape[0]
num_masks = Y.shape[0]  

if num_imgs != num_masks:
    print("Fail. Not the same amount of images an masks in hdf5 file.")
    exit()

train_p = int(math.floor(num_imgs * pt)) #  Train proportion 
test_p = int(math.floor(num_imgs * (1 - pt))) #  Validation proportion 

print("Total images: ", num_imgs)
print("Training images: ", train_p)
print("Testing images: ", test_p)

if train_p + test_p > num_imgs:
    print("Fail. Invalid training and validation proportions. Those values should match.")
    exit()

x_train, y_train = X[:train_p], Y[:train_p]
x_test, y_test = X[train_p:], Y[train_p:]    

xo_train, yo_train = x_train, y_train
xo_test, yo_test  = x_test, y_test 

# Normalized images only
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
# x_test =  np.reshape(x_test,  (len(x_test),  28, 28, 1))
# x_train = np.reshape(x_train, (len(x_train), input_shape[1] , input_shape[2], input_shape[3]))
# x_test =  np.reshape(x_test,  (len(x_test), input_shape[1] , input_shape[2], input_shape[3])

y_train_conv = np.zeros((train_p, 256, 256, 2))
y_test_conv = np.zeros((test_p, 256, 256, 2))

for i in range(train_p):
    y_train_conv[i] = transform_masks(y_train[i]) 
    
for i in range(test_p):
    y_test_conv[i] = transform_masks(y_test[i])

y_train = y_train_conv
y_test = y_test_conv

print("Shape Train Img: ", x_train.shape)
print("Shape Train masks: ", y_train.shape)
print("Shape Test Imgs: ", x_test.shape)
print("Shape Test Masks: ", y_test.shape)

input_shape = (num_imgs, w, h, c)

print("--> Model's input shape: ", input_shape)

# ------------------------------------------------------------------------------------------------------------------------------

# input_img = keras.Input(shape=(28, 28, 1))
input_img = keras.Input(shape=input_shape[1:] )

'''
# (Original)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
'''

# ENCODER
# ------------------------------------------------------------------------------------------------------------------------------

# layer 1 -> Input (256, 256, 3) // Output (128, 128, 32)
# (Before) layer1 = slim.conv2d( input_layer, nbFilter,[3,3], normalizer_fn = slim.batch_norm, scope = 'conv_' + str( 0 )  )
# (Before) layer1 = slim.conv2d( input_layer, nbFilter,[3,3], normalizer_fn = slim.batch_norm, scope = 'conv_' + str( 0 )  )
# (Original) x = layers.Conv2D(16, kernel, activation='relu', padding='same')(input_img)
x = layers.Conv2D(filters = nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(input_img)
x = layers.BatchNormalization()(x)
x = resUnit(x, 1, nbFilter)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# layer 2 -> Input (128, 128, 32) // Output (64, 64, 64)
x = layers.Conv2D(filters = 2 * nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = resUnit(x, 2, 2 * nbFilter)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# (Original) x = layers.Conv2D(4*nbFilter, kernel, activation='relu', padding='same')(x)
# (Original) encoded = layers.MaxPooling2D(pool_kernel, padding='same')(x)

# layer 3 -> Input (64, 64, 64) // Output (32, 32, 128)
x = layers.Conv2D(filters = 4 * nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = resUnit(x, 3, 4 * nbFilter)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# (Original) at this point the representation is (4, 4, 8) i.e. 128-dimensional

# layer 4 -> Input (32, 32, 128) // Output(16, 16, 256)
x = layers.Conv2D(filters = 8 * nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = resUnit(x, 4, 8 * nbFilter)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# Final representation (16, 16, 256)
# ------------------------------------------------------------------------------------------------------------------------------

# CONCATENATE ENCODER + ViT --> Input (16, 16, 32) + (16, 16, 32) // Output (16, 16, 64)
# ------------------------------------------------------------------------------------------------------------------------------

# ENCODER
# Layer from Encoder goes from (16, 16, 256) to (16, 16, 32)
# ViT layer's output should also have (16, 16, 32) shape

x = layers.Conv2D(filters = nbFilter, kernel_size = down_kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# (Before) top = slim.conv2d(layer4,nbFilter,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
# (Before) top = tf.nn.relu(top)
# (Before)concatenate both lstm features and image features
# (Before) joint_out=tf.concat([top,lstm_out],3)

# ViT
# ...

# CONCATENATE
# x = layers.concatenate([x,x], axis = 3)

# ------------------------------------------------------------------------------------------------------------------------------

# DECODER --> Input (16, 16, 64) // Output (256, 256, 2)
# ------------------------------------------------------------------------------------------------------------------------------

# Upsampling to change spatial resolution
# Convolution to change maps' depth

'''
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
'''

# The model uses UpSampling2D layer instead of Conv2DTranspose because the Decoder doesn't learn parameters in order to upsample the feature maps

# Upsampling to batch size (16, 64, 64, 64)
x = layers.UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(x)
# (16, 64, 64, 2)
# Conv2D kernel size modified from (1,1) to (3,3)
x = layers.Conv2D(filters = num_classes, kernel_size = kernel, activation = None, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Upsampling to batch size (16, 256, 256, 2)
x = layers.UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(x)
# Added
x = layers.Conv2D(filters = num_classes, kernel_size = kernel, activation = 'softmax', padding='same')(x)
# x = layers.BatchNormalization()(x)
# x = layers.ReLU()(x)

# ------------------------------------------------------------------------------------------------------------------------------

# SET THE MODEL CONFIGURATIONS
# ------------------------------------------------------------------------------------------------------------------------------

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, x)

# This is the size of our encoded representations
# encoding_dim = 128  # latent representation is (4, 4, 8) i.e. 128-dimensional

# Show the summary of the model architecture
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'] )
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------------------------------------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------------------------------------------------------

history = autoencoder.fit(x_train, y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(x_test, y_test))

# ------------------------------------------------------------------------------------------------------------------------------

# LEARNING CURVES
# ------------------------------------------------------------------------------------------------------------------------------

'''
plt.figure(2)
plt.plot(range(1,epochs+1), history.history['loss'])
plt.plot(range(1,epochs+1), history.history['val_loss'])
#plt.xticks(range(1,epochs+1))
plt.xlim(1,epochs)
plt.ylim(0, 0.25)
plt.show()
'''

# Encode and decode some digits
# Note that we take them from the *test* set

decoded_imgs = autoencoder.predict(x_test)

# Show some image digits examples and their reconstruction

# ------------------------------------------------------------------------------------------------------------------------------

# EXAMPLES RESULT
# ------------------------------------------------------------------------------------------------------------------------------

n = 10  # How many digits we will display

while n > decoded_imgs.shape[0]:
    print("Fail. Testing images cannot be greater than x_test.")
    response = input('New number of testing images? (number/no) ')
    if n == 'no':
        exit()
    n = int(response)

'''
result_file = './hdf5/validation_results.hdf5'
print("Important. Testing of ", n, " images (Original images, masks, and model predictions) will be saved in ", result_file, " file.")

if os.path.exists(result_file):
    print("Fail. ", result_file, " already exists. ")
    response = input("Want to delete? (s/n): ")
    if response == "s":
        os.remove(result_file)

f = h5py.File(result_file, 'w')
'''

# plt.figure(figsize=(20, 4))

# images = []
# predictions = []
# masks = []

for i in range(n):
    '''
    # Display original
    ax = plt.subplot(2, n, i + 1)
    # plt.imshow(x_test[i].reshape(28, 28))
    plt.imshow(x_test[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    # plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.imshow(decoded_imgs[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    '''

    # Added
    pdb.set_trace()
    pred = transform_predictions(decoded_imgs[i], 0.4)
    cv2.imwrite('./results/images/original_' + str(i) + '.jpg', x_test[i] * 255) 

    '''
    images.append(x_test[i] * 255)
    masks.append(yo_test[i] * 255)
    predictions.append(b_pred * 255)
    '''
    cv2.imwrite('./results/masks/mask_' + str(i) + '.png', yo_test[i]  * 255) 
    cv2.imwrite('./results/predictions/predicted_' + str(i) + '.png', pred * 255) 

'''
f.create_dataset("validation_img", shape = (n, w, h, c), data = images)
f.create_dataset("validation_labels", shape = (n, w, h), data = masks)
f.create_dataset("validation_pred", shape = (n, w, h), data = predictions)
'''

autoencoder.save('./model/trained_model.keras', overwrite = True)

# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------
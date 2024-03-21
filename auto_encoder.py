import keras
from keras import layers

import matplotlib.pyplot as plt

# Added
import pdb
import cv2
import tensorflow as tf
import numpy as np

# Added
import h5py
import os

def resUnit(input_layer, i, nbF):
    # Input Layer, number of layer, number of filters to be applied
    x = layers.BatchNormalization()(input_layer)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = nbF, kernel_size = kernel, activation = None, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = nbF, kernel_size = kernel, activation = None, padding = 'same')(x)

    return layers.add([input_layer, x])

def conv_mask_gt(z):
   # Given a mask, returns two class tensor for every pixel
    z = tf.convert_to_tensor(z)

    background = tf.cast(( z == 0), dtype = 'float32')
    manipulated = tf.cast(( z == 1), dtype = 'float32')


    return [background, manipulated] 

num_imgs = 600
input_shape = (num_imgs, 256, 256, 3)
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
epochs = 100

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
x = layers.concatenate([x,x], axis = 3)

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
x = layers.BatchNormalization()(x)
decoded = layers.ReLU()(x)

# ------------------------------------------------------------------------------------------------------------------------------

# SET THE MODEL CONFIGURATIONS
# ------------------------------------------------------------------------------------------------------------------------------

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This is the size of our encoded representations
# encoding_dim = 128  # latent representation is (4, 4, 8) i.e. 128-dimensional

# Show the summary of the model architecture
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'] )
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------------------------------------------------------------------------------------------------------------

# DATA PREPARATION
# ------------------------------------------------------------------------------------------------------------------------------

# from keras.datasets import mnist
# from keras.datasets import cifar10

# (x_train, _), (x_test, _) = mnist.load_data()
# (x_train, _), (x_test, _) = cifar10.load_data()

imgs_file = './training_01.hdf5'

if not os.path.exists(imgs_file):
    print("Fail. ", imgs_file, " doesn't exist.")
    exit()

f = h5py.File(imgs_file, 'r')

X = f["train_img"]
Y = f["train_labels"]  

x_train, y_train = X[:946], Y[:946]
x_test, y_test = X[946:], Y[946:]    

xo_train, yo_train = x_train, y_train
xo_test, yo_test  = x_test, y_test 

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
# x_test =  np.reshape(x_test,  (len(x_test),  28, 28, 1))
# x_train = np.reshape(x_train, (len(x_train), input_shape[1] , input_shape[2], input_shape[3]))
# x_test =  np.reshape(x_test,  (len(x_test), input_shape[1] , input_shape[2], input_shape[3]))

y_train_aux = []
y_test_aux = []  

for m in y_train:
    y_train_aux.append(conv_mask_gt(m))

for m in y_test:
    y_test_aux.append(conv_mask_gt(m))

y_train_conv = np.array(y_train_aux)
y_test_conv = np.array(y_test_aux)

y_train = np.transpose(y_train_aux, (0, 2, 3, 1))
y_test = np.transpose(y_test_aux, (0, 2, 3, 1))

print("Shape Train Img: ", x_train.shape)
print("Shape Train masks: ", y_train.shape)
print("Shape Test Imgs: ", x_test.shape)
print("Shape Test Masks: ", y_test.shape)

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

result_file = './validation_results.hdf5'
f = h5py.File(result_file, 'w')

n = 20  # How many digits we will display
# plt.figure(figsize=(20, 4))

images = []
predictions = []
masks = []

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
    argmax = np.argmax(decoded_imgs[i], axis = -1)
    b_pred = (argmax == 1)

    # cv2.imwrite('original_' + str(i) + '.jpg', x_test[i] * 255) 
    images.append(x_test[i] * 255)
    masks.append(yo_test[i] * 255)
    predictions.append(b_pred * 255)

    # cv2.imwrite('predicted_' + str(i) + '.png', b_pred * 255) 

f.create_dataset("validation_img", shape = (n, 256, 256, 3), data = images)
f.create_dataset("validation_labels", shape = (n, 256, 256), data = masks)
f.create_dataset("validation_pred", shape = (n, 256, 256), data = predictions)

autoencoder.save('./model/trained_model.keras', overwrite = False)

# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------
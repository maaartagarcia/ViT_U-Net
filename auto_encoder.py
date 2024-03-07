import keras
from keras import layers

import matplotlib.pyplot as plt
import pdb

input_shape = (256, 256, 3)
nbFilter = 32 # Filter size  
kernel = (3, 3)
max_kernel = (2, 2) # Max Pooling Kernel Size  

# input_img = keras.Input(shape=(28, 28, 1))
input_img = keras.Input(shape=input_shape)

'''
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
'''

x = layers.Conv2D(16, kernel, activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D(max_kernel, padding='same')(x)
x = layers.Conv2D(8, kernel, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(max_kernel, padding='same')(x)
x = layers.Conv2D(8, kernel, activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(max_kernel, padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
# Final representation (16, 16, ?)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This is the size of our encoded representations
# encoding_dim = 128  # latent representation is (4, 4, 8) i.e. 128-dimensional

# Show the summary of the model architecture
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test =  np.reshape(x_test,  (len(x_test),  28, 28, 1))

print(x_train.shape)
print(x_test.shape)

history = autoencoder.fit(x_train, x_train,
                          epochs=50,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, x_test))

# Learning curves
epochs = 50

plt.figure(2)
plt.plot(range(1,epochs+1), history.history['loss'])
plt.plot(range(1,epochs+1), history.history['val_loss'])
#plt.xticks(range(1,epochs+1))
plt.xlim(1,epochs)
plt.ylim(0, 0.25)
plt.show()

# Encode and decode some digits
# Note that we take them from the *test* set
decoded_imgs = autoencoder.predict(x_test)

# Show some image digits examples and their reconstruction

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

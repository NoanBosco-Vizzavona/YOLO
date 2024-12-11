# <font color='black'>Deep Learning for Image Processing</font>

---
<figure>
<center>
<img src='https://drive.google.com/uc?id=1u-iZKwzco1L8a3gFFLMXbSOU_DsdNZqo' width="300" align="center" />
</center>
</figure>

> Year: **2022**

> Version: **1.1**

- You need to solve this exercises in groups of two or three --> A group of one person will be penalized with -3 points.

Please upload your work on Moodle.

Good work and good luck!

# Generative Adversarial Networks

In this last practical you will learn how to implement and hopefully train a GANs and use it to

## Set-up

Firstly you will import all the packages used through the notebook.  


```python
import keras
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')
```

## Introduction

In what follows, you will learn how to implement a GAN using Keras. You will be using a deep convolutional GAN (DCGAN) As you saw in class it corresponds to a GAN where the generator and discriminator are deep convolutional networks.

You will implement the following methods:
- A generator network that maps vectors of shape (latent_dim,) to images of shape (32, 32, 3).
- A discriminator network that maps images of shape (32, 32, 3) to a score that describes how real the image looks.
- A GAN that combines the generator and the discriminator : GAN(x) = discriminator(generator(x)).
- We train the discriminator using examples of real and fake images along with "real"/"fake" labels, as we would train any regular image classification model.
- As discussed in class, we train the generator to fool the discriminator.


The GAN will be trained on images from CIFAR10. CIFAR10 is a dataset of 50,000 32x32 RGB images belonging to 10 classes (5,000 images per class). In this exercise you will only deal with the class "frog".

### Training tricks

Here are a few of the tricks that will help you train the GANs.

- We use tanh as the last activation in the generator.

- Sample points from the latent space using a normal distribution.

- Introducing randomness during training helps prevent geting "stuck". We will use dropout in the discriminator and  add some random noise to the labels for the discriminator.

- Instead of max pooling, you will be using strided convolutions for downsampling, and LeakyReLU activation.

##  The Generator Network

First, we create the generator network, which transforms a vector into a candidate image. One of the many problems that may arise with GANs training is that the generator gets stuck with images that look like noise. You should try to use dropout on both the discriminator and generator.

Make sure your architectures are small so that you can run them quickly enough.

You will need to use :
x = layers.Conv2D(256,kernel_size=5,padding='same')(x)

x = layers.LeakyReLU()(x)


# We are using Conv2DTranspose to upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)

x = layers.LeakyReLU()(x)



```python
from keras import layers
import numpy as np

latent_dim = 32

height = 32
width = 32
channels = 3

# Create the input placeholder in Keras
generator_input = None

# Transform the input into a 16x16 128 channels  (Dense, LeakyReLU and Reshape layers)
x = None
x = None
x = None

# Add a convolutional layer with 256 channels, f = 5 and same padding ; after add the non-linearity LeakyReLU
x = None
x = None

# We are using Conv2DTranspose to upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# Two more conv layers with 256 channels, f = 5 and same padding followed by LeakyReLU
x = None
x = None
x = None
x = None

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-4-4e9bbc4572e5> in <cell line: 24>()
         22 
         23 # We are using Conv2DTranspose to upsample to 32x32
    ---> 24 x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
         25 x = layers.LeakyReLU()(x)
         26 
    

    /usr/local/lib/python3.10/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
        120             # To get the full stack trace, call:
        121             # `keras.config.disable_traceback_filtering()`
    --> 122             raise e.with_traceback(filtered_tb) from None
        123         finally:
        124             del filtered_tb
    

    /usr/local/lib/python3.10/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
        174         # which does not have a `shape` attribute.
        175         if not hasattr(x, "shape"):
    --> 176             raise ValueError(
        177                 f"Inputs to a layer should be tensors. Got '{x}' "
        178                 f"(of type {type(x)}) as input for layer '{layer_name}'."
    

    ValueError: Inputs to a layer should be tensors. Got 'None' (of type <class 'NoneType'>) as input for layer 'conv2d_transpose_1'.


## The discriminator

The discriminator network takes as input a candidate image and classifies it into two classes, real (that comes from the training set) or generated by the network.


```python
discriminator_input = None
# Define a convolutional layer (128 channels, f = 3) followed by a Leakyrelu
x = None
x = None
# Define a convolutional layer (128 channels, f = 4 and strides = 2) followed by a Leakyrelu
x = None
x = None
# Define a convolutional layer (128 channels, f = 4 and strides = 2)followed by a Leakyrelu
x = None
x = None
# Define a convolutional layer (128 channels, f = 4 and strides = 2) followed by a Leakyrelu
x = None
x = None
# Use a Flatten layer
x = None

# Add a dropout layer with keep_prob = 0.6
x = None

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# We use learning rate decay and gradient clipping in the optimizer.
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
```

## The adversarial network

Finally, we create the GAN, which combines the generator and the discriminator.
During the training, the model will move the generator in order to improve its ability to fool the discriminator.

The discriminator needs to be frozen during training and its weights will not be updated during the training step.


```python
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
```

## Training DCGAN

For the training you need to follow:

for each epoch:
    - Draw random points from the latent space (noise).
    - Use the `generator` to create images from the noise.
    - Mix the generated images with the training ones.
    - Train `discriminator` using these mixed images, with corresponding labels, either "real" or "fake".
    - Draw new random points in the latent space.
    - Train `gan` using these random vectors, with labels that all say "these are real images" and update the weights of the generator and fool the discriminator.


```python
import os
from keras.preprocessing import image

# Load CIFAR10 data
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

# Select frog images (class 6)
x_train = x_train[y_train.flatten() == 6]

# Normalize data
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 5000
batch_size = 20

# MODIFY THE FOLLOWING LINE
save_dir = '/content/drive/MyDrive/0IPSA/Ma511/TP_22/TP5_Answers/gan_images/'

# Start training loop
start = 0
for step in range(iterations):
    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    # Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # Assemble labels discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    # Add random noise to the labels - important trick!
    labels += 0.05 * np.random.random(labels.shape)

    # Train the discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    misleading_targets = np.zeros((batch_size, 1))

    # Train the generator (via the gan model,
    # where the discriminator weights are frozen)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0

    # Occasionally save / plot
    if step % 100 == 0:
        # Save model weights
        gan.save_weights('gan.weights.h5')

        # Print metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # Save one generated image
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

        # Save one real image, for comparison
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))
```

You can now visualize you fake images:


```python
random_latent_vectors = np.random.normal(size=(10, latent_dim))
generated_images = generator.predict(random_latent_vectors)

for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)

plt.show()
```

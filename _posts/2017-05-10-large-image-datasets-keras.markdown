---
layout: post
title: "Training on Large Scale Image Datasets with Keras"
crawlertitle: "How to train large image datasets with keras"
summary: "Training large scale image datasets with keras"
date:   2017-05-10 23:10:47 +0700
categories: posts
tags: 'keras'
author: paul
---

In this post you'll learn how to train on large scale image datasets with Keras.
We'll leverage python generators to load and preprocess images in batches.

Thanks: [https://github.com/fchollet/keras/issues/68](https://github.com/fchollet/keras/issues/68)

We'll use the IMDB-WIKI dataset as an example.

### Loading the Dataset in Batches
Let's load the image dataset in batches of 100 images.

{% highlight python %}
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
'''
example. list of image paths
X_sample = [
  '10/123124.jpg',
  '11/543223.jpg',
  '08/797897897.jpg',
  ...
]

Corresponding age labels
y_sample = [
  28,
  40,
  19,
  ...
]

'''

def load_image(img_path, target_size=(244, 244)):
  im = load_image(img_path, target_size=target_size)
  return img_to_array(im) #converts image to numpy array

def IMDB_WIKI(X_sample, y_sample, batch_size=100):
  X_batches = np.split(X_samples, batch_size)
  y_batches = np.split(y_samples, batch_size)
  for b in range(X_batches):
    yield (map(X_batches[b], load_image), y_batches[b])

{% endhighlight %}

### Keras Image Generators
Keras [image generators](https://keras.io/preprocessing/image/) allow you to preprocess batches of images in real-time.

{% highlight python %}
form keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images

{% endhighlight %}

### Training Code

{% highlight python %}

n_epoch = 10
for e in range(n_epoch):
  print "epoch", e
  for X_train, y_train in IMDB_WIKI(X_sample, y_sample): # chunks of 100 images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32): # chunks of 32 samples
      loss = model.train_on_batch(X_batch, y_batch)


# If no image preprocessing is needed you can simply run
model.fit_generator(
  IMDB_WIKI(X_sample, y_sample, batch_size=100),
  epochs=10,
  steps_per_epoch=100)
{% endhighlight %}

---
layout: post
title:  "Running Tensorflow with Nvidia Docker"
crawlertitle: "How to use tensorflow and nvidia-docker"
summary: "How to use Tensorflow and Docker"
date:   2017-05-10 23:09:47 +0700
categories: posts
tags: 'tensorflow'
author: paul
---

In this post you'll learn how to use nvidia-docker with Tensorflow.

### Pulling the Tensorflow GPU Image

Use the correct image based on your CUDA version:

- CUDA 8.0 use `tensorflow/tensorflow:latest-gpu`
- CUDA 7.5 use `tensorflow/tensorflow:1.0.0-rc0-gpu`

Test nvidia-smi
```
nvidia-docker run --rm tensorflow/tensorflow:latest-gpu nvidia-smi
```

That's it! Now your ready to run your Tensorflow code.

### Example Dockerized Tensorflow Project using GPU

1) Create a Dockerfile
{% highlight bash %}
FROM tensorflow/tensorflow:latest-gpu

COPY . /app
WORKDIR /app
{% endhighlight  %}

2) Create app.py
{% highlight python %}
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# You should see the following output
# ...
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
# name: GeForce 940M
# │major: 5 minor: 0 memoryClockRate (GHz) 1.176
# ...
{% endhighlight  %}

3) Build a running nvidia-docker
{% highlight bash %}
nvidia-docker -t example_tf_gpu .
nvidia-docker run -it example_tf_gpu python app.py
{% endhighlight  %}

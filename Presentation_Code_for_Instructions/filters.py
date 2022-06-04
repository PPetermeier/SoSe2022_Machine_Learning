import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import sys

# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)



# matplotlib requires values in range 0-1 (see https://scikit-learn.org/stable/datasets/loading_other_datasets.html#sample-images)
plt.imshow(dataset[0]/255)
print(dataset[0].shape)
plt.show()


# 2 picture, x and y dimension, 3 colours
batch_size, height, width, channels = dataset.shape

print("Shape dataset",dataset.shape)

# Create 2 filters
# f_h=7, f_w=7, f_channel=3, #filters
filters_test = np.zeros(shape=(7, 7, channels, 3), dtype=np.float32)
filters_test[:, 4, :, 0] = 1 # vertical line
filters_test[4, :, :, 1] = 1 # horizontal line

# Apply two filters
output=tf.nn.conv2d(dataset, filters_test, strides=[1,2,2,1], padding="SAME")

# two filters => two feature maps
print("Shape of feature maps",output.shape)


# Show feature map
plt.subplot(1, 3, 1)
plt.imshow(output[0, :, :, 0])
plt.subplot(1, 3, 2)
plt.imshow(output[0, :, :, 1])
plt.subplot(1, 3, 3)
plt.imshow(output[0, :, :, 2])


plt.show()

# Apply pooling
ksize=[1,2,2,1]
strides=[1,2,2,1]
pool_output=tf.nn.max_pool(dataset,ksize=ksize,strides=strides,padding="VALID")

print("Shape of pooling output",output.shape)


plt.imshow(pool_output[0]/255)
plt.show()
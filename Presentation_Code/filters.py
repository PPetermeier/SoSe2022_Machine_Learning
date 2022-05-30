import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
from matplotlib import pyplot as plt

# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)

# matplotlib requires values in range 0-1 (see https://scikit-learn.org/stable/datasets/loading_other_datasets.html#sample-images)

plt.imshow(dataset[0]/255)
print(dataset[0].shape)
plt.show()

# 2 picture, x and y dimension, 3 colours
batch_size, height, width, channels = dataset.shape

print("Shape dataset",dataset.shape)

# Create 3 filters
# f_h=3, f_w=3, f_channel=3, #filters
f_h=7
f_w=7
# Filter shape: [filter_height, filter_width, in_channels, out_channels]
filters_test = np.zeros(shape=(f_h, f_w, channels, 3), dtype=np.float32)
filters_test[:, 3, :, 0] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line
filters_test[:, :, :, 2] = -1

#filters_test[2]=np.array([[0,-1,0],[-1,5,1],[0,-1,0]])
#print(f.shape)

print(filters_test[:, :, :, 0])

# Apply filters
output=tf.nn.conv2d(dataset, filters_test, strides=[1,2,2,1], padding="SAME")

# two filters => two feature maps
print("Shape of feature maps",output.shape)

plt.show()
# Show feature map
plt.subplot(1, 3, 1)
plt.imshow(output[0, :, :, 0],cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(output[0, :, :, 1],cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(output[0, :, :, 2],cmap="gray")


plt.show()

# Apply pooling
ksize=[1,2,2,1]
strides=[1,2,2,1]
pool_output=tf.nn.max_pool(dataset,ksize=ksize,strides=strides,padding="VALID")

plt.imshow(pool_output[0]/255)
plt.show()
print("Shape of pooling output",output.shape)



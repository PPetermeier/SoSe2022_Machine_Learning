from tensorflow.keras import Sequential
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from Presentation_Code_for_Instructions import utilities
import convLayerAnalysis as cla
from datetime import datetime
import tensorflow as tf

# Load train and test data
train_images, train_labels, test_images, test_labels = utilities.getDigits()

# Normalize color values (here: grey-scales)
train_images = train_images / 255.0
test_images = test_images / 255.0


print(train_images.shape)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

print(train_images.shape)


# Do one-hot encoding / do categorical conversion
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

num_epochs=5

# Extract number of classes from data dimensions
classes = np.shape(train_labels)[1]

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,write_graph=True)


# Define model architecture
model = Sequential()

# First convolutional and pooling layer
model.add(Conv2D(input_shape=(28, 28,1), filters=64, kernel_size=3, padding='valid', activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(strides=2, pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))

model.add(GlobalAveragePooling2D())
# FCC
model.add(Dense(units=16, activation='relu'))

# Classifier
model.add(Dense(units=classes, activation='softmax'))

# Compile model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


print("ConvLayers",cla.getConvLayers(model))

# Train model
training_history = model.fit(
    train_images, # input
    train_labels, # output
    batch_size=32,
    verbose=1, # Suppress chatty output; use Tensorboard instead
    epochs=num_epochs,
    validation_data=(test_images,test_labels),
    callbacks=[tensorboard_callback],
)

model.summary()
# Evaluate model
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print('Test accuracy:', test_acc)


# look into the model


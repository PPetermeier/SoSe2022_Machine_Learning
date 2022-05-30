import tensorflow as tf
from sklearn.datasets import load_digits
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import  plot_model




# scikit learn digits: Load and return the digits dataset (classification).
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
# 8 by 8 digits
def getSmallDigits():
    digits = load_digits()
    return digits

# load data
digits=getSmallDigits()

# split data into training and test
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target, test_size=0.10, random_state=0)
print(x_train.shape)
num_classes=10
# One hot vectors
t_train = tf.keras.utils.to_categorical(t_train, num_classes)
t_test = tf.keras.utils.to_categorical(t_test, num_classes)



train_size=x_train.shape[0]
test_size=x_test.shape[0]
print(train_size)

num_features=x_train.shape[1]
print(num_features)
learning_rate=0.1
training_epochs = 200
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = Sequential()

model.add(Dense(units=num_classes, input_dim=num_features))

model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',learning_rate=learning_rate,optimizer='sgd',metrics=['accuracy'])

#model.fit(x_train, t_train, epochs=training_epochs)
batch_size=16
training_history = model.fit(
    x_train, # input
    t_train, # output
    batch_size=batch_size,
    verbose=1, # Suppress chatty output; use Tensorboard instead
    epochs=training_epochs,
    validation_data=(x_test, t_test),
    callbacks=[tensorboard_callback],
)


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(model.metrics_names, model.evaluate(x_test,t_test))

print(x_test[0])

print(model.predict(x_test[0].reshape(1,-1)))
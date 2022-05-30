import tensorflow as tf
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
from sklearn.model_selection import train_test_split
from sklearn import metrics



# scikit learn digits: Load and return the digits dataset (classification).
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
# 8 by 8 digits
def getSmallDigits():
    digits = load_digits()
    return digits

def initialise(num_features,num_classes):
    W = tf.Variable(rand.random((num_features,num_classes)),name="weight", dtype=tf.float64,shape=(num_features,num_classes))
    return W


def predict(W,x):
    output = tf.nn.softmax(tf.matmul(x, W))
    return output

def error(output, y):
    dot_product = tf.cast(y, tf.float64) * tf.math.log(output)
    xentropy = -tf.reduce_sum(dot_product,axis=1)
    error = tf.reduce_mean(xentropy)
    return error


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    return accuracy

@tf.function
def run_optimisation(optimiser,W,x,t):
    with tf.GradientTape() as g:
        err  = error(predict(W,x),t)
    # Compute gradients.
    gradients = g.gradient(err, [W])
    # Update W and b following gradients.
    optimiser.apply_gradients(zip(gradients, [W]))

# load data
digits=getSmallDigits()

X_train, X_test, T_train, T_test = train_test_split(digits.data, digits.target, test_size=0.10, random_state=0)
num_classes=10
# we have to compute hte target distribution
t_train = tf.keras.utils.to_categorical(T_train, num_classes)
t_test = tf.keras.utils.to_categorical(T_test, num_classes)

train_size=X_train.shape[0]
num_features=X_train.shape[0]+1
test_size=X_test.shape[0]
print(train_size)
x_train = np.c_[np.ones((train_size, 1)), X_train]
x_test = np.c_[np.ones((test_size, 1)), X_test]

num_features=X_train.shape[1]+1

print(num_features)

print("Shapes x und y",x_train.shape,t_train.shape)


learning_rate=0.1
training_epochs = 1000

optimiser = tf.optimizers.SGD(learning_rate=learning_rate, name='SGD')
W=initialise(num_features,num_classes)

for epoch in range(training_epochs):
    run_optimisation(optimiser,W, x_train, t_train)
    if (epoch + 1) % 50 == 0:
        pred = predict(W,x_train)
        err = error(predict(W,x_train), t_train)
        print("step: %i, error: %f" % (epoch, err))

print("Shapes x und y",x_test.shape,t_test.shape)

test_pred = predict(W,x_test)
tf.print(evaluate(test_pred,t_test))
tf.print(error(predict(W,x_train), t_train))

# How can we get a confusion matrix?
#cm = metrics.confusion_matrix(t_train, test_pred)
#print(cm)

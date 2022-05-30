import tensorflow as tf
from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

import numpy as np
from matplotlib import pyplot as plt
import numpy.random as rand
from datetime import datetime

learning_rate = 0.01
training_epochs = 100
n = 100


def createTestData(n):
    x = 2 * rand.rand(n, 1)
    y = 4 + 3 * x + rand.randn(n, 1)
    return x, y


def createMultinomialTestData(n):
    x1 = 2 * rand.rand(n, 1)
    x2 = 3 * rand.rand(n, 1)
    y = 3 * x1 + 4 * x2 + 5 + rand.randn(n, 1)
    return np.hstack((x1, x2)), y


x, t = createMultinomialTestData(100)

# x,t=createTestData(100)
print(x)
num_features = x.shape[1]

model = Sequential()

model.add(Dense(units=1, input_dim=num_features))

model.add(Dense(units=1))

model.compile(loss='mse', optimizer='sgd')

model.fit(x, t, epochs=training_epochs)

if num_features == 1:
    plt.scatter(x, t)
    plt.xlabel('x')
    plt.xlabel('t')
    plt.plot(x, model.predict(x), label='Fitted line')
    plt.title("Training Data")
    plt.show()

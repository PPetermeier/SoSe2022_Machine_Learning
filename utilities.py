import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def randomPoints(n,m=3,b=4,seed=None):
    if seed!=None:
        np.random.seed(seed)
    # Create 100 random values for x between 0 and 2...
    x = 2 * np.random.rand(n, 1)
    # Create 100 values for y linearly dependent on x and add some noise...
    y = b + m * x + np.random.randn(n, 1)
    return x,y

def randomPointsQuadratic(m):
    x = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)
    return x,y


def learningCurvePlot(model, x, y):
    # Split data into trainig and validation, x and y values respectively
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_errors, val_errors = [], []
    # Increase number of training records used
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.ylim(0, 3)
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

def plotPoints(x,y,show=False):
    plt.plot(x, y, 'ro')

def plotLine(theta,show=False):
    xx = np.linspace(0, 2, 1000)
    plt.plot(xx, theta[1] * xx + theta[0], linestyle='solid')
    if show==True:
        plt.show()


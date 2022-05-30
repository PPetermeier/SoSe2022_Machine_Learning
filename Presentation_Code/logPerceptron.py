import utilities as util
import numpy as np
import functions as f
from matplotlib import pyplot as plt


n_epochs = 50
t0, t1 = 5, 50
m=100
eta = 0.1
iter = 1000

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# read iris data
def getData():
    iris=datasets.load_iris()
    x=iris["data"][:,3:]
    #print(x)
    y=(iris["target"]==2).astype(np.int)
    y=y.reshape(150,1)
    return x,y

# calculation of gradient
def getGradient(X,t,theta):
    return 1 / m * X.T.dot(f.sigmoid(X.dot(theta)) - t)

# batch gradient as before
def batchGradient(x,t):
    X = np.c_[np.ones((150, 1)), x]
    theta = np.random.randn(2, 1)
    for i in range(iter):
        gradients = getGradient(X,t,theta)
        theta = theta - eta * gradients
    return theta

# plot....
def doPlot(x,y_proba):
    plt.plot(x,y_proba[:,1],"g-",label="Iris-Virginica")
    plt.plot(x,y_proba[:,0],"b-",label="Not Iris-Virginica")
    plt.xlabel("Petal width (cm)")
    plt.ylabel("P")
    plt.legend()
    plt.show()

# prediction: multiply with theta and apply sigmoid
def doPredict(x,theta):
    preds=f.sigmoid(x.dot(theta))
    res=np.array(preds)
    res1=1-preds
    return np.hstack((res,res1))

def main():
    x,y=getData()
    theta=batchGradient(x,y)
    x_new=np.linspace(0,3,1000)
    X = np.c_[np.ones((1000, 1)), x_new]
    y_proba=doPredict(X,theta)
    doPlot(x_new,y_proba)

if __name__ == "__main__":
    main()

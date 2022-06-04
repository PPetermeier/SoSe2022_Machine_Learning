import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def getData(extract):
    iris=datasets.load_iris()
    x=iris["data"][:,extract:]
    y=(iris["target"]==2).astype(np.int)
    return x,y

def doRegression(x,y):
    log_regression=LogisticRegression(solver='newton-cg')
    log_regression.fit(x,y)
    return log_regression

def doPredict(log_regression):
    x_new=np.linspace(1,3,1000)
    x_new=x_new.reshape(-1,1)
    y_proba=log_regression.predict_proba(x_new)
    print(y_proba)
    return x_new,y_proba

def doPlot(x,y_proba):
    plt.plot(x,y_proba[:,1],"g-",label="Iris-Virginica")
    plt.plot(x,y_proba[:,0],"b-",label="Not Iris-Virginica")
    plt.xlabel("Petal width (cm)")
    plt.ylabel("P")
    plt.legend()
    plt.show()

# see also: https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py
def doPlot2D(lr,x,y):
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()


def main():

    x,y=getData(3)
    lr=doRegression(x,y)
    print(lr)
    # Plotting only for ine independent variable
    print(x)
    print(x.shape)
    if x.shape[1]==1:
        x_new, y_proba = doPredict(lr)
        print(x_new.shape,y_proba.shape)
        doPlot(x_new,y_proba)
    if x.shape[1]==2:
        doPlot2D(lr,x,y)
        print(lr.predict([[5.1, 1.75]]))

if __name__ == "__main__":
    main()
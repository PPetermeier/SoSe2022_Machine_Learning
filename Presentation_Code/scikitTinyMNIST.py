from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# scikit learn digits: Load and return the digits dataset (classification).
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
# 8 by 8 digits
def getSmallDigits():
    digits = load_digits()

    # shape of data
    #print(digits.data.shape)

    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
        plt.title('Image: %i\n' % label, fontsize = 12)

    plt.show()

    return digits


digits=getSmallDigits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=0)

#print(x_train.shape)
logisticRegr = LogisticRegression(max_iter=25,solver='newton-cg',multi_class="multinomial")
logisticRegr.fit(x_train, y_train)

#print("test shape 0",x_test[0].shape)
#print(x_test[0].reshape(1,-1).shape)
#print(x_test[0])
#print(x_test[0].reshape(1,-1))
#print(logisticRegr.predict(x_test[0].reshape(1,-1)))

# make predictions on test dataset
#print(x_test.shape)
#print(x_test)
predictions = logisticRegr.predict(x_test)

# score is accuracy, why not just error?
score = logisticRegr.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

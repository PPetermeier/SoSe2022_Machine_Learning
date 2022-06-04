import numpy as np

from sklearn.metrics import mean_squared_error

np.random.seed(0)

def activation(func,x):
    if func=="sigmoid":
        return sigmoid(x)
    if func=="identity":
        return identity(x)

def activation_derivative(func,x):
    if func=="sigmoid":
        return sigmoid_derivative(x)
    if func=="identity":
        return identity_derivative(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity(x):
    return x

def sigmoid_derivative(sx):
    return sx * (1 - sx)

def identity_derivative(sx):
    return 1

# Cost functions.
def cost(predicted, truth):
    m=len(predicted)
    return (truth - predicted)


xor_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_output = np.array([[0, 1, 1, 0]]).T

X = xor_input
Y = xor_output

num_data, input_dim = X.shape
hidden_dim = 5
W1 = np.random.random((input_dim, hidden_dim))
print(W1.shape)
output_dim = len(Y.T)
W2 = np.random.random((hidden_dim, output_dim))
print(W2.shape)

num_epochs = 1000
learning_rate = 1.0
func="sigmoid"

for epoch_n in range(num_epochs):
    layer0 = X

    # forward propagation
    layer1 = activation(func,np.dot(layer0, W1))
    layer2 = activation(func,np.dot(layer1, W2))

    # backpropagation

    layer2_error = cost(layer2, Y)
    layer2_delta = layer2_error * activation_derivative(func,layer2)

    layer1_error = np.dot(layer2_delta, W2.T)
    layer1_delta = layer1_error * activation_derivative(func,layer1)

    # update weights
    W2 += learning_rate * np.dot(layer1.T, layer2_delta)
    W1 += learning_rate * np.dot(layer0.T, layer1_delta)

for x, y in zip(X, Y):
    layer1_prediction = activation(func,np.dot(W1.T, x)) # Feed the unseen input into trained W.
    prediction = layer2_prediction = activation(func,np.dot(W2.T, layer1_prediction)) # Feed the unseen input into trained W.
    print(int(prediction > 0.5),x,y)

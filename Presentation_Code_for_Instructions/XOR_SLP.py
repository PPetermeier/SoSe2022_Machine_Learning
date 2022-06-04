import numpy as np

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
print(xor_input)
print(xor_output)

# Lets drop the last row of data and use that as unseen test.
X = xor_input
Y = xor_output

# Define the shape of the weight vector.
num_data, input_dim = X.shape
print(X.shape)
# Lets set the dimensions for the intermediate layer.
output_dim = len(Y.T)
print(output_dim)

# Initialize weights between the input layers and the hidden layer.
W = np.random.random((input_dim, output_dim))
print(W.shape)
print(W)

num_epochs = 1000
learning_rate = 1.0
func="sigmoid"

for epoch_n in range(num_epochs):
    layer = X
    layer = activation(func,np.dot(layer, W))
    layer_error = cost(layer, Y)
    layer_delta = layer_error * activation_derivative(func,layer)

    W += learning_rate * np.dot(layer.T, layer_delta)

for x, y in zip(X, Y):
    prediction = activation(func,np.dot(W.T, x))
    print(int(prediction > 0.5),x,y)

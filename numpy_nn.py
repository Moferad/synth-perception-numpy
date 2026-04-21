import numpy as np
import copy

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def initialize_parameters(n_x,n_h,n_y):

    ## n_x - size of the input layer
    ## n_h - size of the hidden layer
    ## n_y - size of the output layer

    ## returns:
    ## parameters - python dictionary containing parameters:
    ## W1 - weight matrix of shape (n_h,n_x)
    ## b1 - bias vector of shape (n_h,1)
    ## W2 - weight matrix of shape (n_y,n_h)
    ## b2 - bias vector of shape (n_y, 1)

    np.random.seed(1)

    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1" : W1,"b1": b1, "W2": W2, "b2": b2 }

    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)

    cache = (linear_cache,activation_cache)

    return A, cache

def compute_cost(AL,Y):

    ## AL - probability vector corresponding to your label predictions, shape (1, number of examples)
    ## Y - true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    #Returns:
    ## cost - cross-entropy cost

    m = Y.shape[1]
    cost = -1/m * np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)

    return cost

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

def update_parameters(params,grads,learning_rate):

    ## update parameters using gradient descent

    parameters = copy.deepcopy(params)
    L = len(parameters)//2 #divide by 2 for W & b on each layer

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


def train(X, Y, n_h, learning_rate=0.01, num_iterations=1000, print_cost=True):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    costs = []
    
    parameters = initialize_parameters(n_x,n_h,n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(num_iterations):
        
        A1, cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2, cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        
        cost = compute_cost(A2,Y)

        dA2 = - (np.divide(Y,A2) - np.divide(1-Y,1-A2))

        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,"relu")
        
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
            
        if print_cost and i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters, costs

def predict(X,parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    A1, cache1 = linear_activation_forward(X,W1,b1,"relu")
    A2, cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
    predictions = (A2 > 0.5).astype(int)
    
    return predictions
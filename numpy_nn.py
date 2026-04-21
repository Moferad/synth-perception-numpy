import numpy as np

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
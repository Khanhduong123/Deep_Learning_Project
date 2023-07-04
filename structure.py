import numpy as np

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) - 1
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        print(f'Shape of parameters W{l}: ',parameters['W' + str(l)].shape)  
        print(f'Shape of parameters b{l}: ',parameters['b' + str(l)].shape) 
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    print('Shape Z: ',Z.shape)
    # print('Shape cache: ',cache.shape)
    return Z, cache

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    
    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    cross_entropy_cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
    
    L2_regularization_cost = 0
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
    
    L2_regularization_cost *= lambd / (2 * m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

def linear_backward(dZ, cache, lambd):
    A_prev, W, _ = cache
    print('Shape of A_previous: ',A_prev.shape)
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m + (lambd / m) * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    dZ = dA * sig * (1 - sig)
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_activation_backward(dA, cache, activation, lambd):
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu", lambd)
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
        
    return parameters

def predict(X, parameters):
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

# Importing Numpy Library
import numpy as np
import scipy.io
import math
import sys

# Importing Pandas
import pandas as pd

# Importing classification evaluation metrics from Sklearn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score


## Defining function: Sigmoid Function
###########################################################################################################################
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    x[x > 500] = 500
    x[x < -500] = -500
    s = 1/(1+np.exp(-x))
    return s


## Defining function: ReLU Function
###########################################################################################################################
def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s


## Defining function: Initialize Parameters
###########################################################################################################################
def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], layer_dims[l-1]
        assert parameters['W' + str(l)].shape[0] == layer_dims[l], 1
        
    return parameters
  

## Defining function: Initialize Adam
###########################################################################################################################
def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(1, L + 1):

        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    return v, s


## Defining function: Update Parameters With Adam
###########################################################################################################################
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    t -- Adam variable, counts the number of taken steps
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*grads['dW' + str(l)]
        v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*grads['db' + str(l)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1-beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)]/(1-beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = beta2*s["dW" + str(l)] + (1-beta2)*grads['dW' + str(l)]**2
        s["db" + str(l)] = beta2*s["db" + str(l)] + (1-beta2)*grads['db' + str(l)]**2

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l)] = s["dW" + str(l)]/(1-beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)]/(1-beta2**t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*v_corrected["dW" + str(l)]/(np.sqrt(s_corrected["dW" + str(l)])+epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*v_corrected["db" + str(l)]/(np.sqrt(s_corrected["db" + str(l)])+epsilon)

    return parameters, v, s, v_corrected, s_corrected  


## Defining function: Random Mini Batches
###########################################################################################################################
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    inc = mini_batch_size

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    # Number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k*inc:(k+1)*inc]
        mini_batch_Y = shuffled_Y[:, k*inc:(k+1)*inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, (k+1)*inc:]
        mini_batch_Y = shuffled_Y[:, (k+1)*inc:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


## Defining function: Forward Propagation Architecture version 01
###########################################################################################################################
def forward_propagation(X, parameters):

    """
  Retrieve the mlflow experiment with the given id or name.

  Parameters:
  ----------
  experiment_id: str
    The id of the experiment to retrieve.
  experiment_name:
    The name of the experiment to retrieve.
  
  Returns:
  ----------
  experiment: mlflow.entities.Experiment
    The mlflow experiment with the given id or name.
  """
    """
    Implements forward propagation.
    
    Arguments:
    ----------
    X: numpy.ndarray 
        The input dataset of shape (input size, number of examples)

    parameters: dict
        A python dictionary containing the trained parameters "W1", "b1", "W2", "b2", "W3", "b3":
        W1: weight matrix of shape (n_neurons_l1, input size)
        b1: bias vector of shape (n_neurons_l1, 1)
        W2: weight matrix of shape (n_neurons_l2, n_neurons_l1)
        b2: bias vector of shape (n_neurons_l2, 1)
        W3: weight matrix of shape (n_neurons_l3, n_neurons_l2)
        b3: bias vector of shape (n_neurons_l3, 1)
    
    Returns:
    ----------
    a3: numpy.ndarray
        The values calculated after the acivation function for each neuron in layer l3

    cache: tuple
        A tuple containing all calculaions that will be used during backpropagation.
    """
    
    # Retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)

    # LINEAR -> RELU
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)

    # LINEAR -> CUSTOM (SIGMOID + IDENTITY)
    # Neuron with sigmoid activation function predicts propensity
    # Neuron with identity activation function predicts Customer Lifetime Value
    z3 = np.dot(W3, a2) + b3
    a3 = np.concatenate([sigmoid(z3[0,:]).reshape(1,-1), z3[1,:].reshape(1,-1)], axis=0)
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    
    return a3, cache

## Defining function: Backward Propagation Architecture version 01
###########################################################################################################################
def loss_function(a3, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    total_loss - value of the cost function without dividing by number of training examples
    
    Note: 
    This is used with mini-batches, 
    so we'll first accumulate costs over an entire epoch 
    and then divide by the m training examples
    """

    def binary_cross_entropy_loss(y_true, y_pred):

        # Defining a small value to prevent divide by zero
        epsilon=1e-8

        # Creating a Greater-than-zero Indicator Array
        ind_1 = y_true.copy()
        ind_1[ind_1 > 0] = 1

        # Calculating Binary Cross-Entropy Loss
        logprobs = np.multiply(-np.log(y_pred+epsilon), ind_1) + np.multiply(-np.log(1-y_pred+epsilon), 1 - ind_1)
        bce_loss =  np.sum(logprobs)

        return bce_loss

    def mean_squared_error_loss(y_true, y_pred):

        # Defining a small value to prevent divide by zero
        epsilon=1e-8

        # Creating a Greater-than-zero Indicator Array
        ind_1 = y_true.copy()
        ind_1[ind_1 > 0] = 1

        # Calculating Mean Squared Error
        mse_error = np.multiply((y_true-y_pred)**2, ind_1)
        mse_loss = np.sum(mse_error)

        return mse_loss

    # Reshaping vectors
    Y  = Y.reshape(1,-1).T
    p  = a3[0,:].reshape(1,-1).T
    mu = a3[1,:].reshape(1,-1).T
    
    # Calculating Binary Cross-Entropy Loss
    bce_loss =  binary_cross_entropy_loss(y_true=Y, y_pred=p)

    # Calculating Root Mean Squared Error
    mse_loss = mean_squared_error_loss(y_true=Y, y_pred=mu)

    # Calculating Total Cost
    total_loss = bce_loss + mse_loss
    
    return total_loss


## Defining function: Backward Propagation Architecture version 01
###########################################################################################################################
def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    # Creating a Greater-than-zero Indicator Array
    I1 = Y.copy()
    I1[I1 > 0] = 1
    
    # Calculating layer 3 backpropagation
    dz3_0 =  (1/m) * ((1 - I1) / (1 - a3[0,:]) - I1 / a3[0,:]) * a3[0,:]  # This derivative considers that the first neuron in the last layer uses a sigmoid activation function and a BCE cost function
    dz3_1 =  (2/m) * (a3[1,:] - Y) * I1 # This derivative considers that the second neuron in the last layer uses a identity activation function and a MSE cost function
    dz3 = np.concatenate([dz3_0.reshape(1,-1), dz3_1.reshape(1,-1)], axis=0)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    
    # Calculating layer 2 backpropagation
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    # Calculating layer 1 backpropagation
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients


## Defining function: Print Progress
###########################################################################################################################
def print_progress(epoch_id, batch_id, n_batches, tr_loss, vd_loss):
  """
  This function receives the current time, the start time (when the network clustering process actually started), the number of clusters found in the network,
  as for the current time, the number of remaining sub-clusters in the network and the original size of the network. The function then prints a log of the
  status of the clustering process.

  Args:
      epoch_id (int): The epoch number of the current run.
      batch_id (int): The batch number of the current run.
      n_batches (int): The total number of batches.
      mean_loss (float): The current mean loss in train set.
      tr_auc (float): Train AUC.
      vd_auc (float): Validation AUC.

  Returns:
      The function prints a log of the status of the clustering process.
  """

  # Creating response string
  response = "Epoch: {} | Batch: {}/{} | Train loss: {} | Valid loss: {}"

  # Plot status
  sys.stdout.write('\r')
  sys.stdout.write(response.format(
    str(epoch_id).zfill(4), 
    str(batch_id).zfill(4), 
    str(n_batches).zfill(4), 
    np.round(tr_loss,4), 
    np.round(vd_loss,4),  
    )
  )
  sys.stdout.flush()


## Deep Learning Model
###########################################################################################################################
def model(X, Y, X_valid, Y_valid, layers_dims, file_name, params=None, learning_rate=0.0007, mini_batch_size=64, 
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    optimizer -- the optimizer to be passed, gradient descent, momentum or adam
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    n = X_valid.shape[1]             # number of validation examples

    # Deining Weights Storing Location
    file_path = "../data/neural_network_classification/nn_{param}_{file_name}.json"
    
    # Initialize parameters
    if params == None:
        parameters = initialize_parameters(layer_dims=layers_dims)
    else:
        parameters = params

    # Initialize Adam optimizer
    v, s = initialize_adam(parameters=parameters)
        
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X=X, Y=Y, mini_batch_size=mini_batch_size, seed=seed)
        total_loss = 0
        k = len(minibatches) # number of mini-batches
        
        # Iterating through all mini-batches
        for j, minibatch in enumerate(minibatches):

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            total_loss += loss_function(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s, _, _ = update_parameters_with_adam(
                parameters=parameters, 
                grads=grads, 
                v=v, 
                s=s, 
                t=t, 
                learning_rate=learning_rate, 
                beta1=beta1, 
                beta2=beta2, 
                epsilon=epsilon
            )

            # Calcularing the AUC on the train set
            tr_loss = total_loss/((j+1)*mini_batch_size)

            # Calcularing the AUC on the validation set
            a3_valid, _ = forward_propagation(X_valid, parameters)
            vd_loss = loss_function(a3_valid, Y_valid)/n
            
            # Printing progress
            print_progress(
                epoch_id=i, 
                batch_id=j, 
                n_batches=k, 
                tr_loss=tr_loss,
                vd_loss=vd_loss
            )
            
            # Ssve weights every 1000 mini-batch
            if j%1000 == 0:
                for param in parameters:
                    with open(file_path.format(param=param, file_name=file_name), 'wb') as f:
                        np.save(f, parameters[param])
            
    return parameters, costs

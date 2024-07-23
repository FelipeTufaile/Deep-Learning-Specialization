## DEPENDENCIES
##############################################################################################################

# Importing Tensorflow and Tensorflow layers 
import tensorflow as tf
import tensorflow.keras.layers as tfl

# Importing Keras
import keras 

# Importing Numpy
import numpy


## Defining cost function: Zero-Inflated Mean Squared Error (ZIMSE)
##############################################################################################################
def zero_inflated_mean_squared_error_loss(y_true:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
  """Computes the zero-inflated mean squared error loss.

  Note: In order to really leverage the capabilities of the Zero-Inflated Mean Squared Error model make sure
  that the customer lifetime value labels are as close as possible to a normal distribution. In case it is not
  normally distributed, it is advisable to transaform the feature using the function y = np.log(x + 1).
  In case the previous transformation is implemented, you may want to apply the following inverse transformation
  to restore customer lifetime values: x = np.exp(y). It is not necessary to remove the unit ("-1") in the inverse
  function. The model itself is capable of abstracting this constant during training. 

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('Adam', loss=zero_inflated_mean_squared_error_loss)
  ```

  Arguments:
    y_true [tf.Tensor]: True targets, tensor of shape [batch_size, 1].
    y_pred [tf.Tensor]: Tensor of output layer, tensor of shape [batch_size, 2].

  Returns:
    Zero-inflated mean squared error loss value.
  """

  ## Creating a tensor from y_true. 
  # This will a tensor with shape (m, 1), where "m" is the number of samples
  tf_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

  ## Creating a tensor from y_pred. 
  # This will a tensor with shape (m, n), where "m" is the number of samples and "n" is the number of outputs.
  # Output 1: probability of transaction (calculated from sigmoid activation function)
  # Output 2: Customer Lifetime Value (calculated from linear activation function)
  tf_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

  ## Calculating true labels.
  # This is necessary for the classification part of the cost function:
  # We want to distinguish CLV greater than 0 from CLV equal to zero.
  true_labels = tf.cast(tf_true > 0, tf.float32)

  ## Calculating predicted labels.
  # Transaction probabilities correspond to the first output of the y_pred tensor.
  pred_labels = tf_pred[..., :1]

  ## Classification loss
  # In order to calculate the classification loss we use Binary Cross-Entropy
  classification_loss = tf.keras.losses.binary_crossentropy(true_labels, pred_labels, from_logits=False)
  classification_loss = tf.reshape(classification_loss, [-1]) # Reshapes into a vector (m,)

  ## Adapting Customer Lifetime Value vector
  #pred_clv = tf.math.multiply(true_labels, tf_pred[..., 1:2])
  pred_clv = tf_pred[..., 1:2]

  ## Regression loss
  # In order to calculate the regression loss we use Mean-Squared Error
  regression_loss = tf.math.square(tf_true - pred_clv)
  regression_loss = tf.reshape(regression_loss, [-1]) # Reshapes into a vector (m,)

  ## Total Loss
  # Finally the total loss will be the sum of the classification loss and the regression loss
  total_loss = classification_loss + regression_loss

  return total_loss


## Defining model: Zero-Inflated Mean Squared Error Model
##############################################################################################################
def zimse_model(input_shape:int, print_summary=False) -> keras.engine.functional.Functional:
  """Creates a neural network model to be used with the zero-inflated mean squared error loss function.

  Note: In order to really leverage the capabilities of the Zero-Inflated Mean Squared Error model make sure
  that the customer lifetime value labels are as close as possible to a normal distribution. In case it is not
  normally distributed, it is advisable to transaform the feature using the function y = np.log(x + 1).
  In case the previous transformation is implemented, you may want to apply the following inverse transformation
  to restore customer lifetime values: x = np.exp(y). It is not necessary to remove the unit ("-1") in the inverse
  function. The model itself is capable of abstracting this constant during training. 

  Usage with tf.keras API:

  ```python
  model.compile('Adam', loss=zero_inflated_mean_squared_error_loss)
  ```

  Arguments:
    input_shape [int]: An integer informing the number of features in the data.

  Returns:
    Model [keras.engine.functional.Functional]: TF Keras model (object containing the information for the entire training process using the zero-inflated mean squared error loss function).
  """

  # Defining input layer
  L0 = tf.keras.Input(shape=input_shape)

  # Defining Layer 1
  # Layer 1 has 64 neurons (unities) and a relu activation function
  L1 = tfl.Dense(units=64, activation="relu")(L0)

  # Defining Layer 2
  # Layer 2 has 32 neurons (unities) and a relu activation function
  L2 = tfl.Dense(units=32, activation="relu")(L1)

  # Defining Layer 3
  # Layer 3 has two neurons (unities) which one with a different activation function
  # The first activation function is a "sigmoid" function that is used to calculate the propensity score
  # The second activation functin is a "linear" function that is used to calculate customer lifetime value
  L3 = tfl.concatenate(
    [
      tfl.Dense(units=1, activation="sigmoid")(L2), # This neuron (unity) calculates the propensity score
      tfl.Dense(units=1, activation="relu")(L2) # This neuron (unity) calculates the customer lifetime value
    ]
  )

  # Defiing the zimse model
  # The zimse model has two outputs:
  # Output 1: propensity score (also interpreted as probability of churn)
  # Output 2: customer lifetime value
  model = tf.keras.Model(inputs=L0, outputs=L3)

  # check if it should print a summary of the model
  if print_summary:

    # summarize layers
    print(model.summary())

  return model


## Defining cost function: Zero-Inflated Normal Maximum Log-Likelihood Estimator (ZINMLE)
##############################################################################################################
def zero_inflated_normal_maximum_loglikelihood_estimator_loss(y_true:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
  """Computes the zero-inflated normal maximum log-likelihood estimator loss.

  Note: In order to really leverage the capabilities of the Zero-Inflated Normal Maximum Log-Likelihood Estimator model 
  make sure that the customer lifetime value labels are as close as possible to a normal distribution. In case it is not
  normally distributed, it is advisable to transaform the feature using the function y = np.log(x + 1).
  In case the previous transformation is implemented, you may want to apply the following inverse transformation
  to restore customer lifetime values: x = np.exp(y). It is not necessary to remove the unit ("-1") in the inverse
  function. The model itself is capable of abstracting this constant during training. 

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('Adam', loss=zero_inflated_normal_maximum_loglikelihood_estimator_loss)
  ```

  Arguments:
    y_true [tf.Tensor]: True targets, tensor of shape [batch_size, 1].
    y_pred [tf.Tensor]: Tensor of output layer, tensor of shape [batch_size, 3].

  Returns:
    zero-inflated normal maximum log-likelihood estimator loss value.
  """

  ## Creating a tensor from y_true. 
  # This will a tensor with shape (m, 1), where "m" is the number of samples
  tf_true = tf.reshape(tf.convert_to_tensor(y_true, dtype=tf.float32), [-1])

  ## Creating a tensor from y_pred. 
  # This will a tensor with shape (m, n), where "m" is the number of samples and "n" is the number of outputs.
  # Output 1: probability of transaction (result from sigmoid activation function)
  # Output 2: the "center" value of the customer lifetime value normal distribution (calculated from a linear activation function)
  # Output 3: the "standard deviation" of the customer lifetime value normal distribution (calculated from a softplus activation function)
  tf_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

  ## Calculating true labels.
  # This is necessary for the classification part of the cost function:
  # We want to distinguish CLV greater than 0 from CLV equal to zero.
  true_labels = tf.cast(tf_true > 0, tf.float32)

  ## Extracting tensors from y_pred.
  p_tensor  = tf.reshape(tf_pred[..., 0:1], [-1]) # Tensor corresponding to the probability of transaction.
  mu_tensor = tf.reshape(tf_pred[..., 1:2], [-1]) # Tensor corresponding to the center of the CLV normal distribution.
  sd_tensor = tf.reshape(tf_pred[..., 2:3], [-1]) # Tensor corresponding to the standard deviation of the CLV normal distribution.

  ##### Classification loss #####
  # In order to calculate the classification loss it will be used Binary Cross-Entropy
  classification_loss = tf.keras.losses.binary_crossentropy(true_labels, p_tensor, from_logits=False)

  ##### Regression loss #####
  # In order to calculate the regression loss it will be used the Maximum Log-Likelihood Estimator applied to a Normal distribution.

  # Defining a constant "k"
  k = 0.9189385332046727
  
  ## Calculating the log of the standard deviation
  # The softplus function makes sure that the standard deviation is always greater than zero.
  # Therefore, there is no need to adjust the log calculation for values <= 0.
  log_sd_tensor = tf.math.log(sd_tensor)

  ## Calculating variance
  var_tensor = tf.math.square(sd_tensor)

  ## Calculating the squared error
  sq_err_tensor = tf.math.square(tf_true - mu_tensor)

  ## Calculating Regression Loss
  regression_loss = k + log_sd_tensor + tf.math.divide(sq_err_tensor, var_tensor)/2

  ##### Total Loss #####
  # Finally the total loss will be the sum of the classification loss and the regression loss
  total_loss = classification_loss + regression_loss

  return total_loss


## Defining model: Zero-Inflated Normal Maximum Log-Likelihood Estimator (ZINMLE) Model
##############################################################################################################
def zinmle_model(input_shape:int, print_summary=False) -> keras.engine.functional.Functional:
  """Creates a neural network model to be used with the zero-inflated normal maximum log-likelihood estimator loss function.

  Note: In order to really leverage the capabilities of the Zero-Inflated Normal Maximum Log-Likelihood Estimator model 
  make sure that the customer lifetime value labels are as close as possible to a normal distribution. In case it is not
  normally distributed, it is advisable to transaform the feature using the function y = np.log(x + 1).
  In case the previous transformation is implemented, you may want to apply the following inverse transformation
  to restore customer lifetime values: x = np.exp(y). It is not necessary to remove the unit ("-1") in the inverse
  function. The model itself is capable of abstracting this constant during training. 

  Usage with tf.keras API:

  ```python
  model.compile('Adam', loss=zero_inflated_mean_squared_error_loss)
  ```

  Arguments:
    input_shape [int]: An integer informing the number of features in the data.

  Returns:
    Model [keras.engine.functional.Functional]: TF Keras model 
      (object containing the information for the entire training process using the zinmle loss function).
  """

  # Defining input layer
  L0 = tf.keras.Input(shape=input_shape)

  # Defining Layer 1
  # Layer 1 has 64 neurons (unities) and a relu activation function
  L1 = tfl.Dense(units=64, activation="relu")(L0)

  # Defining Layer 2
  # Layer 2 has 32 neurons (unities) and a relu activation function
  L2 = tfl.Dense(units=32, activation="relu")(L1)

  # Defining Layer 3
  # Layer 3 has two neurons (unities) which one with a different activation function
  # The first activation function is a "sigmoid" function that is used to calculate the propensity score
  # The second activation functin is a "linear" function that is used to calculate customer lifetime value
  L3 = tfl.concatenate(
    [
      tfl.Dense(units=1, activation="sigmoid")(L2), # This neuron (unity) calculates the propensity score
      tfl.Dense(units=1, activation="relu")(L2), # This neuron (unity) calculates the center of the CLV normal distribution
      tfl.Dense(units=1, activation="softplus")(L2) # This neuron (unity) calculates the standard deviation of the CLV normal distribution
    ]
  )

  # Defiing the zimse model
  # The zimse model has two outputs:
  # Output 1: propensity score (also interpreted as probability of churn)
  # Output 2: center of the CLV normal distribution
  # Output 3: standard deviation of the CLV normal distribution
  model = tf.keras.Model(inputs=L0, outputs=L3)

  # check if it should print a summary of the model
  if print_summary:

    # summarize layers
    print(model.summary())

  return model

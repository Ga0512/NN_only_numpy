# NN_only_numpy
A standard neural network using just numpy and math, no tensorflow or pytorch.

**Input Layer**: 784 (28x28)

**Hidden layer**: 10
ReLu

**Output layer**: 10
SoftMax

## Functions used for the neural network

**init_params()**: Initializes the neural network parameters with random values ​​between -0.5 and 0.5.

**ReLU(Z)**: Applies the ReLU activation function to Z values, replacing negative ones with 0.

**softmax(Z)**: Converts Z values ​​into normalized probabilities for classification.

**forward_prop(W1, b1, W2, b2, X)**: Performs direct propagation in the neural network, calculating layer activations.

**ReLU_deriv(Z)**: Calculates the derivative of the ReLU function to adjust the weights during backpropagation.

**one_hot(Y)**: Converts Y labels to one-hot encoding to calculate the error during backpropagation.

**backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)**: Calculates the gradients of weights and biases during backpropagation.

**update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)**: Updates the neural network parameters with the calculated gradients.

**get_predictions(A2)**: Returns the neural network predictions based on the activations of the last layer.

**get_accuracy(predictions, Y)**: Calculates the accuracy of the neural network's predictions relative to the true labels.

**gradient_descent(X, Y, alpha, iterations)**: Runs the gradient descent algorithm to train the neural network, updating the parameters iteratively.

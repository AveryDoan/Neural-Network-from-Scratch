import numpy as np

class Layer:
    """
    Abstract Base Class for all neural network layers.
    Each layer must implement forward and backward passes.
    """
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError

class Linear(Layer):
    """
    Fully Connected (Linear) Layer.
    Computes: Output = Input * W + b
    """
    def __init__(self, input_size, output_size, weight_scale=0.01):
        super().__init__()
        # Initialize weights with small random values from a normal distribution
        self.params['W'] = weight_scale * np.random.randn(input_size, output_size)
        # Initialize biases to zeros
        self.params['b'] = np.zeros((1, output_size))
        
        self.grads['W'] = None
        self.grads['b'] = None
        self.input = None

    def forward(self, input_data):
        """
        Forward pass: compute dot product and add bias.
        """
        self.input = input_data
        return np.dot(input_data, self.params['W']) + self.params['b']

    def backward(self, output_error):
        """
        Backward pass: compute gradients for weights, biases, and input.
        """
        # Gradient with respect to weights: dL/dW = X.T * dL/dY
        self.grads['W'] = np.dot(self.input.T, output_error)
        # Gradient with respect to bias: dL/db = sum(dL/dY) across batch
        self.grads['b'] = np.sum(output_error, axis=0, keepdims=True)
        # Gradient with respect to input: dL/dX = dL/dY * W.T
        input_error = np.dot(output_error, self.params['W'].T)
        return input_error

class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) Activation Layer.
    Computes: f(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_error):
        """
        Gradient of ReLU is 1 for x > 0 and 0 otherwise.
        """
        dz = np.copy(output_error)
        dz[self.input <= 0] = 0
        return dz

class Sequential:
    """
    Container to stack layers in sequence.
    """
    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        """
        Chain the forward passes of all layers.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_error):
        """
        Chain the backward passes in reverse order.
        """
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error)
        return output_error

class CrossEntropyLoss:
    """
    Softmax activation + Cross Entropy Loss combined for numerical stability.
    """
    def __init__(self):
        self.probs = None

    def loss(self, scores, y_true):
        """
        Computes the loss and saves probabilities for backward pass.
        y_true should be one-hot encoded or indices.
        """
        # Numerical stability: shift scores so max is 0
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        N = scores.shape[0]
        # Cross-entropy loss: -log(p_i) for the correct class
        # Assuming y_true is one-hot
        log_likelihood = -np.log(self.probs[y_true.astype(bool)] + 1e-10)
        return np.mean(log_likelihood)

    def backward(self, y_true):
        """
        Gradient of Softmax + Cross Entropy: (probs - target) / N
        """
        N = y_true.shape[0]
        return (self.probs - y_true) / N

class SGD:
    """
    Stochastic Gradient Descent Optimizer.
    """
    def __init__(self, layers, lr=0.01, reg=1e-4):
        self.layers = layers
        self.lr = lr
        self.reg = reg

    def step(self):
        """
        Update parameters using gradients and optional L2 regularization.
        """
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    grad = layer.grads[key]
                    # Add L2 regularization gradient if it's a weight matrix
                    if key == 'W':
                        grad += self.reg * layer.params[key]
                    layer.params[key] -= self.lr * grad

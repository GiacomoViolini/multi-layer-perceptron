import numpy as np
import pandas as pd
import time

# Load data
data = pd.read_csv('./data/train.csv')

# Preprocess data
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


hidden_neurons = 256
# Initialize parameters
def init_params():
    W1 = np.random.rand(256, 784) - 0.5
    b1 = np.random.rand(256, 1) - 0.5
    W2 = np.random.rand(10, 256) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Activations
def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y) # encode y values to match nodes activations
    dZ2 = A2 - one_hot_Y # derivative of cross-entropy loss
    dW2 = 1 / m_train * dZ2.dot(A1.T) # Z2​=W2​A1​+b2​ dW2​=​∂L∂W2​​=​∂L∂Z2​​⋅​∂Z2∂W2​​=dZ2​A1T​
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True) # db2​=​∂L∂b2​​=​∂L∂Z2​​⋅​∂Z2∂b2​​=dZ2​
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # A1​=ReLU(Z1​) ​∂L​∂Z1=​∂L​∂A1​​⊙∂A1​​∂Z1​ ​∂L​∂A1=​∂L​∂Z2​​⋅∂Z2​​∂A1​
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Predictions and accuracy
def get_predictions(A2):
    return np.argmax(A2, 0)

def  get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    start_time = time.time()
    forward_times , backward_times = 0, 0
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        start_fwd = time.time()
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        forward_times += time.time() - start_fwd
        start_bwd = time.time()
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        backward_times += time.time() - start_bwd
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            acc = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration {i}, accuracy: {acc:.4f}")
    end_time = time.time()
    print(f"Average forward propagation time: {forward_times / iterations:.6f}s")
    print(f"Average backward propagation time: {backward_times / iterations:.6f}s")
    print("Total training time: ", end_time - start_time)
    return W1, b1, W2, b2

iterations=100
# Train the network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, iterations)

# Make predictions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Test a single prediction
def test_prediction(index, W1, b1, W2, b2):
    make_predictions(X_train[:, index, None], W1, b1, W2, b2)

# Examples
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# Evaluate on dev set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

import jax
import jax.numpy as jnp
from jax import random, jit
import pandas as pd
import time

data = pd.read_csv('/kaggle/input/mnist-dataset/train.csv')

data = jnp.array(data.values, dtype=jnp.float32)
m, n = data.shape

key = random.PRNGKey(42)
key, subkey = random.split(key)
data = random.permutation(subkey, data)

data_dev = data[0:1000].T
Y_dev = data_dev[0].astype(jnp.int32)
X_dev = data_dev[1:n] / 255.

data_train = data[1000:m].T
Y_train = data_train[0].astype(jnp.int32)
X_train = data_train[1:n] / 255.
_, m_train = X_train.shape

def init_params(key):
    k1, k2 = random.split(key)
    scale = jnp.sqrt(2.0 / 784) 
    W1 = (random.uniform(k1, (256, 784)) * 2.0 - 1.0) * scale
    b1 = jnp.zeros((256, 1))
    W2 = (random.uniform(k2, (10, 256)) * 2.0 - 1.0) * scale
    b2 = jnp.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return jnp.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    A = jnp.exp(Z) / sum(jnp.exp(Z))
    return A

@jit
def forward_prop(W1, b1, W2, b2, X):
    Z1 = jnp.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = jnp.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    return jax.nn.one_hot(Y, 10).T

@jit
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m_train):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * jnp.dot(dZ2, A1.T)
    db2 = 1 / m_train * jnp.sum(dZ2, axis=1, keepdims=True)
    dZ1 = jnp.dot(W2.T, dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_train * jnp.dot(dZ1, X.T)
    db1 = 1 / m_train * jnp.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

@jit
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return jnp.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return jnp.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    forward_times, backward_times = 0, 0
    key = random.PRNGKey(0)
    W1, b1, W2, b2 = init_params(key)
    
    start_time = time.time()
    for i in range(iterations):
        start_fwd = time.time()
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        A2.block_until_ready() 
        forward_times += time.time() - start_fwd
        
        start_bwd = time.time()
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m_train)
        dW1.block_until_ready()
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

iterations = 200

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, iterations)

Z1, A1, Z2, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
dev_predictions = get_predictions(A2_dev)
acc = get_accuracy(dev_predictions, Y_dev)
print("Test accuracy: ", acc)
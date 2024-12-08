import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('neural-network/data/train.csv')
print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

print(m, n)

data_dev = data[0:1000].T

Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

def init_params():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1/784)
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1/10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1/20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1/784)

    return W1, b1, W2, b2

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z -= np.max(Z)
    return np.exp(Z) / sum(np.exp(Z))

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1

    return one_hot_Y.T

def deriv_ReLu(Z):
    return Z > 0

def back_propagation(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2 @ A1.T
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T @ dZ2 * deriv_ReLu(Z1)
    dW1 = 1 / m * dZ1 @ X.T
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
            if i % 100 == 0:
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                for j in range(5):
                    img = X[:, j].reshape(28, 28) * 255
                    axes[j].imshow(img, cmap='gray')
                    axes[j].set_title(f"Prediction: {get_predictions(A2)[j]}")
                    axes[j].axis('off')
                plt.show()
    
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)
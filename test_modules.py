import numpy as np
from neural_network import Linear, ReLU, Sequential, CrossEntropyLoss, SGD
from sklearn.datasets import make_blobs

def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        plus = f(x)
        x[idx] = old - eps
        minus = f(x)
        x[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad

def test_gradient_check():
    print("Running Gradient Check...")
    np.random.seed(42)
    mini_X = np.random.randn(5, 4)
    mini_y = np.eye(3)[np.random.randint(0, 3, 5)]
    
    linear = Linear(4, 3)
    relu = ReLU()
    loss_fn = CrossEntropyLoss()
    
    # Forward
    out1 = linear.forward(mini_X)
    out2 = relu.forward(out1)
    loss = loss_fn.loss(out2, mini_y)
    
    # Backward
    dloss = loss_fn.backward(mini_y)
    dout2 = relu.backward(dloss)
    _ = linear.backward(dout2)
    
    # Check W
    def f_W(W):
        linear.params['W'] = W
        o1 = linear.forward(mini_X)
        o2 = relu.forward(o1)
        return loss_fn.loss(o2, mini_y)
    
    num_grad_W = numerical_grad(f_W, linear.params['W'].copy())
    diff = np.linalg.norm(linear.grads['W'] - num_grad_W) / (np.linalg.norm(linear.grads['W']) + np.linalg.norm(num_grad_W))
    print(f"Linear W grad diff: {diff}")
    assert diff < 1e-7
    
    # Check b
    def f_b(b):
        linear.params['b'] = b
        o1 = linear.forward(mini_X)
        o2 = relu.forward(o1)
        return loss_fn.loss(o2, mini_y)
    
    num_grad_b = numerical_grad(f_b, linear.params['b'].copy())
    diff = np.linalg.norm(linear.grads['b'] - num_grad_b) / (np.linalg.norm(linear.grads['b']) + np.linalg.norm(num_grad_b))
    print(f"Linear b grad diff: {diff}")
    assert diff < 1e-7
    print("Gradient Check Passed!\n")

def test_convergence():
    print("Running Convergence Test...")
    np.random.seed(42)
    X, y = make_blobs(n_samples=500, centers=3, n_features=4, random_state=42)
    y_oh = np.eye(3)[y]
    
    model = Sequential([
        Linear(4, 16),
        ReLU(),
        Linear(16, 3)
    ])
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.layers, lr=0.1)
    
    initial_loss = loss_fn.loss(model.forward(X), y_oh)
    print(f"Initial loss: {initial_loss:.4f}")
    
    for epoch in range(100):
        # Forward
        scores = model.forward(X)
        loss = loss_fn.loss(scores, y_oh)
        
        # Backward
        dloss = loss_fn.backward(y_oh)
        model.backward(dloss)
        
        # Update
        optimizer.step()
        
    final_loss = loss_fn.loss(model.forward(X), y_oh)
    print(f"Final loss after 100 epochs: {final_loss:.4f}")
    
    assert final_loss < initial_loss
    print("Convergence Test Passed!")

if __name__ == "__main__":
    test_gradient_check()
    test_convergence()
